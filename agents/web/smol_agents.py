import argparse
import json
import os
import time
import torch

from dataclasses import asdict, is_dataclass
from dotenv import load_dotenv
from llmlingua import PromptCompressor
from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.helpers import safe_json_dumps
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from openinference.instrumentation.smolagents._wrappers import (
    CHAIN,
    INPUT_VALUE,
    LLM,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_VALUE,
    _get_input_value,
    _ModelWrapper,
)
from opentelemetry import trace
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    MLXModel,
    MultiStepAgent,
    Tool,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)
from smolagents.memory import PlanningStep
from smolagents.models import agglomerate_stream_deltas
from typing import Any, Callable, Mapping, Tuple
from utils import WebAgent, get_custom_arg_parser, NO_THINK
from wrapt import wrap_function_wrapper


class _GeneratorStepWrapper:
    """Wrap each step of a generator in its own span"""
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        ended = False
        while not ended:
            with self._tracer.start_as_current_span(
                "Step",
                attributes={OPENINFERENCE_SPAN_KIND: CHAIN},
            ) as span:
                try:
                    item = next(generator)
                    yield item
                    span.update_name(f"{item.__class__.__name__}")
                    output_val = None
                    if is_dataclass(item):
                        output_val = item.dict() if hasattr(item, "dict") else asdict(item)
                    span.set_attribute(OUTPUT_VALUE, safe_json_dumps(output_val))
                except StopIteration:
                    ended = True
                    span.set_attribute("generator_ended", True)
                finally:
                    span.set_status(trace.StatusCode.OK)


class _SmolAgentModelGenerateWrapper:
    """Replace generate to use generate_stream internally to get token-level data"""
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any], # This is never used
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        with self._tracer.start_as_current_span(
            f"{instance.__class__.__name__}.generate_stream",
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
            },
        ) as span:
            gen_stream = instance.generate_stream(*args, **kwargs)
            deltas = []
            prefill_tps = None
            generation_tps = None
            first_item_ts = None
            tic = time.perf_counter()
            
            for delta in gen_stream:
                if first_item_ts is None:
                    prefill_time = time.perf_counter() - tic
                    first_item_ts = time.time_ns()
                    tic = time.perf_counter()
                deltas.append(delta)
            
            generation_time = time.perf_counter() - tic
            chat_msg = agglomerate_stream_deltas(deltas)
            prefill_tps = chat_msg.token_usage.input_tokens / prefill_time
            generation_tps = chat_msg.token_usage.output_tokens / generation_time

            span.set_attribute("prefill_tps", prefill_tps)
            span.set_attribute("generation_tps", generation_tps)
            span.set_attribute("first_token_ts", first_item_ts)
            span.set_status(trace.StatusCode.OK)
                    
            return chat_msg


class CustomSmolagentsInstrumentor(SmolagentsInstrumentor):
    """Add step-by-step tracing for smolagents"""

    def __init__(self, model, stream=False, add_pause=False):
        super().__init__()
        self.model = model
        self.stream = stream
        self.add_pause = add_pause

    def _instrument(self, **kwargs):
        super()._instrument(**kwargs)

        self._original_run_stream = getattr(MultiStepAgent, "_run_stream", None)
        wrap_function_wrapper(
            module="smolagents",
            name=f"{MultiStepAgent.__name__}._run_stream",
            wrapper=_GeneratorStepWrapper(tracer=self._tracer),
        )

        if isinstance(self.model, MLXModel): # MLX is always streaming
            self._original_mlxmodel_stream_generate = self.model.stream_generate
            def custom_mlx_stream_generate(*args, **kwargs):
                if self.add_pause:
                    with self._tracer.start_as_current_span(
                        "Pause",
                        attributes={
                            OPENINFERENCE_SPAN_KIND: CHAIN,
                        },
                    ) as span:
                        time.sleep(90)
                        span.set_status(trace.StatusCode.OK)
                with self._tracer.start_as_current_span(
                    f"{MLXModel.__name__}.stream_generate",
                    attributes={
                        OPENINFERENCE_SPAN_KIND: LLM,
                    },
                ) as span:
                    first_item_ts = None
                    last_item = None
                    try:
                        for item in self._original_mlxmodel_stream_generate(*args, **kwargs):
                            if first_item_ts is None:
                                first_item_ts = time.time_ns()
                            last_item = item
                            yield item
                    finally:
                        if last_item is not None:
                            span.set_attribute("prefill_tps", last_item.prompt_tps)
                            span.set_attribute("generation_tps", last_item.generation_tps)
                            span.set_attribute("first_token_ts", first_item_ts)
                        span.set_status(trace.StatusCode.OK)

            self.model.stream_generate = custom_mlx_stream_generate
        elif self.stream and isinstance(self.model, LiteLLMModel):
            # Undo super's wrap
            setattr(LiteLLMModel, "generate", self._original_model_generate_methods[LiteLLMModel])
            
            # Wrap our replacement
            wrap_function_wrapper(
                module="smolagents",
                name=f"{LiteLLMModel.__name__}.generate",
                wrapper=_SmolAgentModelGenerateWrapper(tracer=self._tracer),
            )

            # Redo the super's wrap
            wrap_function_wrapper(
                module="smolagents",
                name=f"{LiteLLMModel.__name__}.generate",
                wrapper=_ModelWrapper(tracer=self._tracer),
            )


    def _uninstrument(self, **kwargs):
        super()._uninstrument(**kwargs)
        if self._original_run_stream is not None:
            setattr(MultiStepAgent, "_run_stream", self._original_run_stream)
            self._original_run_stream = None

        if self._original_mlxmodel_stream_generate is not None:
            self.model = self._original_mlxmodel_stream_generate
            self._original_mlxmodel_stream_generate = None

        if self._original_model_generate_methods is not None:
            setattr(LiteLLMModel, "generate", self._original_model_generate_methods[LiteLLMModel])


class BasicSmolAgent(WebAgent):
    def __init__(
        self,
        model_name,
        model_id,
        model_type,
        stream=False,
        api_key=None,
        thinking=False,
        **kwargs,
    ):
        super().__init__(model_name, model_id, model_type, stream=stream, **kwargs)
        self.thinking = thinking
        self.init_model(model_id, model_type, api_key=api_key, thinking=thinking)
        self.init_agent()
        
    def init_model(
        self,
        model_id,
        model_type,
        api_key=None,
        max_tokens=20000,
        thinking=False,
    ):
        if model_type == "mlx":
            # MLX is customized for Apple silicon
            # Default is greedy sampling
            self.model = MLXModel(
                model_id=model_id,
                trust_remote_code=True,
                max_tokens=max_tokens,
                apply_chat_template_kwargs={
                    "enable_thinking": thinking,
                },
            )
        elif model_type == "litellm":
            self.model = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=0.0,
                think=thinking,
            )
        else:
            raise NotImplemented(f"{model_type} is not supported.")

    def init_agent(self):
        self.search_agent = ToolCallingAgent(
            tools=[WebSearchTool()],
            model=self.model,
            name="search_agent",
            description="This is an agent that can do web search.",
        )

        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            managed_agents=[self.search_agent],
        )

    def get_instrumentor(self):
        if self.instrumentor is None:
            self.instrumentor = CustomSmolagentsInstrumentor(
                self.model,
                self.stream,
                # add_pause=True,
            )
        return self.instrumentor

    def run(self, prompt):
        self.agent.run(prompt)


class TextCompressionTool(Tool):
    name = "text_compression"
    description = "Compresses texts like web search results and returns a string for the compressed text."
    inputs = {"text": {"type": "string", "description": "The text to compress."}}
    output_type = "string"

    def __init__(
        self,
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device=None,
        compression_ratio=0.7,
    ):
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.compressor = PromptCompressor(
            model_name=model_name,
            device_map=device,
            use_llmlingua2=True
        )
        self.compression_ratio = compression_ratio

    def forward(self, text):
        res = self.compressor.compress_prompt(
            text,
            rate=self.compression_ratio,
            force_tokens=['#', '\n', ',', '.', '-', '?'],
            force_reserve_digit=True,
            drop_consecutive=True,
        )
        return res["compressed_prompt"]


class WebSearchCompressTool(WebSearchTool):
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, compression_ratio=0.7, add_no_think=False):
        super().__init__()
        self.compress_tool = TextCompressionTool(compression_ratio=compression_ratio)
        self.add_no_think = add_no_think

    def forward(self, query):
        results = self.search(query)
        if len(results) == 0:
            msg = "No results found! Try a less restrictive/shorter query."
            if self.add_no_think:
                msg = f"{msg} {NO_THINK}"
            raise Exception(msg)
        res = "## Search Results\n\n" + "\n\n".join([
            f"{r['title']}\n{self.compress_tool.forward(r['description'])}"
            for r in results
        ])
        if self.add_no_think:
            return f"{res} {NO_THINK}"
        else:
            return res


class SmolAgentWithCompression(BasicSmolAgent):
    def __init__(self, *args, **kwargs):
        super().__init__("SmolAgentWithCompression", *args, **kwargs)
        
    def init_agent(self):
        instructions = None
        add_no_think = (
            self.model_type == "litellm" and
            self.model_id.startswith("ollama_chat/qwen3") and
            not self.thinking
        )
        
        if add_no_think:
            instructions = NO_THINK
        
        custom_search_tool = WebSearchCompressTool(
            compression_ratio=0.5,
            add_no_think=add_no_think,
        )
        self.agent = CodeAgent(
            tools=[custom_search_tool],
            model=self.model,
            instructions=instructions,
        )


if __name__ == "__main__":
    load_dotenv()

    example_text = """
Examples:

Run with an MLX model:
python smol_agents.py \\
--model_id mlx-community/Qwen3-32B-4bit \\
--model_type mlx \\
--trace_path ./trace.json \\
--prompt "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"

Run with Anthropic via LiteLLM:
python smol_agents.py \\
--model_id anthropic/claude-sonnet-4-20250514 \\
--model_type litellm \\
--trace_path ./trace.json \\
--prompt "Summarize the US Constitution." \\
--api_key_env ENV_FIELD_FOR_YOUR_API_KEY

Run with Ollama via LiteLLM with streaming:
python smol_agents.py \\
--model_id ollama_chat/qwen3:32b \\
--model_type litellm \\
--stream \\
--trace_path ./trace.json \\
--prompt "Summarize the US Constitution."
"""

    parser = get_custom_arg_parser(description="Run a SmolAgent with tracing.", example_text=example_text)
    parser.add_argument(
        "--planning_interval", type=int, default=None, help="Planning interval."
    )
    args = parser.parse_args()

    agent = SmolAgentWithCompression(
        model_id=args.model_id,
        model_type=args.model_type,
        stream=args.stream,
        thinking=args.thinking,
        api_key=os.getenv(args.api_key_env) if args.api_key_env is not None else None,
    )
    if args.trace_path is not None:
        agent.enable_tracing(args.trace_path)
    agent.run(args.prompt)
