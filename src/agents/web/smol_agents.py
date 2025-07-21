import argparse
import json
import os
import pymupdf
import pymupdf4llm
import re
import requests
import tempfile
import time
import torch

from agents.web.utils import WebAgent, get_custom_arg_parser, NO_THINK
from bs4 import BeautifulSoup
from dataclasses import asdict, is_dataclass
from dotenv import load_dotenv
from fake_useragent import UserAgent
from llmlingua import PromptCompressor
from markdownify import markdownify
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
from requests.exceptions import RequestException
from smolagents import (
    ApiWebSearchTool,
    CodeAgent,
    LiteLLMModel,
    MLXModel,
    MultiStepAgent,
    Tool,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
    WikipediaSearchTool,
)
from smolagents.memory import PlanningStep
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    ChatMessageToolCallStreamDelta,
    MessageRole,
)
from smolagents.monitoring import TokenUsage
from typing import Any, Callable, Mapping, Tuple
from uuid import uuid4
from wrapt import wrap_function_wrapper

load_dotenv()

def should_add_new_tool_call(
    current_calls: list[ChatMessageToolCallStreamDelta],
    call_delta,
) -> bool:
    if len(current_calls) == 0:
        return True
    current_call = current_calls[-1]

    if (
        call_delta.id is not None and
        current_call.id is not None and
        call_delta.id != current_call.id
    ):
        return True

    if call_delta.function is not None:
        cur_name = current_call.function.name
        delta_name = call_delta.function.name
        if cur_name and delta_name != cur_name:
            return True

        current_arg = current_call.function.arguments
        delta_arg = call_delta.function.arguments
        if current_arg and delta_arg:
            if current_arg.strip().startswith("{"):
                try:
                    json.loads(current_arg)
                    return True
                except:
                    return False
    
    return False
        
# Fixing a bug with parallel function calling in agglomerate_stream_deltas
# See https://github.com/huggingface/smolagents/issues/1569
def agglomerate_litellm_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    """
    Agglomerate a list of stream deltas into a single stream delta.
    """
    accumulated_tool_calls: dict[int, list[ChatMessageToolCallStreamDelta]] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.tool_calls:
            for tool_call_delta in stream_delta.tool_calls:  # Normally there should be only one call at a time
                # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                idx = tool_call_delta.index
                if idx is not None:
                    # Check to see if a new tool call needs to be added
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = []
                    calls = accumulated_tool_calls[idx]
                    if should_add_new_tool_call(calls, tool_call_delta):
                        calls.append(ChatMessageToolCallStreamDelta(
                            id=tool_call_delta.id,
                            type=tool_call_delta.type,
                            function=ChatMessageToolCallFunction(name="", arguments=""),
                        ))

                    # Update the last tool call at the specific index if it's incomplete
                    tool_call = calls[-1]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    func = tool_call_delta.function
                    if func:
                        if func.name and len(func.name) > 0:
                            tool_call.function.name = func.name
                        if func.arguments:
                            tool_call.function.arguments += func.arguments
                else:
                    raise ValueError(f"Tool call index is not provided in tool delta: {tool_call_delta}")

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or str(uuid4()), # None IDs prevent parallel calls
                type="function",
            )
            for tool_call_stream_deltas in accumulated_tool_calls.values()
            for tool_call_stream_delta in tool_call_stream_deltas
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )

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


class _SmolLiteLLMGenerateWrapper:
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
            chat_msg = agglomerate_litellm_stream_deltas(deltas)
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
                wrapper=_SmolLiteLLMGenerateWrapper(tracer=self._tracer),
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
        assert (type(compression_ratio) is float and 0.0 < compression_ratio <= 1.0), "Invalid compression ratio"

        super().__init__()
        if compression_ratio < 1.0:
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
        if self.compression_ratio == 1.0:
            return text
        res = self.compressor.compress_prompt(
            text,
            rate=self.compression_ratio,
            force_tokens=['#', '\n', ',', '.', '-', '?'],
            force_reserve_digit=True,
            drop_consecutive=True,
        )
        return res["compressed_prompt"]


class CustomToolWithCompression():
    """Compress results and add /no_think for Qwen"""
    def __init__(self, *args, compression_ratio=1.0, add_no_think=False, **kwargs):
        self.compress_tool = TextCompressionTool(compression_ratio=compression_ratio)
        self.add_no_think = add_no_think
        super().__init__(*args, **kwargs)

    def forward(self, query):
        res = self._forward(query)
        res = self.compress_tool.forward(res)
        if self.add_no_think:
            res = f"{res} {NO_THINK}"
        return res

    def _forward(self, query):
        raise NotImplementedError("This method must be implemented!")


class CustomWebSearchTool(CustomToolWithCompression, WebSearchTool):
    """Compress each search result individually"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, query):
        res = self._forward(query)
        if self.add_no_think:
            res = f"{res} {NO_THINK}"
        return res

    def _forward(self, query):
        results = self.search(query)
        if len(results) == 0:
            return "No results found! Try a less restrictive/shorter query."

        res = "## Search Results\n\n" + "\n\n".join([
            f"[{r['title']}]({r['link']})\n{self.compress_tool.forward(r['description'])}"
            for r in results
        ])
        return res


class CustomApiWebSearchTool(CustomToolWithCompression, ApiWebSearchTool):
    """Compress each search result individually"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, query):
        res = self._forward(query)
        if self.add_no_think:
            res = f"{res} {NO_THINK}"
        return res

    def _forward(self, query):
        return ApiWebSearchTool.forward(self, query)

    def extract_results(self, data: dict) -> list:
        results = []
        for result in data.get("web", {}).get("results", []):
            desc = result.get("description", "")
            if len(desc) > 0:
                desc = BeautifulSoup(desc, "html.parser").get_text()
                desc = self.compress_tool.forward(desc)
            results.append(
                {"title": result["title"], "url": result["url"], "description": desc}
            )
        return results


class CustomWikipediaSearchTool(CustomToolWithCompression, WikipediaSearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _forward(self, query):
        return WikipediaSearchTool.forward(self, query)


class CustomVisitWebpageTool(CustomToolWithCompression, VisitWebpageTool):
    """Has better user-agent and ability to convert pdf to markdown"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ua = UserAgent(platforms="desktop").random

    def forward(self, url):
        return super().forward(url)

    def _forward(self, url):
        try:
            headers = {
                "User-Agent": self.ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()  # Raise an exception for bad status codes

            content_type = response.headers.get("Content-Type", "").lower()

            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                doc = pymupdf.open(stream=response.content, filetype="pdf")
                markdown_content = pymupdf4llm.to_markdown(doc).strip()
            elif "text/html" in content_type or "xml" in content_type:
                markdown_content = markdownify(response.text).strip()
            else:
                return f"This content type is not supported: {content_type}"

            # Clean up extra newlines
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            return self._truncate_content(markdown_content, self.max_output_length)
        
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


class BasicSmolAgent(WebAgent):
    def __init__(
        self,
        model_name,
        model_id,
        model_type,
        stream=False,
        api_key=None,
        api_base=None,
        thinking=False,
        **kwargs,
    ):
        super().__init__(model_name, model_id, model_type, stream=stream, **kwargs)
        self.thinking = thinking
        self.init_model(
            model_id,
            model_type,
            api_key=api_key,
            api_base=api_base,
            thinking=thinking
        )
        self.init_agent()
        
    def init_model(
        self,
        model_id,
        model_type,
        api_key=None,
        api_base=None,
        max_tokens=40000, # Qwen 3's context length limit
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
            flatten = None
            if "mlc" in model_id.lower():
                flatten = True
            self.model = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                api_base=api_base,
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=180,
                seed=47,
                flatten_messages_as_text=flatten,
            )
        else:
            raise NotImplementedError(f"{model_type} is not supported.")

    def init_agent(self):
        raise NotImplementedError("Agents must be specified.")
    
    def get_tools(self):
        raise NotImplementedError("Tools must be specified.")

    def should_add_no_think(self):
        return (
            self.model_type == "litellm" and
            "qwen3" in self.model_id.lower() and
            not self.thinking
        )

    def get_instrumentor(self):
        if self.instrumentor is None:
            self.instrumentor = CustomSmolagentsInstrumentor(
                self.model,
                self.stream,
                # add_pause=True,
            )
        return self.instrumentor

    def run(self, prompt, *args, **kwargs):
        return self.agent.run(prompt, *args, **kwargs)


class WebCodeAgent(BasicSmolAgent):
    def __init__(self, *args, compression_ratio=1.0, **kwargs):
        self.compression_ratio = compression_ratio
        super().__init__(self.__class__.__name__, *args, **kwargs)

    def init_agent(self):
        self.agent = CodeAgent(
            tools=self.get_tools(),
            model=self.model,
            instructions=NO_THINK if self.should_add_no_think() else None,
        )

    def get_tools(self):
        add_no_think = self.should_add_no_think()
        return [
            CustomApiWebSearchTool(
                compression_ratio=self.compression_ratio,
                add_no_think=add_no_think,
            ),
            CustomWikipediaSearchTool(
                user_agent="MyWebAgent (dpham@brave.com)",
                language="en",
                content_type="summary",
                extract_format="WIKI",
                compression_ratio=self.compression_ratio,
                add_no_think=add_no_think,
            ),
            CustomVisitWebpageTool(
                compression_ratio=self.compression_ratio,
                add_no_think=add_no_think,
                max_output_length=10000,
            ),
        ]


class WebToolCallingAgent(WebCodeAgent):
    def init_agent(self):
        self.agent = ToolCallingAgent(
            tools=self.get_tools(),
            model=self.model,
            instructions=NO_THINK if self.should_add_no_think() else None,
        )


class WebManagedAgent(WebCodeAgent):
    def init_agent(self):
        search_agent = ToolCallingAgent(
            tools=self.get_tools(),
            model=self.model,
            name="search_agent",
            description="This is an agent that can do web search and retrieve web pages.",
            instructions=NO_THINK if self.should_add_no_think() else None,
        )
        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            managed_agents=[search_agent],
        )


class AgentType:
    CODE = "code"
    TOOL = "tool"
    MANAGED = "managed"

AGENT_MAP = {
    AgentType.CODE: WebCodeAgent,
    AgentType.TOOL: WebToolCallingAgent,
    AgentType.MANAGED: WebManagedAgent,
}


if __name__ == "__main__":
    example_text = """
Examples:

Run with an MLX model:
python -m agents.web.smol_agents \\
--agent_type code \\
--model_id mlx-community/Qwen3-32B-4bit \\
--model_type mlx \\
--trace_path ./trace.json \\
--prompt "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"

Run with Anthropic via LiteLLM:
python -m agents.web.smol_agents \\
--agent_type tool \\
--model_id anthropic/claude-sonnet-4-20250514 \\
--model_type litellm \\
--trace_path ./trace.json \\
--prompt "Summarize the US Constitution." \\
--api_key_env ENV_FIELD_FOR_YOUR_API_KEY

Run with Ollama via LiteLLM with streaming:
python -m agents.web.smol_agents \\
--agent_type code \\
--model_id ollama_chat/qwen3:32b \\
--model_type litellm \\
--stream \\
--trace_path ./trace.json \\
--prompt "What is the meaning of life?"

Run with MLC-LLM via LiteLLM
(assuming you have run something like
`mlc_llm serve HF://mlc-ai/Qwen3-32B-q4f16_1-MLC \\
    --overrides "gpu_memory_utilization=1.0;prefill_chunk_size=1024"`:

python -m agents.web.smol_agents \\
--agent_type code \\
--model_id openai/HF://mlc-ai/Qwen3-32B-q4f16_1-MLC \\
--model_type litellm \\
--api_base http://127.0.0.1:8000/v1 \\
--stream \\
--trace_path ./trace.json \\
--prompt "How to learn to vocal fry?"
"""

    parser = get_custom_arg_parser(description="Run a SmolAgent with tracing.", example_text=example_text)
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=list(AGENT_MAP.keys()),
        help=f"Type of agent.",
    )
    parser.add_argument(
        "--planning_interval", type=int, default=None, help="Planning interval."
    )
    parser.add_argument("--api_base", type=str, default=None, help="API base for LiteLLM.")
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=1.0,
        help="Compression ratio for all results from the web."
    )
    args = parser.parse_args()

    agent = AGENT_MAP[args.agent_type](
        model_id=args.model_id,
        model_type=args.model_type,
        stream=args.stream,
        thinking=args.thinking,
        api_key=os.getenv(args.api_key_env) if args.api_key_env else "FAKE_KEY",
        api_base=args.api_base,
        compression_ratio=args.compression_ratio,
    )
    if args.trace_path is not None:
        agent.enable_tracing(args.trace_path)
    agent.run(args.prompt, max_steps=10)
