import argparse
import json
import os

from dotenv import load_dotenv
from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.helpers import safe_json_dumps
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from openinference.instrumentation.smolagents._wrappers import (
    OPENINFERENCE_SPAN_KIND,
    CHAIN,
    INPUT_VALUE,
    OUTPUT_VALUE,
    _get_input_value,
)
from opentelemetry import trace
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    MLXModel,
    MultiStepAgent,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)
from smolagents.memory import PlanningStep
from typing import Any, Callable, Mapping, Tuple
from utils import WebAgent, get_custom_arg_parser
from wrapt import wrap_function_wrapper


class _GeneratorStepWrapper:
    def __init__(self, tracer, is_planning_step=False):
        self._tracer = tracer
        self._is_planning_step = is_planning_step

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        span_name = (
            PlanningStep.__name__
            if self._is_planning_step
            else args[0].__class__.__name__
        )
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: CHAIN,
                INPUT_VALUE: _get_input_value(wrapped, *args, **kwargs),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            last_item = None
            for item in wrapped(*args, **kwargs):
                yield item
                last_item = item
            if self._is_planning_step:
                assert isinstance(last_item, PlanningStep)
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(last_item.dict()))
            else:
                step_log = args[0]  # ActionStep
                span.set_attribute(OUTPUT_VALUE, safe_json_dumps(step_log.dict()))
                if step_log.error is not None:
                    span.record_exception(step_log.error)
            span.set_status(trace.StatusCode.OK)


class CustomSmolagentsInstrumentor(SmolagentsInstrumentor):
    """Add step-by-step tracing for smolagents"""

    def _instrument(self, **kwargs):
        super()._instrument(**kwargs)

        self._original_execute_step = getattr(MultiStepAgent, "_execute_step", None)
        wrap_function_wrapper(
            module="smolagents",
            name=f"{MultiStepAgent.__name__}._execute_step",
            wrapper=_GeneratorStepWrapper(tracer=self._tracer),
        )

        self._original_generate_planning_step = getattr(
            MultiStepAgent, "_generate_planning_step", None
        )
        wrap_function_wrapper(
            module="smolagents",
            name=f"{MultiStepAgent.__name__}._generate_planning_step",
            wrapper=_GeneratorStepWrapper(tracer=self._tracer, is_planning_step=True),
        )

    def _uninstrument(self, **kwargs):
        super()._uninstrument(**kwargs)
        if self._original_execute_step is not None:
            setattr(MultiStepAgent, "_execute_step", self._original_execute_step)
            self._original_execute_step = None

        if self._original_generate_planning_step is not None:
            setattr(
                MultiStepAgent, "_execute_step", self._original_generate_planning_step
            )
            self._original_generate_planning_step = None


class SmolAgent(WebAgent):
    def __init__(
        self,
        model_id,
        model_type,
        max_tokens=20000,
        planning_interval=None,
        api_key=None,
    ):
        super().__init__("SmolAgents")
        if model_type == "mlx":  # For Apple silicon
            self.model = MLXModel(
                model_id=model_id,
                trust_remote_code=True,
                max_tokens=max_tokens,
            )
        elif model_type == "litellm":
            self.model = LiteLLMModel(model_id=model_id, api_key=api_key)
        else:
            raise NotImplemented(f"{model_type} is not supported.")

        self.search_agent = ToolCallingAgent(
            tools=[WebSearchTool(), VisitWebpageTool()],
            model=self.model,
            name="search_agent",
            description="This is an agent that can do web search.",
        )

        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            managed_agents=[self.search_agent],
            planning_interval=planning_interval,
        )

    def get_instrumentor(self):
        if self.instrumentor is None:
            self.instrumentor = CustomSmolagentsInstrumentor()
        return self.instrumentor

    def run(self, prompt):
        self.agent.run(prompt)


if __name__ == "__main__":
    load_dotenv()

    example_text = """
Examples:

Run with an MLX model:
python smol_agents.py
--model_id mlx-community/Qwen2.5-Coder-32B-Instruct-4bit \\
--model_type mlx \\
--trace_path ./trace.json \\
--prompt "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"

Run with a LiteLLM model:
python smol_agents.py
--model_id anthropic/claude-sonnet-4-20250514 \\
--model_type litellm \\
--trace_path ./trace.json \\
--prompt "Summarize the US Constitution." \\
--api_key_env ENV_FIELD_FOR_YOUR_API_KEY
"""

    parser = get_custom_arg_parser(description="Run a SmolAgent with tracing.", example_text=example_text)
    parser.add_argument(
        "--planning_interval", type=int, default=None, help="Planning interval."
    )
    args = parser.parse_args()

    agent = SmolAgent(
        model_id=args.model_id,
        model_type=args.model_type,
        api_key=os.getenv(args.api_key_env) if args.api_key_env is not None else None,
    )
    if args.trace_path is not None:
        agent.enable_tracing(args.trace_path)
    agent.run(args.prompt)
