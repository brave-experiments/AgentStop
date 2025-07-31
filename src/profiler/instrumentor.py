
import json
import time

from dataclasses import asdict, is_dataclass
from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.helpers import safe_json_dumps
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from openinference.instrumentation.smolagents._wrappers import (
    AGENT,
    CHAIN,
    INPUT_VALUE,
    LLM,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_VALUE,
    _bind_arguments,
    _flatten,
    _get_input_value,
    _smolagent_run_attributes,
    _ModelWrapper,
)
from opentelemetry import trace
from smolagents import (
    LiteLLMModel,
    MLXModel,
    MultiStepAgent,
)
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


class _RunStreamWrapper: 
    '''Base on _RunWrapper'''
    def __init__(self, tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ):
        span_name = f"{instance.__class__.__name__}.run"
        agent = instance
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(wrapped, *args, **kwargs),
                        **dict(_smolagent_run_attributes(agent, arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            run_stream = wrapped(*args, **kwargs)
            for step in run_stream:
                yield step

            input_cnt = agent.monitor.total_input_token_count
            output_cnt = agent.monitor.total_output_token_count
            span.set_attribute(LLM_TOKEN_COUNT_PROMPT, input_cnt)
            span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, output_cnt)
            span.set_attribute(LLM_TOKEN_COUNT_TOTAL, input_cnt + output_cnt)
            span.set_status(trace.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, str(step))


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
    def __init__(self, model, stream_llm=False, stream_run=False, add_pause=False):
        super().__init__()
        self.model = model
        self.stream_llm = stream_llm # Stream the LLM output
        self.stream_run = stream_run # Stream the agent's run step
        self.add_pause = add_pause

    def _instrument(self, **kwargs):
        super()._instrument(**kwargs)

        self._original_run_stream = getattr(MultiStepAgent, "_run_stream", None)
        wrap_function_wrapper(
            module="smolagents",
            name=f"{MultiStepAgent.__name__}._run_stream",
            wrapper=_GeneratorStepWrapper(tracer=self._tracer),
        )

        if self.stream_run:
            # Undo the original instrumentation wrap
            if self._original_run_method is not None:
                MultiStepAgent.run = self._original_run_method
            wrap_function_wrapper(
                module="smolagents",
                name="MultiStepAgent.run",
                wrapper=_RunStreamWrapper(tracer=self._tracer),
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
        elif self.stream_llm and isinstance(self.model, LiteLLMModel):
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