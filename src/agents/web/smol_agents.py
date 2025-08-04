import importlib.resources
import json
import os
import time
import yaml

from dotenv import load_dotenv
from profiler.instrumentor import CustomSmolagentsInstrumentor
from rich.rule import Rule
from rich.text import Text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    MLXModel,
    ToolCallingAgent,
)
from smolagents.agents import RunResult
from smolagents.memory import ActionStep, PlanningStep
from smolagents.monitoring import LogLevel, Timing
from .utils import (
    create_tools,
    get_custom_arg_parser,
    NO_THINK,
    WebAgent,
)

load_dotenv()

def create_model(
        model_id,
        model_type="litellm",
        api_key=None,
        api_base=None,
        context_size=40960, # Ollama Qwen 3's context length limit
        max_tokens=512, # Max number of tokens to generate per LLM call
        thinking=False,
        temperature=0.0,
        top_p=1.0,
        top_k=20,
        min_p=0.0,
        **kwargs,
    ):
    if model_type == "mlx": # MLX is customized for Apple silicon
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        return MLXModel(
            model_id=model_id,
            trust_remote_code=True,
            max_tokens=max_tokens,
            apply_chat_template_kwargs={
                "enable_thinking": thinking,
            },
            sampler=sampler,
        )
    elif model_type == "litellm":
        flatten = None
        if "mlc" in model_id.lower():
            flatten = True
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            num_ctx=context_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            timeout=300, # Should be big enough to accomodate large models
            seed=47,
            flatten_messages_as_text=flatten,
        )
    else:
        raise NotImplementedError(f"{model_type} is not supported.")

class BasicSmolAgent(WebAgent):
    def __init__(
        self,
        model_id,
        model_type="litellm",
        stream_llm=False,
        stream_run=False,
        api_key=None,
        api_base=None,
        thinking=False,
        planning_interval=None,
        **kwargs,
    ):
        super().__init__("SmolAgent", model_id, model_type=model_type, stream_llm=stream_llm, **kwargs)
        self.thinking = thinking
        self.stream_run = stream_run
        self.model = create_model(
            model_id,
            model_type,
            api_key=api_key,
            api_base=api_base,
            thinking=thinking,
            **kwargs,
        )
        self.should_add_no_think = (
            self.model_type == "litellm" and
            "qwen3" in self.model_id.lower() and
            "instruct" not in self.model_id.lower() and
            not self.thinking
        )
        self.tools = create_tools(self.should_add_no_think, self.compression_ratio)
        self.planning_interval = planning_interval
        self.init_agent()

    def init_agent(self):
        raise NotImplementedError("Agents must be specified.")

    def reset_tools(self):
        for t in self.tools:
            t.reset_history()

    def should_add_no_think(self):
        return (
            self.model_type == "litellm" and
            "qwen3" in self.model_id.lower() and
            not self.thinking
        )

    def get_prompt_template(self, agent_type=None):
        if not self.should_add_no_think:
            return None # Default templates
        if agent_type == "code":
            file_path = "code_agent.yaml"
        elif agent_type == "tool":
            file_path = "toolcalling_agent.yaml"
        else:
            raise Exception("Invalid agent type for retrieving default templates.")
        
        template = yaml.safe_load(importlib.resources.files("smolagents.prompts").joinpath(file_path).read_text())
        fields = {
            "system_prompt": None,
            "planning": ["initial_plan", "update_plan_post_messages"],
            "managed_agent": ["task"],
        }
        for field, subfields in fields.items():
            if subfields is None:
                template[field] = template[field] + " " + NO_THINK
            else:
                for subfield in subfields:
                    template[field][subfield] = template[field][subfield] + " " + NO_THINK
        
        return template

    def get_instrumentor(self):
        if self.instrumentor is None:
            self.instrumentor = CustomSmolagentsInstrumentor(
                self.model,
                stream_llm=self.stream_llm,
                stream_run=self.stream_run,
                add_pause=False,
            )
        return self.instrumentor

    def get_current_results(self):
        return RunResult(
            output=None,
            token_usage=None,
            messages=self.agent.memory.get_full_steps(),
            timing=None,
            state=None
        )

    def run(self, prompt, **kwargs):
        return self.agent.run(prompt, **kwargs)


class WebCodeAgent(BasicSmolAgent):
    def __init__(self, *args, compression_ratio=1.0, **kwargs):
        self.compression_ratio = compression_ratio
        super().__init__(*args, **kwargs)

    def init_agent(self):
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            prompt_templates=self.get_prompt_template("code"),
            planning_interval=self.planning_interval,
            return_full_result=True,
        )


class WebToolCallingAgent(WebCodeAgent):
    def init_agent(self):
        self.agent = ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            prompt_templates=self.get_prompt_template("tool"),
            planning_interval=self.planning_interval,
            return_full_result=True,
        )


class WebManagedAgent(WebCodeAgent):
    def init_agent(self):
        search_agent = ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            prompt_templates=self.get_prompt_template("tool"),
            name="search_agent",
            description="This is an agent that can do web search and retrieve web pages.",
        )
        self.agent = CodeAgent(
            tools=[],
            model=self.model,
            prompt_templates=self.get_prompt_template("code"),
            managed_agents=[search_agent],
            planning_interval=self.planning_interval,
            return_full_result=True,
        )


class BasicCascadeAgent(WebCodeAgent):
    def __init__(
        self,
        model_ids,
        model_type,
        should_compress_context=False,
        **kwargs,
    ):
        assert len(model_ids) >= 2
        super().__init__(stream_run=True, **kwargs)
        self.model_ids = model_ids
        self.model_idx = 0
        self.should_compress_context = should_compress_context

    def cascade(self):
        if self.model_idx == len(self.model_ids) - 1:
            print("Currently using the last model in the cascade.")
            return
        
        self.model_idx += 1
        self.model_id = self.model_ids[self.model_idx]
        print(f"\n*** Cascading to model {self.model_id} ***\n")
        self.model = create_model(
            self.model_id,
            self.model_type,
            api_key=self.model.api_key,
            api_base=self.model.api_base,
            thinking=self.thinking,
            **self.kwargs,
        )
        self.agent.model = self.model

    def run(self, prompt, **kwargs):
        for step in self.agent.run(prompt, stream=True, **kwargs):
            if self.should_cascade(step):
                if self.should_compress_context:
                    self.compress_context(original_task=prompt, **kwargs)
                self.cascade()
        return step

    def should_cascade(self, step):
        raise NotImplementedError("You need to implement the cascading criteria.")

    def compress_context(self, original_task=None):
        raise NotImplementedError("You need to implement the context compression method.")


class FixedCascadeAgent(BasicCascadeAgent):
    '''
        Only cascade exactly once at a fixed step number.
        If cascade_step is 1, the model will be cascaded after step 1 is finished.
        Step number starts at 1.
    '''
    def __init__(self, cascade_step, *args, **kwargs):
        assert isinstance(cascade_step, int) and cascade_step >= 1
        super().__init__(*args, **kwargs)
        self.cascade_step = cascade_step
    
    def should_cascade(self, step):
        return (
            isinstance(step, ActionStep) and
            not step.is_final_answer and
            step.step_number == self.cascade_step
        )


class FixedCascadeAgentWithCompression(FixedCascadeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, should_compress_context=True, **kwargs)

    def compress_context(self, original_task, **kwargs):
        start_time = time.time()
        summary = []
        idx = 0
        all_tool_calls = set()
        for step in self.agent.memory.steps:
            if not (isinstance(step, ActionStep) and step.error is None and step.tool_calls is not None):
                continue
            step_dict = step.dict()
            tool_calls = []
            for tool_call in step_dict["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                tool_call_str = f"{tool_name}_{tool_args}"
                if tool_call_str not in all_tool_calls:
                    all_tool_calls.add(tool_call_str)
                    tool_calls.append({"name": tool_name, "arguments": tool_args})

            if len(tool_calls) > 0:
                tool_output = " ".join(w for w in step_dict["observations"].split(" ") if w not in ENGLISH_STOP_WORDS)
                summary.append({
                    "step_number": idx,
                    "tool_calls": tool_calls,
                    "tool_response": tool_output,
                })
                idx += 1
        
        plan = f"""Below is a brief JSON-formatted report of a previous partial attempt at solving the task.
I need to consider the report to determine what I should or should not do in my own attempt at the task.
If there is sufficient information from the report to solve the task, then I should attempt to solve it.
I need to avoid repeating what has already been tried in the report.

<summary>
{json.dumps(summary, indent=2)}
</summary>
"""
        self.agent.logger.log(Rule(f"[bold]Context compressions", style="orange"), Text(plan), level=LogLevel.INFO)
        planning_step = PlanningStep(
            model_input_messages=[],
            plan=plan,
            model_output_message=None,
            token_usage=None,
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

        # Edit agent's memory to remove all previous steps except the task step
        self.agent.memory.steps = self.agent.memory.steps[:1]
        self.agent.memory.steps.append(planning_step)


class AgentType:
    CODE = "code"
    TOOL = "tool"
    MANAGED = "managed"
    FIXED_CASCADE = "fixed_cascade"
    FIXED_CASCADE_COMPRESS = "fixed_cascade_compress"

AGENT_MAP = {
    AgentType.CODE: WebCodeAgent,
    AgentType.TOOL: WebToolCallingAgent,
    AgentType.MANAGED: WebManagedAgent,
    AgentType.FIXED_CASCADE: FixedCascadeAgent,
    AgentType.FIXED_CASCADE_COMPRESS: FixedCascadeAgentWithCompression,
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
    parser.add_argument("--planning_interval", type=int, default=None, help="Planning interval.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps for the agent.")
    parser.add_argument("--fixed_cascade_step", type=int, default=None, help="Fixed step number to cascade if using fixed cascade agent.")
    parser.add_argument("--api_base", type=str, default=None, help="API base for LiteLLM.")
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=1.0,
        help="Compression ratio for all results from the web."
    )
    args = parser.parse_args()

    agent = AGENT_MAP[args.agent_type](
        model_id=args.model_id[0],
        model_ids=args.model_id,
        model_type=args.model_type,
        stream_llm=args.stream,
        thinking=args.thinking,
        api_key=os.getenv(args.api_key_env) if args.api_key_env else "FAKE_KEY",
        api_base=args.api_base,
        compression_ratio=args.compression_ratio,
        planning_interval=args.planning_interval,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        cascade_step=args.fixed_cascade_step,
    )
    if args.trace_path is not None:
        agent.enable_tracing(args.trace_path)
    agent.run(args.prompt, max_steps=args.max_steps)
