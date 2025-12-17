import json
import importlib.resources
import time
import types
import yaml

from efficient_agents.agents.llm_backend import LlmBackend
from efficient_agents.agents.utils import (
    create_model,
    get_json_exporter,
    NO_THINK,
)
from efficient_agents.profiler.instrumentor import CustomSmolagentsInstrumentor
from rich.rule import Rule
from rich.text import Text
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from smolagents import CodeAgent
from smolagents.agents import RunResult
from smolagents.models import ChatMessage, MessageRole
from smolagents.memory import ActionStep, PlanningStep
from smolagents.monitoring import LogLevel, Timing


class BaseAgent:
    def __init_subclass__(cls, key=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "subclasses"):
            cls.subclasses = {}

        reg_key = key or cls.__name__
        if reg_key in cls.subclasses:
            raise ValueError(f"Duplicate key '{reg_key}' registered")
        cls.subclasses[reg_key] = cls

    @classmethod
    def create(cls, key, *args, **kwargs):
        if key not in cls.subclasses:
            raise ValueError(f"No backend registered under key '{key}'")
        return cls.subclasses[key](*args, **kwargs)

    def __init__(
        self,
        model_id: str,
        model_type="litellm",
        stream_llm=False,
        stream_run=False,
        api_key=None,
        api_base=None,
        thinking=False,
        planning_interval=None,
        compression_ratio=1.0,
        **kwargs,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.stream_llm = stream_llm
        self.instrumentor = None
        self.kwargs = kwargs
        self.thinking = thinking
        self.stream_run = stream_run
        self.compression_ratio = compression_ratio
        self.planning_interval = planning_interval
        self.should_add_no_think = (
            self.model_type == "litellm"
            and "instruct" not in self.model_id.lower()
            and not self.thinking
        )

        self.model = create_model(
            model_id,
            model_type,
            api_key=api_key,
            api_base=api_base,
            thinking=thinking,
            **kwargs,
        )
        self.tools = self.get_tools()
        self.agent = self.get_agent()

    def get_agent(self):
        return CodeAgent(
            tools=self.tools,
            model=self.model,
            prompt_templates=self.get_prompt_template("code"),
            planning_interval=self.planning_interval,
            return_full_result=True,
        )

    def get_tools(self):
        return []

    def reset_tools(self):
        for t in self.tools:
            t.reset_history()

    def get_prompt_template(self, agent_type=None):
        if not self.should_add_no_think:
            return None  # Default templates
        if agent_type == "code":
            file_path = "code_agent.yaml"
        elif agent_type == "tool":
            file_path = "toolcalling_agent.yaml"
        else:
            raise Exception("Invalid agent type for retrieving default templates.")

        template = yaml.safe_load(
            importlib.resources.files("smolagents.prompts")
            .joinpath(file_path)
            .read_text()
        )
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
                    template[field][subfield] = (
                        template[field][subfield] + " " + NO_THINK
                    )

        return template

    def enable_tracing(self, output_path):
        self.get_instrumentor().instrument(
            tracer_provider=get_json_exporter("Agent", output_path)
        )

    def disable_tracing(self):
        self.get_instrumentor().uninstrument()

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
            timing=Timing(start_time=time.time()),
            state="success",
        )

    def run(self, prompt, **kwargs):
        if self.agent is None:
            raise Exception("Agent is not initialized")
        return self.agent.run(prompt, **kwargs)


class BasicCascadeAgent(BaseAgent, key="basic_cascade"):
    def __init__(
        self,
        model_ids,
        should_compress_context=False,
        llm_backend=None,
        **kwargs,
    ):
        assert len(model_ids) >= 2
        super().__init__(stream_run=True, **kwargs)
        self.model_ids = model_ids
        self.model_idx = 0
        self.should_compress_context = should_compress_context
        self.llm_backend = LlmBackend.create(llm_backend) if llm_backend is not None else None

    def cascade(self):
        if self.model_idx == len(self.model_ids) - 1:
            print("Currently using the last model in the cascade.")
            return
        
        self.model_idx += 1
        self.model_id = self.model_ids[self.model_idx]
        print(f"\n*** Cascading to model {self.model_id} ***\n")
        if self.llm_backend is not None: # Ollama doesn't need this step
            model_id = self.model_id
            if self.model_type == "litellm" and model_id.count("/") > 1 and model_id.startswith("openai/"):
                model_id = model_id.split("/", 1)[1]
            self.llm_backend.start(model_id)
        
        self.model = create_model(
            self.model_id,
            self.model_type,
            api_key=getattr(self.model, "api_key", None),
            api_base=getattr(self.model, "api_base", None),
            thinking=self.thinking,
            **self.kwargs,
        )
        self.agent.model = self.model

    def run(self, prompt, **kwargs):
        agent_run = self.agent.run(prompt, stream=True, **kwargs)
        assert isinstance(agent_run, types.GeneratorType)
        for step in agent_run:
            if self.model_idx < len(self.model_ids) - 1 and self.should_cascade(step):
                if self.should_compress_context:
                    self.compress_context(original_task=prompt, **kwargs)
                self.cascade()
        return step

    def should_cascade(self, step):
        raise NotImplementedError("You need to implement the cascading criteria.")

    def compress_context(self, original_task=None, **kwargs):
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
            model_output_message=ChatMessage(MessageRole.ASSISTANT),
            token_usage=None,
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

        # Edit agent's memory to remove all previous steps except the task step
        self.agent.memory.steps = self.agent.memory.steps[:1]
        self.agent.memory.steps.append(planning_step)


class FixedCascadeAgent(BasicCascadeAgent, key="fixed_cascade"):
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


class FixedCascadeAgentWithCompression(FixedCascadeAgent, key="fixed_cascade_compress"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, should_compress_context=True, **kwargs)


class LogProbsCascadeAgent(BasicCascadeAgent, key="logprobs_cascade"):
    '''
        Retrieves logprobs from LLM
    '''
    def __init__(self, *args, logprobs_threshold=None, **kwargs):
        assert isinstance(logprobs_threshold, float)
        super().__init__(*args, logprobs=True, top_logprobs=5, **kwargs)
        self.logprobs_threshold = logprobs_threshold
    
    def should_cascade(self, step):
        return False