from efficient_agents.agents.base import BaseAgent, LogProbsCascadeAgent
from efficient_agents.agents.utils import (
    CustomApiWebSearchTool,
    CustomWikipediaSearchTool,
    CustomVisitWebpageTool,
)
from importlib import resources

def create_web_tools(add_no_think, compression_ratio):
    return [
        CustomApiWebSearchTool(
            compression_ratio=compression_ratio,
            add_no_think=add_no_think,
        ),
        CustomWikipediaSearchTool(
            user_agent="MyWebAgent (dpham@brave.com)",
            language="en",
            content_type="text",
            extract_format="WIKI",
            compression_ratio=compression_ratio,
            add_no_think=add_no_think,
            max_output_length=10000,
        ),
        CustomVisitWebpageTool(
            compression_ratio=compression_ratio,
            add_no_think=add_no_think,
            max_output_length=10000,
        ),
    ]

class WebCodeAgent(BaseAgent, key="web_basic"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_tools(self):
        return create_web_tools(self.should_add_no_think, self.compression_ratio)

class WebCodeLogProbsAgent(LogProbsCascadeAgent, key="web_logprobs"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_tools(self):
        return create_web_tools(self.should_add_no_think, self.compression_ratio)

class WebCodeLogProbsIntrinsicExitAgent(WebCodeLogProbsAgent, key="web_logprobs_intrinsic_exit"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with resources.open_text("efficient_agents.config", "intrinsic_exit_system_prompt.txt") as f:
            intrinsic_exit_prompt = f.read()
        self.agent.prompt_templates["system_prompt"] = intrinsic_exit_prompt

    def get_tools(self):
        return create_web_tools(self.should_add_no_think, self.compression_ratio)
