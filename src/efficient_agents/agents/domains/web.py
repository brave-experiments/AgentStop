from efficient_agents.agents.base import BaseAgent, LogProbsCascadeAgent
from efficient_agents.agents.utils import (
    CustomApiWebSearchTool,
    CustomWikipediaSearchTool,
    CustomVisitWebpageTool,
)

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
