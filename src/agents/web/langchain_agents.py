import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import BraveSearch
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from openinference.instrumentation.langchain import LangChainInstrumentor
from agents.web.utils import WebAgent, get_custom_arg_parser

class LangChainAgent(WebAgent):
    def __init__(
        self,
        model_id,
        model_type="ollama",
        stream=False,
        max_tokens=20000,
        api_key=None,
        **kwargs,
    ):
        super().__init__("LangChain", model_id, model_type, stream=stream, **kwargs)

        if model_type == "ollama":
             # num_predict=-2 means fill the context window
            model = ChatOllama(model=model_id, num_ctx=max_tokens, num_predict=-2)
        elif model_type == "anthropic":
            model = ChatAnthropic(model=model_id, max_tokens=max_tokens)
        else:
            raise NotImplementedError("Invalid model type")

        tools = [BraveSearch()]
        self.agent = create_react_agent(model, tools)

    def run(self, prompt):
        if not self.stream:
            for step in self.agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()
        else:
            pass

    def get_instrumentor(self):
        if self.instrumentor is None:
            self.instrumentor = LangChainInstrumentor()
        return self.instrumentor

if __name__ == "__main__":
    load_dotenv()

    example_text = """
Examples:

Run with an Ollama model:
python -m agents.web.langchain_agents \\
--model_id qwen2.5-coder:32b \\
--model_type ollama \\
--trace_path ./trace.json \\
--prompt "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"

Run with an Anthropic model:
python -m agents.web.langchain_agents \\
--model_id claude-sonnet-4-20250514 \\
--model_type anthropic \\
--trace_path ./trace.json \\
--prompt "Summarize the US Constitution." \\
--api_key_env ENV_FIELD_FOR_YOUR_API_KEY
"""

    parser = get_custom_arg_parser(description="Run a LangChain agent with tracing.", example_text=example_text)
    args = parser.parse_args()

    agent = LangChainAgent(
        model_id=args.model_id,
        model_type=args.model_type,
        stream=args.stream,
        api_key=os.getenv(args.api_key_env) if args.api_key_env is not None else None,
    )
    if args.trace_path is not None:
        agent.enable_tracing(args.trace_path)
    agent.run(args.prompt)