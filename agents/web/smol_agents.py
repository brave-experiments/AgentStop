import argparse
import json
import os

from dotenv import load_dotenv
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from smolagents import (
    CodeAgent,
    MLXModel,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool
)

### OpenTelemetry code for exporting trace to JSON

class JsonSpanExporter:
    """Simple file-based span exporter that writes JSON traces to a file"""
    
    def __init__(self, file_path="traces.json"):
        self.file_path = file_path
        self.spans_data = []
    
    def export(self, spans):
        """Export spans to JSON file"""
        for span in spans:
            span_dict = {
                "name": span.name,
                "trace_id": format(span.context.trace_id, '032x'),
                "span_id": format(span.context.span_id, '016x'),
                "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status": span.status.status_code.name,
                "attributes": dict(span.attributes) if span.attributes else {},
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes) if event.attributes else {}
                    }
                    for event in span.events
                ]
            }
            self.spans_data.append(span_dict)
        
        # Write all spans to JSON file
        with open(self.file_path, 'w') as f:
            json.dump(self.spans_data, f, indent=2, default=str)
        
        return True
    
    def shutdown(self):
        """Cleanup method"""
        pass

def get_json_exporter(service_name, output_file):
    """Setup OpenTelemetry with file export"""
    
    resource = Resource.create({
        "service.name": service_name,
    })
    tracer_provider = TracerProvider(resource=resource)
    file_exporter = JsonSpanExporter(output_file)
    span_processor = BatchSpanProcessor(file_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    return tracer_provider

### Agent definitions

class SmolAgent:
    def __init__(self, name, model_id, model_type, max_tokens=20000, api_key=None):
        self.name = name
        if model_type == "mlx": # For Apple silicon
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
        )

        self.instrumentor = SmolagentsInstrumentor()

    def enable_trace(self, output_path):
        self.instrumentor.instrument(tracer_provider=get_json_exporter(self.name, output_path))

    def disable_trace(self):
        self.instrumentor.uninstrument()

    def run(self, prompt=None):
        self.agent.run(prompt)

if __name__ == "__main__":
    load_dotenv()

    example_text = """
Examples:

  Run with an MLX model:
    python smol_agents.py --agent_name BasicMlx
                          --model_id mlx-community/Qwen2.5-Coder-32B-Instruct-4bit \\
                          --model_type mlx \\
                          --trace_path ./mlx_trace.json \\
                          --prompt "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"

  Run with a LiteLLM model:
    python smol_agents.py --agent_name BasicClaude4Sonnet
                          --model_id anthropic/claude-sonnet-4-20250514 \\
                          --model_type litellm \\
                          --trace_path ./litellm_trace.json \\
                          --prompt "Summarize the US Constitution." \\
                          --api_key_env ENV_FIELD_FOR_YOUR_API_KEY
"""

    parser = argparse.ArgumentParser(
        description="Run a SmolAgent with tracing.",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--agent_name", type=str, required=True, help="Agent's name")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use (e.g., HuggingFace, anthropic/<MODEL_NAME>), etc.")
    parser.add_argument("--model_type", type=str, choices=["mlx", "litellm"], required=True, help="Type of the model backend.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to give to the agent.")
    parser.add_argument("--trace_path", type=str, default=None, help="Path to the trace output JSON file.")
    parser.add_argument("--api_key_env", type=str, default=None, help="Optional .env's field for API key for litellm models.")

    args = parser.parse_args()

    agent = SmolAgent(
        name=args.agent_name,
        model_id=args.model_id,
        model_type=args.model_type,
        api_key=os.getenv(args.api_key_env)
    )
    if args.trace_path is not None:
        agent.enable_trace(args.trace_path)
    agent.run(args.prompt)