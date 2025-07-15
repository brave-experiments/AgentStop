import argparse
import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from pathlib import Path

NO_THINK = "/no_think"

class JsonSpanExporter(InMemorySpanExporter):
    """Simple file-based span exporter that writes JSON traces to a file"""

    def __init__(self, file_path="traces.json"):
        super().__init__()
        self.file_path = file_path

    def shutdown(self):
        """Write everything to file once shutdown"""

        super().shutdown()
        spans = []
        for span in self.get_finished_spans():
            span_dict = {
                "name": span.name,
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
                "parent_span_id": (
                    format(span.parent.span_id, "016x") if span.parent else None
                ),
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status": span.status.status_code.name,
                "attributes": dict(span.attributes) if span.attributes else {},
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": (
                            dict(event.attributes) if event.attributes else {}
                        ),
                    }
                    for event in span.events
                ],
            }
            spans.append(span_dict)

        path = Path(self.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(spans, f, indent=2, default=str)


class WebAgent:
    def __init__(
        self,
        name,
        model_id,
        model_type,
        *args,
        stream=False,
        **kwargs
    ):
        self.name = name
        self.model_id = model_id
        self.model_type = model_type
        self.stream = stream
        self.instrumentor = None
        self.args = args
        self.kwargs = kwargs

    def get_instrumentor(self):
        raise NotImplementedError("You need to implement this method.")

    def enable_tracing(self, output_path):
        self.get_instrumentor().instrument(
            tracer_provider=get_json_exporter(self.name, output_path)
        )

    def disable_tracing(self):
        self.get_instrumentor().uninstrument()

    def run(self, *args, **kwargs):
        raise NotImplementedError("You need to implement this method.")


def get_json_exporter(service_name, output_file):
    resource = Resource.create({"service.name": service_name})
    file_exporter = JsonSpanExporter(output_file)
    span_processor = BatchSpanProcessor(file_exporter)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor)
    return tracer_provider


def get_custom_arg_parser(description, example_text):
    parser = argparse.ArgumentParser(
        description=description,
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model backend.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to give to the agent.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False, help="Enable streaming.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False, help="Enable thinking.")
    parser.add_argument("--trace_path", type=str, default=None, help="Path to save the JSON trace.")
    parser.add_argument("--api_key_env", type=str, default=None, help="Optional .env's field for the model API key.")

    return parser