import argparse
import json
import pandas as pd
import pymupdf
import pymupdf4llm
import re
import requests
import torch

from bs4 import BeautifulSoup
from collections.abc import Generator
from dataclasses import dataclass
from fake_useragent import UserAgent
from llmlingua import PromptCompressor
from markdownify import markdownify
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from pathlib import Path
from requests.exceptions import RequestException
from smolagents import (
    ApiWebSearchTool,
    Tool,
    VisitWebpageTool,
    WebSearchTool,
    WikipediaSearchTool,
)
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    ChatMessageToolCallStreamDelta,
    LiteLLMModel,
    MessageRole,
)
from smolagents.monitoring import TokenUsage
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

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
        stream_llm=False,
        **kwargs
    ):
        self.name = name
        self.model_id = model_id
        self.model_type = model_type
        self.stream_llm = stream_llm
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

@dataclass
class CustomChatMessageStreamDelta(ChatMessageStreamDelta):
    logprobs: Any | None = None

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
    stream_deltas: list[CustomChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    """
    Agglomerate a list of stream deltas into a single stream delta.
    """
    accumulated_tool_calls: dict[int, list[ChatMessageToolCallStreamDelta]] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    logprobs = []
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.logprobs:
            logprobs.append(stream_delta.logprobs)
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
        raw={"logprobs": logprobs},
    )


class CustomLiteLLMModel(LiteLLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[CustomChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        for event in self.client.completion(**completion_kwargs, stream=True, stream_options={"include_usage": True}):
            if getattr(event, "usage", None):
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield CustomChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield CustomChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                        logprobs=getattr(choice, "logprobs", None),
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")


class CustomTool():
    """
        - Check for duplidate input
        - Compress results
        - Add /no_think to disable thinking
    """
    def __init__(
        self,
        *args,
        no_duplicate=True,
        compression_ratio=1.0,
        compress_final_output=True,
        add_no_think=False,
        **kwargs
    ):
        self.no_duplicate = no_duplicate
        self.input_history = set()
        self.compress_tool = TextCompressionTool(compression_ratio=compression_ratio)
        self.compression_ratio = compression_ratio
        self.compress_final_output = compress_final_output
        self.add_no_think = add_no_think
        super().__init__(*args, **kwargs)

    def forward(self, query):
        if self.no_duplicate and query in self.input_history:
            res = f"Error: You have already tried this query. Please check the results of your previous queries. " \
                " Do not repeat this query. Please try a different query; otherwise, you will not get any results."
        else:
            self.input_history.add(query)
            res = self._forward(query)
            if self.compress_final_output:
                res = self.compress_tool.forward(res)
        
        if self.add_no_think:
            res = f"{res} {NO_THINK}"
        
        return res

    def _forward(self, query):
        raise NotImplementedError("This method must be implemented!")

    def reset_history(self):
        self.input_history.clear()


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


class CustomWebSearchTool(CustomTool, WebSearchTool):
    """Compress each search result individually"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, compress_final_output=False, **kwargs)

    def _forward(self, query):
        results = self.search(query)
        if len(results) == 0:
            return "No results found! Try a less restrictive/shorter query."

        res = "## Search Results\n\n" + "\n\n".join([
            f"[{r['title']}]({r['link']})\n{self.compress_tool.forward(r['description'])}"
            for r in results
        ])
        return res


class CustomApiWebSearchTool(CustomTool, ApiWebSearchTool):
    """Compress each search result individually"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, compress_final_output=False, **kwargs)

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


class CustomWikipediaSearchTool(CustomTool, WikipediaSearchTool):
    def __init__(self, *args, max_output_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_output_length = max_output_length

    def _forward(self, query):
        try:
            page = self.wiki.page(query)
            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            try:
                tables = pd.read_html(url)
            except:
                tables = []
            if len(tables) > 0:
                joined_table = "\n\n".join(t.to_markdown() for t in tables)
                table = f"\n\n**Tables:**\n\n{joined_table}\n\n"
            else:
                table = ""

            if self.max_output_length is not None and len(text) > self.max_output_length:
                text = text[:self.max_output_length]

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n{table}🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class CustomVisitWebpageTool(CustomTool, VisitWebpageTool):
    """
        Features:
        - Better user-agent
        - Ability to convert pdf to markdown
        - Automatically switches to Wikipedia API if url is for Wikipedia
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ua = UserAgent(platforms="desktop").random
        self.wiki = CustomWikipediaSearchTool(
            user_agent="MyWebAgent (dpham@brave.com)",
            language="en",
            content_type="text",
            extract_format="WIKI",
            compression_ratio=self.compression_ratio,
            max_output_length=self.max_output_length,
        )

    def forward(self, url): # Needs to match the argument name with the VisitWebPageTool
        return super().forward(url)

    def _forward(self, url):
        wiki_pattern = "wikipedia.org/wiki/"
        if wiki_pattern in url:
            url_parts = url.split(wiki_pattern)
            if len(url_parts) != 2 or len(url_parts[1]) == 0:
                return "Error: Invalid wikipedia URL"
            return self.wiki.forward(unquote(url_parts[1]))
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

def create_tools(add_no_think, compression_ratio):
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
    parser.add_argument("--model_id", type=str, nargs="+", required=True, help="Model ID(s) to use.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model backend.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to give to the agent.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False, help="Enable LLM streaming.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False, help="Enable thinking.")
    parser.add_argument("--trace_path", type=str, default=None, help="Path to save the JSON trace.")
    parser.add_argument("--api_key_env", type=str, default=None, help="Optional .env's field for the model API key.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p")
    parser.add_argument("--top_k", type=int, default=20, help="Sampling top_k")
    parser.add_argument("--min_p", type=float, default=0.0, help="Sampling min_p")
    return parser