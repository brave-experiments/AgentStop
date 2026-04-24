import argparse
import os

from dotenv import load_dotenv

import efficient_agents.agents.domains  # Register all agent implementations
from efficient_agents.agents.base import BaseAgent
from efficient_agents.agents.llm_backend import LlmBackend

load_dotenv()

def get_custom_arg_parser(description, example_text):
    parser = argparse.ArgumentParser(
        description=description,
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, nargs="+", required=True, help="Model ID(s) to use.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model backend.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to give to the agent. Either prompt or prompt_path must be specified.")
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to the prompt to give to the agent. Either prompt or prompt_path must be specified.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False, help="Enable LLM streaming.")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False, help="Enable thinking.")
    parser.add_argument("--trace_path", type=str, default=None, help="Path to save the JSON trace.")
    parser.add_argument("--api_key_env", type=str, default=None, help="Optional .env's field for the model API key.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p")
    parser.add_argument("--top_k", type=int, default=20, help="Sampling top_k")
    parser.add_argument("--min_p", type=float, default=0.0, help="Sampling min_p")
    parser.add_argument(
        "--agent_type",
        type=str,
        required=True,
        choices=list(BaseAgent.subclasses.keys()),
        help=f"Type of agent.",
    )
    parser.add_argument("--planning_interval", type=int, default=None, help="Planning interval.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps for the agent.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max output tokens per step")
    parser.add_argument("--fixed_cascade_step", type=int, default=None, help="Fixed step number to cascade if using fixed cascade agent.")
    parser.add_argument("--api_base", type=str, default=None, help="API base for LiteLLM.")
    parser.add_argument("--llm_backend", type=str, choices=LlmBackend.subclasses.keys(), default=None, help="LLM backend that will be used. This is only needed for cascading.")
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=1.0,
        help="Compression ratio for all results."
    )
    parser.add_argument("--docker_id", type=str, default=None, help="Docker instance ID")
    return parser

if __name__ == "__main__":
    example_text = """
Examples:

Run with an MLX model:
python -m efficient_agents.agents.run \\
--agent_type web_basic \\
--model_id mlx-community/Qwen3-32B-4bit \\
--model_type mlx \\
--trace_path ./trace.json \\
--prompt "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"

Run with Anthropic via LiteLLM:
python -m efficient_agents.agents.run \\
--agent_type web_basic \\
--model_id anthropic/claude-sonnet-4-20250514 \\
--model_type litellm \\
--trace_path ./trace.json \\
--prompt "Summarize the US Constitution." \\
--api_key_env ENV_FIELD_FOR_YOUR_API_KEY

Run with Ollama via LiteLLM with streaming:
python -m efficient_agents.agents.run \\
--agent_type web_basic \\
--model_id ollama_chat/qwen3:32b \\
--model_type litellm \\
--stream \\
--trace_path ./trace.json \\
--prompt "What is the meaning of life?"

Run with MLC-LLM via LiteLLM
(assuming you have run something like
`mlc_llm serve HF://mlc-ai/Qwen3-32B-q4f16_1-MLC \\
    --overrides "gpu_memory_utilization=1.0;prefill_chunk_size=1024"`:

python -m efficient_agents.agents.run \\
--agent_type web_basic \\
--model_id openai/HF://mlc-ai/Qwen3-32B-q4f16_1-MLC \\
--model_type litellm \\
--api_base http://127.0.0.1:8000/v1 \\
--stream \\
--trace_path ./trace.json \\
--prompt "How to learn to vocal fry?"
"""

    parser = get_custom_arg_parser(description="Run an agent with tracing enabled.", example_text=example_text)
    args = parser.parse_args()

    agent = BaseAgent.create(
        args.agent_type,
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
        max_tokens=args.max_tokens,
        cascade_step=args.fixed_cascade_step,
        llm_backend=args.llm_backend,
        docker_id=args.docker_id,
    )
    if args.trace_path is not None:
        agent.enable_tracing(args.trace_path)

    prompt = args.prompt
    if prompt is None:
        assert args.prompt_path is not None
        try:
            with open(args.prompt_path, "r") as fp:
                prompt = fp.read()
        except:
            raise Exception(f"Cannot read prompt at {args.prompt_path}")
    agent.run(prompt, max_steps=args.max_steps)
