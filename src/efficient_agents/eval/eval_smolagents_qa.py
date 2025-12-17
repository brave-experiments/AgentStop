import argparse
import dataclasses
import itertools
import json
import pandas as pd
import subprocess
import time
import traceback

from efficient_agents.agents.domains import (
    WebCodeAgent,
)
from smolagents.agent_types import AgentText

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM smolagents on Q&A using Ollama",
        epilog="Example: python -m benchmark.eval_smolagents_qa " \
            "--model_id qwen3:4b qwen3:8b qwen3:32b " \
            "--compression_ratio 0.5, 0.75 1.0 " \
            "--planning_interval 0" \
            "--temperature 0.7" \
            "--top_p 0.8" \
            "--top_k 20" \
            "--min_p 0.0"
            "--input_path ../data/simpleqa/llm_simpleqa_results_judged_filtered.csv " \
            "--question_col problem" \
            "--answer_col answer" \
            "--output_path ../data/simpleqa/agent_simpleqa_results.csv" \
            "--output_col_pattern code_{0}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, nargs="+", required=True, help="Ollama model ID")
    parser.add_argument("--compression_ratio", type=float, nargs="+", default=[1.0], help="Compression ratio")
    parser.add_argument("--planning_interval", type=int, default=None, help="Planning interval.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p")
    parser.add_argument("--top_k", type=int, default=20, help="Sampling top_k")
    parser.add_argument("--min_p", type=float, default=0.0, help="Sampling min_p")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input")
    parser.add_argument("--question_col", type=str, required=True, help="Question column name")
    parser.add_argument("--answer_col", type=str, required=True, help="Answer column name")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    parser.add_argument("--output_col_pattern", type=str, required=True, help="Python column name pattern for the output, with exactly one argument for the model id. (e.g., code_{0})")
    parser.add_argument("--trace_dump_path", type=str, required=True, help="Path to save the full execution trace as a JSON file")

    args = parser.parse_args()
    model_ids = args.model_id
    compression_ratios = args.compression_ratio

    df = pd.read_csv(args.input_path)
    problems = df[args.question_col].to_list()
    targets = df[args.answer_col].to_list()
    col_pattern = args.output_col_pattern
    all_results = []

    for compression_ratio, model_id in itertools.product(compression_ratios, model_ids):
        print(f"*** Evaluating smolagent {model_id} with compression ratio {compression_ratio} ***")

        agent = WebCodeAgent(
            model_id=f"ollama_chat/{model_id}",
            model_type="litellm",
            compression_ratio=compression_ratio,
            planning_interval=args.planning_interval,
            thinking=False,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
        )
        
        answer = [None] * len(problems)
        for i, prob in enumerate(problems):
            print(f"\n*** Solving problem {i+1}/{len(problems)}  ***\n")
            max_try = 3
            trace_data = {
                "model_id": model_id,
                "compression_ratio": compression_ratio,
                "question_idx": i,
                "question": prob,
                "answer": targets[i],
            }
            for try_num in range(max_try):
                try:
                    print(f"\nAttempt {try_num+1}:\n")
                    agent.reset_tools()
                    full_results = agent.run(prob, max_steps=10)
                    trace_data["agent_trace"] = dataclasses.asdict(full_results)
                    res = full_results.output
                    if isinstance(res, AgentText):
                        res = res.to_string()
                        if "<think>" in res and "</think>" in res:
                            res = res[res.rfind("</think>")+len("</think>"):]
                        answer[i] = res.strip()
                    else:
                        answer[i] = str(res)
                    break
                except Exception as e:
                    print(traceback.format_exc())
                    answer[i] = f"Exception: {e}"
                    all_results["agent_trace"] = dataclasses.asdict(agent.get_current_results())
                    time.sleep(30)

            print(f"\nTarget: {targets[i]} | Answer to problem {i+1}: {answer[i]}\n")

            df[col_pattern.format(model_id)] = answer
            df.to_csv(args.output_path, index=False)
            for m in trace_data["agent_trace"]["messages"]:
                if m is None:
                    continue
                output_msg = m.get("model_output_message", None)
                if output_msg is None:
                    continue
                if "raw" in output_msg:
                    output_msg["raw"] = None
            all_results.append(json.dumps(trace_data))
        
        with open(args.trace_dump_path, "w") as f:
            for r in all_results:
                f.write(f"{r}\n")
