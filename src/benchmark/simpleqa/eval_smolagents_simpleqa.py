import argparse
import itertools
import pandas as pd
import subprocess
import traceback

from agents.web.smol_agents import FullSmolAgent
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM smolagents on SimpleQA using Ollama",
        epilog="Example: python -m benchmark.simpleqa.eval_smolagents_simpleqa " \
            "--model_id qwen3:4b qwen3:8b qwen3:32b " \
            "--compression_ratio 0.5, 0.75 1.0 " \
            "--input_path ../data/simpleqa/llm_simpleqa_results_judged_filtered.csv " \
            "--output_path ../data/simpleqa/agent_simpleqa_results.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, nargs="+", required=True, help="Ollama model ID")
    parser.add_argument("--compression_ratio", type=float, nargs="+", default=[1.0], help="Compression ratio")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    args = parser.parse_args()
    model_ids = args.model_id
    compression_ratios = args.compression_ratio

    df = pd.read_csv(args.input_path)

    for model_id, compression_ratio in itertools.product(model_ids, compression_ratios):
        print(f"*** Evaluating smolagent {model_id} with compression ratio {compression_ratio} ***")

        agent = FullSmolAgent(
            model_id=f"ollama_chat/{model_id}",
            model_type="litellm",
            compression_ratio=compression_ratio,
        )
        
        answer = []
        for prob in tqdm(df["problem"].to_list()):
            try:
                res = agent.run(prob, max_steps=10).to_string()
                if "<think>" in res and "</think>" in res:
                    res = res[res.rfind("</think>")+len("</think>"):]
                answer.append(res.strip())
            except Exception as e:
                print(traceback.format_exc())
                answer.append(f"Exception: {e}")
        
        df[f"smolagent_{model_id}_compression_{compression_ratio}"] = answer
        df.to_csv(args.output_path, index=False)