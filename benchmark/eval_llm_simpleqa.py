import argparse
import ollama
import pandas as pd

from tqdm import tqdm

start_match = "<answer>"
end_match = "</answer>"
template = """Answer the following question:
<question>{0}</question>

Output a short answer in this format:

<answer>[Your answer]</answer>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLMs on SimpleQA using Ollama",
        epilog="Example: python eval_llm_simpleqa.py --model_id qwen3:4b qwen3:8b qwen3:32b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_id", type=str, nargs="+", required=True, help="Ollama model ID")
    args = parser.parse_args()
    model_ids = args.model_id

    df = pd.read_csv("hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv")

    for model_id in model_ids:
        print(f"************ Evaluating {model_id} *************")
        answer = []
        for prob in tqdm(df["problem"].to_list()):
            prompt = template.format(prob)
            res = ollama.generate(
                model=model_id,
                prompt=prompt,
                think=True,
                options={"temperature": 0.0, "num_predict": 128},
            ).response

            if start_match in res and end_match in res:
                parsed_result = res[res.find(start_match) + len(start_match):res.rfind(end_match)]
                answer.append(parsed_result.strip())
            else:
                answer.append(None)
        
        df[model_id] = answer
        df.to_csv("../data/llm_simpleqa_results.csv")