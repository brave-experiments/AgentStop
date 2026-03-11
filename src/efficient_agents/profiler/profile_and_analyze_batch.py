import argparse
import pandas as pd
import time
import traceback

from efficient_agents.agents.llm_backend import LlmBackend
from efficient_agents.profiler.profile import Profiler
from efficient_agents.profiler.analyze import Analyzer, DeviceType
from pathlib import Path

if __name__ == "__main__":
    example_text = """
Examples:

sudo python -m profiler.profile_and_analyze_multiple \\
--script_template "python -m efficient_agents.agents.web \\
    --agent_type fixed_cascade_compress \\
    --model_id ollama_chat/qwen3:1.7b ollama_chat/qwen3:14b \\
    --model_type litellm \\
    --prompt {prompt} \\
    --stream \\
    --fixed_cascade_step 1 \\
    --trace_path {agent_trace_path}" \\
--input_path ../data/frames/frames_code_1.7b_incorrect_14b_correct.csv \\
--question_col Prompt \\
--num_repeats 3 \\
--base_output_path ../logs/frames_ollama_qwen3_fixed_cascade_compress_1.7_to_14_step_1 \\
--preload_model_id ollama_chat/qwen3:1.7b \\
--device_type apple_laptop
"""

    parser = argparse.ArgumentParser(
        description="Profile and analyze using a target script on multiple data inputs from a csv file",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--script_template", type=str, required=True, help="Target agent script template.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input.")
    parser.add_argument("--question_col", type=str, required=True, help="Column name for the questions in the input file.")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repetitions.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds) for target script. Default to no timeout.")
    parser.add_argument("--num_retries", type=int, default=0, help="Number of retries if profiling fails. Default to 0.")
    parser.add_argument("--base_output_path", type=str, required=True, help="Base path to store output.")
    parser.add_argument("--preload_model_id", type=str, default=None, help="Model id for the profiler.")
    parser.add_argument("--device_type", type=str, required=True, choices=DeviceType.ALL, help="Type of device for analysis.")
    parser.add_argument("--llm_backend", type=str, default=None, choices=LlmBackend.subclasses.keys(), help="LLM backend.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    problems = df[args.question_col].to_list()
    Path(args.base_output_path).mkdir(parents=True, exist_ok=True)
    num_repeats = args.num_repeats
    num_retries = args.num_retries
    assert num_repeats > 0
    assert num_retries >= 0

    for run_idx in range(args.num_repeats):
        print(f"** Run {run_idx + 1}/{args.num_repeats} **")
        for idx, prob in enumerate(problems):
            base_path = f"{args.base_output_path}/{idx}/run_{run_idx}"
            glances_log_path = f"{base_path}/raw/glances.jsonl"
            agent_trace_path = f"{base_path}/raw/trace.json"
            power_log_path = f"{base_path}/raw/power.jsonl"
            analysis_output_path = f"{base_path}/analysis"

            Path(f"{base_path}/raw").mkdir(parents=True, exist_ok=True)
            Path(f"{base_path}/analysis").mkdir(parents=True, exist_ok=True)

            prob = prob.replace('"', '\\"')
            script = args.script_template.format(prompt=f'"{prob}"', agent_trace_path=agent_trace_path)

            retry_count = 0
            success = False
            while not success and retry_count <= num_retries:
                print(f"\n** [{time.time()}] Profiling started (attempt {retry_count}/{num_retries}) for script:\n{script}\n")
                success = Profiler(
                    script,
                    ".*python.*",
                    glances_log_path,
                    power_output_path=power_log_path,
                    interval=100,
                    timeout=args.timeout,
                    capture_stdout=True,
                    llm_backend=args.llm_backend,
                    preload_model_id=args.preload_model_id,
                ).start_profiling()
                if not success:
                    retry_count += 1

            if success:
                print(f"** [{time.time()}] Profiling finished. **")
                try:
                    print(f"\n** Producing analytics... **\n")
                    Analyzer(
                        args.device_type,
                        glances_log_path,
                        agent_trace_path,
                        power_log_path=power_log_path,
                        model_id=None,
                        full_execution=False,
                        output_dir=analysis_output_path,
                        output_ext=["pdf"],
                        display_plots=False,
                        display_summary=False,
                    ).analyze()
                except Exception as e:
                    print(f"Analytics encountered an exception: {e}")
                    traceback.print_exc()
            else:
                print(f"** [{time.time()}] Profiling failed. **")
