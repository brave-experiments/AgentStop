import argparse
import pandas as pd
import subprocess
import time
import traceback

from efficient_agents.agents.llm_backend import LlmBackend
from efficient_agents.profiler.profile import Profiler
from efficient_agents.profiler.analyze import Analyzer, DeviceType
from minisweagent.environments.docker import DockerEnvironment
from pathlib import Path

if __name__ == "__main__":
    example_text = """
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

    df = pd.read_parquet(args.input_path)
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

            prompt_path = Path(f"{base_path}/prompt.txt").resolve()
            with open(prompt_path, "w") as fp:
                fp.write(prob)

            retry_count = 0
            success = False
            while not success and retry_count <= num_retries:
                print(f"\n** [{time.time()}] Starting Docker image...\n")
                # Set up Docker image for the problem instance
                instance_id = df["instance_id"].iloc[idx]
                id_docker_compatible = instance_id.replace("__", "_1776_")
                image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
                try:
                    docker_env = DockerEnvironment(image=image_name, cwd="/testbed")
                except:
                    print(f"\n** [{time.time()}] Ran into an exception. Reclaiming Docker space and retrying...\n")
                    res = subprocess.run(["docker", "image", "prune", "-a", "--force"], check=True, capture_output=True, text=True)
                    print(res.stdout)
                    docker_env = DockerEnvironment(image=image_name, cwd="/testbed")
                print(f"\n** [{time.time()}] Docker image started.\n")

                script = args.script_template.format(prompt_path=prompt_path, agent_trace_path=agent_trace_path, docker_id=docker_env.container_id)
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

                print(f"\n** [{time.time()}] Cleaning up Docker image...\n")
                docker_env.cleanup()
                print(f"\n** [{time.time()}] Docker image cleaned up.\n")

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
                        output_ext=["png", "pdf"],
                        display_plots=False,
                        display_summary=False,
                        include_extra_info=False,
                    ).analyze()
                except Exception as e:
                    print(f"Analytics encountered an exception: {e}")
                    traceback.print_exc()
            else:
                print(f"** [{time.time()}] Profiling failed. **")
