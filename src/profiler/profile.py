#!/usr/bin/env python3
"""
System performance profiling using glances.
"""

import argparse
import json
import ollama
import os
import platform
import plistlib
import psutil
import re
import shlex
import signal
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from datetime import timezone
from pathlib import Path

LLM_BACKENDS = ["ollama", "mlc"]
LLM_BACKEND_PROCESS_FILTER = {
    "ollama": r"(o|O)llama",
    "mlc": r"mlc_llm",
}

class Profiler:
    def __init__(
        self,
        target_script,
        process_filter,
        glances_output_path,
        power_output_path=None,
        interval=1000, # ms
        capture_stdout=False,
        llm_backend=None,
        ollama_model_id=None,
    ):
        self.target_script = target_script
        self.process_filter = process_filter
        self.glances_output_path = glances_output_path
        self.power_output_path = power_output_path
        self.interval = interval
        self.capture_stdout = capture_stdout

        if llm_backend is not None:
            assert llm_backend in LLM_BACKENDS and llm_backend in LLM_BACKEND_PROCESS_FILTER
        self.llm_backend = llm_backend
        self.llm_backend_filter = LLM_BACKEND_PROCESS_FILTER.get(llm_backend, None)
        self.ollama_model_id = ollama_model_id

        self.glances_process = None
        self.glances_tmp_file = None
        self.power_process = None
        self.power_tmp_file = None
        self.target_process = None
        self.target_pids = set()  # Track PIDs created by the target script
        self.monitoring_thread = None
        self.stop_monitoring = False

    def track_process_tree(self):
        """Monitor and track all processes created by the target script"""

        def monitor_processes():
            while not self.stop_monitoring and self.target_process:
                try:
                    if self.target_process.poll() is None:  # Process still running
                        # Get the main process
                        main_process = psutil.Process(self.target_process.pid)
                        self.target_pids.add(self.target_process.pid)

                        # Get all child processes recursively
                        try:
                            children = main_process.children(recursive=True)
                            for child in children:
                                self.target_pids.add(child.pid)
                        except psutil.NoSuchProcess:
                            pass  # Process may have ended

                    time.sleep(0.1)  # Check every 100ms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        self.monitoring_thread = threading.Thread(target=monitor_processes, daemon=True)
        self.monitoring_thread.start()

    def terminate_ollama(self):
        assert isinstance(self.ollama_model_id, str)
        print(f"Stopping model {self.ollama_model_id}...")
        subprocess.run(["ollama", "stop", self.ollama_model_id.split("/")[1]])

    def start_ollama(self):
        print("Starting Ollama...")
        env = os.environ.copy()
        env["OLLAMA_CONTEXT_LENGTH"] = "40960"
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_KV_CACHE_TYPE"] = "f16"
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        model_id = self.ollama_model_id.split("/")[-1]

        # Unload the model if currently running to reset K-V cache
        running_models = ollama.ps()
        for model in running_models.models:
            if model_id == model.model:
                self.terminate_ollama()
                break

        print(f"Preloading model {model_id}...")
        ollama.generate(model=model_id)
        print("Ollama started and preloaded.")

    def start_glances(self):
        """Start glances with json export"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.glances_tmp_file = tmp.name
        plugins = "cpu,gpu,mem,memswap,sensors,network,diskio,processlist"
        with open(self.glances_tmp_file, "w") as f:
            backend_filter = f"|.*{self.llm_backend_filter}.*" if self.llm_backend_filter else ""
            self.glances_process = subprocess.Popen(
                [
                    "glances",
                    "--stdout-json",
                    plugins,
                    "--process-filter",
                    self.process_filter + backend_filter,
                    "--time",
                    str(self.interval / 1000),
                    "--disable-check-update",
                ],
                stdout=f,
                stderr=subprocess.DEVNULL,
            )
            # Give glances a moment to start
            time.sleep(3)
            print(
                f"Started glances logging to {self.glances_tmp_file} (PID = {self.glances_process.pid})"
            )

    def start_power_measurement(self):
        os_name = platform.system()
        if os_name == "Darwin":
            self.start_powermetrics()
        elif os_name == "Linux" and "tegra" in platform.release():
            self.start_tegrastats()
        else:
            raise NotImplementedError("Power measurement is not supported on your device.")

    def start_powermetrics(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.power_tmp_file = tmp.name

        self.power_process = subprocess.Popen(
            [
                "powermetrics",
                "--format", "plist",
                "--sample-rate", str(self.interval),
                "--samplers", "cpu_power,gpu_power,thermal",
                "--order", "pid",
                "--output-file", self.power_tmp_file
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give program a moment to start
        time.sleep(3)
        print(
            f"Started powermetrics logging to {self.power_tmp_file} (PID = {self.power_process.pid})"
        )

    def start_tegrastats(self):
        self.power_process = subprocess.Popen(
            [
                "python",
                "-m", "profiler.jetson",
                "--output_path", self.power_output_path,
                "--interval", str(self.interval),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        print(
            f"Started tegrastats logging to {self.power_output_path} (PID = {self.power_process.pid})"
        )

    def start_target_script(self):
        """Run the target Python script and track its process tree"""
        print(f"Starting target script: {self.target_script}")
        start_time = time.time_ns()
        output = None if self.capture_stdout else subprocess.DEVNULL
        try:
            # Run the target script
            self.target_process = subprocess.Popen(
                [s.strip() for s in shlex.split(self.target_script)],
                stdout=output,
                stderr=output,
            )

            # Start tracking the process tree
            self.track_process_tree()

            # Wait for the script to complete
            stdout, stderr = self.target_process.communicate()
            end_time = time.time_ns()
            self.stop_monitoring = True

            print(
                f"Target script completed in {round((end_time - start_time) / 1_000_000_000)}s"
            )
            print(
                f"Tracked {len(self.target_pids)} process IDs: {sorted(list(self.target_pids))}"
            )

            if self.target_process.returncode != 0:
                print("Target script completed with errors:")
                print(stderr.decode())
            else:
                print("Target script completed successfully")

            return True

        except Exception as e:
            print(f"Error running target script: {e}")
            self.stop_monitoring = True
            return False

    def stop_process(self, process):
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        time.sleep(2)

    def stop_glances(self):
        if self.glances_process is not None:
            print("Stopping glances...")
            self.stop_process(self.glances_process)
            self.glances_process = None
        print("Glances stopped")

    def stop_power_measurement(self):
        if self.power_process is not None:
            print("Stopping power measurement...")
            self.stop_process(self.power_process)
            self.power_process = None
        print("Power measurement stopped")

    def should_include_process(self, process):
        if process["status"] == "Z": # Zombie
            return False
        if process["pid"] in self.target_pids:
            return True
        if self.llm_backend is not None:
            if self.llm_backend == "ollama":
                target = process["name"]
            elif self.llm_backend == "mlc":
                target = " ".join(process["cmdline"])
            return bool(re.search(self.llm_backend_filter, target))
        return False

    def save_jsonl(self, lines, path_name):
        path = Path(path_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in lines:
                f.write(json.dumps(item) + "\n")
        setattr(self, path_name, None)

    def save_glances_log(self):
        """Filter the glances log for processes created by the target script and time period"""
        # Read
        logs = []
        with open(self.glances_tmp_file, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line.strip()))

        # Filter out processes that are not relevant
        for l in logs:
            l["processlist"] = [p for p in l["processlist"] if self.should_include_process(p)]
        # Filter out logs that do not include target processes and llm backend
        logs = [l for l in logs if len(l["processlist"]) > 0]
        valid_len = len(logs)
        # We don't need llm backend after the target script ends
        for l in reversed(logs):
            if any(p["pid"] in self.target_pids for p in l["processlist"]):
                break
            valid_len -= 1
        logs = logs[:valid_len]

        # Save
        self.save_jsonl(logs, self.glances_output_path)
        print(f"Glances log saved")

    def save_power_measurement(self):
        os_name = platform.system()
        if os_name == "Darwin":
            self.save_powermetrics()
        elif os_name == "Linux":
            if "tegra" not in platform.release():
                raise NotImplementedError("Power measurement on Linux is currently only supported for Nvidia Jetson.")
        else:
            raise NotImplementedError("Power measurement is not supported on your device.")

    def save_powermetrics(self):
        with open(self.power_tmp_file, "rb") as f:
            data = f.read()
        logs = [plistlib.loads(d) for d in data.split(b'\x00')]
        for log in logs:
            # Convert datetime object to Unix (ns) for JSON
            # Need to set timezone to UTC manually
            fixed_ts = log["timestamp"].replace(tzinfo=timezone.utc).timestamp()
            log["timestamp"] = int(fixed_ts * 1_000_000_000)
        self.save_jsonl(logs, self.power_output_path)
        print(f"Powermetrics log saved")

    def cleanup(self):
        print("Cleaning up...")
        self.stop_power_measurement()
        self.stop_glances()
        if self.glances_tmp_file is not None:
            os.remove(self.glances_tmp_file)
            self.glances_tmp_file = None
        if self.power_tmp_file is not None:
            os.remove(self.power_tmp_file)
            self.power_tmp_file = None
        if self.ollama_model_id is not None:
            self.terminate_ollama()
        

    def start_profiling(self):
        """Main method to run the complete monitoring process"""
        try:
            # Start
            self.start_glances()
            if self.power_output_path is not None:
                self.start_power_measurement()
            if self.ollama_model_id is not None:
                self.start_ollama()
            self.start_target_script()

            # Stop
            self.stop_power_measurement()
            self.stop_glances()

            # Save
            if self.power_output_path is not None:
                self.save_power_measurement()
            self.save_glances_log()
            return True
        except BaseException as e:
            print(f"Caught an Exception: {e}")
            traceback.print_exc()
            return False
        finally:
            self.cleanup()


def main():
    example_text = """
Examples:

sudo python -m profiler.profile \\
--script "python -m agents.web.smol_agents \\
    --agent_type code --model_id ollama_chat/qwen3:8b --model_type litellm \\
    --prompt 'What is the meaning of 42?' \\
    --stream \\
    --trace_path ../logs/smol_ollama_qwen3_8b_compressed_stream/smolagent_trace.json" \\
--glances_process_filter ".*python.*" \\
--glances_output_path ../logs/smol_ollama_qwen3_1.7b_compressed_stream/smolagent_glances.jsonl \\
--power_output_path ../logs/smol_ollama_qwen3_1.7b_compressed_stream/smolagent_powermetrics.jsonl \\
--frequency 100 \\
--capture_stdout \\
--llm_backend ollama \\
--ollama_model_id ollama_chat/qwen3:8b
"""

    parser = argparse.ArgumentParser(
        description="Generate a system resource profile of a target agent",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--script", type=str, required=True, help="Target agent script")
    parser.add_argument("--glances_process_filter", type=str, default=None, help="Regex for filtering glances process.")
    parser.add_argument("--glances_output_path", type=str, default=None, help="Path to save the glances log in JSONL.")
    parser.add_argument("--power_output_path", type=str, default=None, help="Path to save the processed power profile log in JSONL format.")
    parser.add_argument("--interval", type=int, default=1000, help="Samples resource metrics every <interval> milliseconds")
    parser.add_argument("--capture_stdout", action=argparse.BooleanOptionalAction, help="Print the output of the agent to stdout.")
    parser.add_argument("--llm_backend", type=str, default=None, choices=LLM_BACKENDS, help="LLM backend to include in profiling results.")
    parser.add_argument("--ollama_model_id", type=str, default=None, help="If specified, start Ollama server and preload the model. The model will also be stopped at the end.")
    
    args = parser.parse_args()

    if args.power_output_path is not None:
        system_name = platform.system()
        if not (system_name == "Darwin" or (system_name == "Linux" and "tegra" in platform.release())):
            raise Exception("Power measurement is not supported on your device.")
        if os.geteuid() != 0:
            raise Exception("Must run with sudo to enable power measurement.")

    profiler = Profiler(
        args.script,
        args.glances_process_filter,
        args.glances_output_path,
        power_output_path=args.power_output_path,
        interval=args.interval,
        capture_stdout=args.capture_stdout,
        llm_backend=args.llm_backend,
        ollama_model_id=args.ollama_model_id,
    )
    success = profiler.start_profiling()
    if success:
        print("Profiling completed successfully!")
    else:
        print("Profiling failed or was interrupted")


if __name__ == "__main__":
    main()
