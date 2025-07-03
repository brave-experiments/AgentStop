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
import signal
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from datetime import timezone
from pathlib import Path


class Profiler:
    def __init__(
        self,
        target_script,
        args,
        process_filter,
        glances_output_path,
        power_output_path=None,
        frequency=1000, # ms
        capture_stdout=False,
        include_ollama=False,
        model_id=None,
    ):
        self.target_script = target_script
        self.args = args
        self.process_filter = process_filter
        self.glances_output_path = glances_output_path
        self.power_output_path = power_output_path
        self.frequency = frequency
        self.capture_stdout = capture_stdout
        self.include_ollama = include_ollama
        self.model_id = model_id

        if self.include_ollama and self.model_id is None:
            raise Exception("Model ID must be specified if Ollama is included.")

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
        # Stops all ollama processes
        print("Attempting to terminate any running Ollama processes...")
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if "ollama" in proc.info["name"].lower():
                    print(f"Terminating Ollama process {proc.info['pid']}")
                    self.stop_process(proc)
            except Exception as e:
                print(f"Exception occurred when terminating {proc.pid}: {e}")
                continue

    def start_ollama(self):
        print("Starting Ollama...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        model_id = self.model_id.split("/")[-1]
        print(f"Preloading model {model_id}...")
        ollama.generate(
            model=model_id,
            prompt=" ",
            # messages=[{"role": "user", "content": " "}],
            options={"temperature": 0.0, "num_predict": 1},
        )
        print("Ollama started and preloaded.")

    def start_glances(self):
        """Start glances with json export"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.glances_tmp_file = tmp.name
        plugins = "cpu,gpu,mem,memswap,sensors,network,diskio,processlist"
        with open(self.glances_tmp_file, "w") as f:
            self.glances_process = subprocess.Popen(
                [
                    "glances",
                    "--stdout-json",
                    plugins,
                    "--process-filter",
                    self.process_filter
                    + ("|.*(o|O)llama.*" if self.include_ollama else ""),
                    "--time",
                    str(self.frequency / 1000),
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
        else:
            raise NotImplemented("Power measurement is currently only supported on Mac.")

    def start_powermetrics(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.power_tmp_file = tmp.name

        self.power_process = subprocess.Popen(
            [
                "powermetrics",
                "--format", "plist",
                "--sample-rate", str(self.frequency),
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

    def start_target_script(self):
        """Run the target Python script and track its process tree"""
        print(f"Starting target script: {self.target_script}")
        start_time = time.time_ns()
        output = None if self.capture_stdout else subprocess.DEVNULL
        try:
            # Run the target script
            self.target_process = subprocess.Popen(
                [sys.executable, self.target_script, *self.args],
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

    def should_include_process(self, pid, name):
        return pid in self.target_pids or (self.include_ollama and "ollama" in name.lower())

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
            l["processlist"] = [
                p for p in l["processlist"]
                if self.should_include_process(p["pid"], p["name"])
                and p["status"] != "Z" # Zombie
            ]
        # Filter out logs that do not include target processes and Ollama
        logs = [l for l in logs if len(l["processlist"]) > 0]
        valid_len = len(logs)
        # We don't need Ollama after the target script ends
        for l in reversed(logs):
            if any([p["pid"] in self.target_pids for p in l["processlist"]]):
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
        else:
            raise NotImplemented("Power measurement is currently only supported on Mac.")

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
        if self.include_ollama:
            self.terminate_ollama()
        

    def start_profiling(self):
        """Main method to run the complete monitoring process"""
        try:
            # Start
            if self.include_ollama:
                self.terminate_ollama()
            self.start_glances()
            if self.power_output_path is not None:
                self.start_power_measurement()
            if self.include_ollama:
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

python profile.py \\
--script ../agents/web/smol_agents.py \\
--model_id mlx-community/Qwen3-32B-4bit \\
--model_type mlx \\
--prompt "What is the meaning of 42?" \\
--trace_path ./logs/smolagent_trace.json \\
--glances_process_filter ".*python.*" \\
--glances_output_path logs/smolagent_glances.jsonl \\
--power_output_path logs/smolagent_power.jsonl \\
--frequency 500 \\
--capture_stdout

python profile.py \\
--script ../agents/web/langchain_agents.py \\
--model_id qwen3:32b \\
--model_type ollama \\
--prompt "What is the meaning of 42?" \\
--trace_path ./logs/langchain_trace.json \\
--glances_process_filter ".*python.*" \\
--glances_output_path logs/langchain_glances.jsonl \\
--power_output_path logs/langchain_power.jsonl \\
--frequency 500 \\
--capture_stdout \\
--include_ollama
"""

    parser = argparse.ArgumentParser(
        description="Generate a system resource profile of a target agent",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--script", type=str, required=True, help="Target agent script")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to use.")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model backend.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to give to the agent.")
    parser.add_argument("--trace_path", type=str, default=None, help="Path to save the JSON trace.")
    parser.add_argument("--api_key_env", type=str, default=None, help="Optional .env's field for the model API key.")
    parser.add_argument("--glances_process_filter", type=str, default=None, help="Regex for filtering glances process.")
    parser.add_argument("--glances_output_path", type=str, default=None, help="Path to save the glances log in JSONL.")
    parser.add_argument("--power_output_path", type=str, default=None, help="Path to save the processed power profile log in JSONL format.")
    parser.add_argument("--frequency", type=int, default=1000, help="Samples resource metrics every <frequency> milliseconds")
    parser.add_argument("--capture_stdout", action=argparse.BooleanOptionalAction, help="Print the output of the agent to stdout.")
    parser.add_argument("--include_ollama", action=argparse.BooleanOptionalAction, help="Include Ollama in the profiling.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, help="Enable streaming.")

    args = parser.parse_args()

    if args.power_output_path is not None and platform.system() == "Darwin" and os.geteuid() != 0:
        raise Exception("Must run with sudo to enable power measurement on Mac")
    
    # Create and run the profiler
    script_args = [
        ("--model_id", args.model_id),
        ("--model_type", args.model_type),
        ("--stream" if args.stream else "--no-stream",),
        ("--prompt", args.prompt),
        ("--trace_path", args.trace_path),
        ("--api_key_env", args.api_key_env),
    ]
    script_args = [
        a for arg in script_args for a in arg
        if len(arg) == 1 or arg[1] is not None
    ]

    profiler = Profiler(
        args.script,
        script_args,
        args.glances_process_filter,
        args.glances_output_path,
        power_output_path=args.power_output_path,
        frequency=args.frequency,
        capture_stdout=args.capture_stdout,
        include_ollama=args.include_ollama,
        model_id=args.model_id,
    )
    success = profiler.start_profiling()
    if success:
        print("Profiling completed successfully!")
    else:
        print("Profiling failed or was interrupted")


if __name__ == "__main__":
    main()
