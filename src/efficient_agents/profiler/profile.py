#!/usr/bin/env python3
"""
System performance profiling using glances.
"""

import argparse
import json
import os
import platform
import plistlib
import psutil
import shlex
import signal
import subprocess
import threading
import tempfile
import time
import traceback

from datetime import timezone
from efficient_agents.agents.llm_backend import LlmBackend
from pathlib import Path

SEC_TO_NANOSEC = 1_000_000_000

class Profiler:
    def __init__(
        self,
        target_script,
        process_filter,
        glances_output_path,
        power_output_path=None,
        interval=1000, # ms
        timeout=None, # sec
        capture_stdout=False,
        llm_backend=None,
        preload_model_id=None,
    ):
        self.target_script = target_script
        self.process_filter = process_filter
        self.glances_output_path = glances_output_path
        self.power_output_path = power_output_path
        self.interval = interval
        self.timeout = timeout
        self.capture_stdout = capture_stdout

        if llm_backend is not None:
            self.llm_backend = LlmBackend.create(llm_backend)
        self.preload_model_id = preload_model_id

        self.glances_process = None
        self.glances_tmp_file = None
        self.power_process = None
        self.power_tmp_file = None
        self.target_process = None
        self.target_pids = set()  # Track PIDs created by the target script
        self.monitoring_thread = None
        self.stop_monitoring = False

    def track_process(self, start_time=None):
        """Monitor and track all processes created by the target script"""

        def monitor_processes():
            while not self.stop_monitoring and self.target_process:
                try:
                    if self.target_process.poll() is not None:  # Process terminated
                        break

                    # Get the main process
                    self.target_pids.add(self.target_process.pid)
                    main_process = psutil.Process(self.target_process.pid)

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

                if self.timeout is not None and start_time is not None:
                    elapsed_time = (time.time_ns() - start_time) / SEC_TO_NANOSEC
                    if elapsed_time > self.timeout * 1.1:
                        print(f"WARNING: Process has exceeded timeout without raising an exception. {elapsed_time}s have elapsed.")
                        self.stop_process(self.target_process, process_group=True)
                        break

        self.monitoring_thread = threading.Thread(target=monitor_processes, daemon=True)
        self.monitoring_thread.start()

    def start_glances(self):
        """Start glances with json export"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.glances_tmp_file = tmp.name
        plugins = "cpu,gpu,mem,memswap,sensors,network,diskio,processlist"
        with open(self.glances_tmp_file, "w") as f:
            backend_filter = self.llm_backend.process_filter
            backend_filter = f"|.*{backend_filter}.*" if backend_filter is not None else ""
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
        args = [
            "python",
            "-m", "profiler.jetson",
            "--output_path", self.power_output_path,
            "--interval", str(self.interval),
        ]
        self.power_process = subprocess.Popen(
            args,
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
                start_new_session=True,
            )

            # Start tracking the process tree
            self.stop_monitoring = False
            self.track_process(start_time=start_time)

            # Wait for the script to complete
            return_code = self.target_process.wait(timeout=self.timeout)
            end_time = time.time_ns()
            duration = round((end_time - start_time) / SEC_TO_NANOSEC)

            print(f"Target script completed in {duration}s")
            print(f"Tracked {len(self.target_pids)} process IDs: {sorted(list(self.target_pids))}")

            if return_code < 0:
                raise Exception(f"Target process exited with status code {return_code}")
        except BaseException as e:
            raise e
        finally:
            if self.target_process is not None:
                self.stop_process(self.target_process, process_group=True)
                self.target_process = None
            self.stop_monitoring = True
            if self.monitoring_thread is not None:
                self.monitoring_thread.join()
                self.monitoring_thread = None

    def stop_process(self, process, process_group=False, grace_period=5):
        if process is None or process.poll() is not None:
            return
        if process_group:
            pg_id = os.getpgid(process.pid)
            print(f"Attempting to terminate process group {pg_id}...")
            os.killpg(pg_id, signal.SIGTERM)
            try:
                process.wait(timeout=grace_period)
            except subprocess.TimeoutExpired:
                print(f"Forcefully terminating process group {pg_id}...")
                os.killpg(pg_id, signal.SIGKILL)
        else:
            print(f"Attempting to terminate process {process.pid}")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=grace_period)
            except subprocess.TimeoutExpired:
                print(f"Forcefully terminating process {process.pid}")
                process.kill()
        time.sleep(2)

    def stop_glances(self):
        if self.glances_process is not None:
            print("Stopping glances...")
            self.stop_process(self.glances_process, grace_period=10)
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
            return self.llm_backend.is_glances_process(process)
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
        assert self.glances_tmp_file is not None
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
        assert self.power_tmp_file is not None
        with open(self.power_tmp_file, "rb") as f:
            data = f.read()
        logs = [plistlib.loads(d) for d in data.split(b'\x00')]
        for log in logs:
            # Convert datetime object to Unix (ns) for JSON
            # Need to set timezone to UTC manually
            fixed_ts = log["timestamp"].replace(tzinfo=timezone.utc).timestamp()
            log["timestamp"] = int(fixed_ts * SEC_TO_NANOSEC)
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
        if self.llm_backend is not None:
            self.llm_backend.stop()
        

    def start_profiling(self):
        """Main method to run the complete monitoring process"""
        try:
            # Start
            self.start_glances()
            if self.power_output_path is not None:
                self.start_power_measurement()
            if self.llm_backend is not None and self.preload_model_id is not None:
                self.llm_backend.start(self.preload_model_id)
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
--preload_model_id ollama_chat/qwen3:8b
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
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds) for target script. Default to no timeout.")
    parser.add_argument("--capture_stdout", action=argparse.BooleanOptionalAction, help="Print the output of the agent to stdout.")
    parser.add_argument("--llm_backend", type=str, default=None, choices=LlmBackend.subclasses.keys(), help="LLM backend to include in profiling results. Required for preload_model_id")
    parser.add_argument("--preload_model_id", type=str, default=None, help="If specified, start the llm backend, unload then preload the model, and also unload it at the end.")
    
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
        timeout=args.timeout,
        capture_stdout=args.capture_stdout,
        llm_backend=args.llm_backend,
        preload_model_id=args.preload_model_id,
    )
    success = profiler.start_profiling()
    if success:
        print("Profiling completed successfully!")
    else:
        print("Profiling failed or was interrupted")


if __name__ == "__main__":
    main()
