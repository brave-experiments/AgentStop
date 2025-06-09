#!/usr/bin/env python3
"""
System performance profiling using glances.
"""

import json
import os
import psutil
import signal
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path

class Profiler:
    def __init__(self, target_script, args, process_filter, log_output_path, frequency=0.1):
        self.target_script = target_script
        self.args = args
        self.glances_process = None
        self.glances_output_file = None
        self.target_process = None
        self.process_filter = process_filter
        self.target_pids = set()  # Track PIDs created by the target script
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.log_output_path = log_output_path
        self.frequency = frequency
        
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

    def start_glances(self):
        """Start glances with json export"""
        print("Starting glances monitoring...")
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                self.glances_output_file = tmp.name
            plugins = "cpu,gpu,mem,memswap,sensors,network,diskio,processlist"
            with open(self.glances_output_file, "w") as f:
                self.glances_process = subprocess.Popen([
                    'glances',
                    '--stdout-json', plugins,
                    '--process-filter', self.process_filter,
                    '--time', str(self.frequency),
                ],
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                )
                
                # Give glances a moment to start
                time.sleep(2)
                print(f"Started glances logging to {self.glances_output_file} (PID = {self.glances_process.pid})")
                return True
        except Exception as e:
            print(f"Error starting glances: {e}")
            return False
    
    def run_target_script(self):
        """Run the target Python script and track its process tree"""
        print(f"Starting target script: {self.target_script}")
        start_time = time.time_ns()
        
        try:
            # Run the target script
            self.target_process = subprocess.Popen([
                sys.executable, self.target_script, *self.args
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Start tracking the process tree
            self.track_process_tree()
            
            # Wait for the script to complete
            stdout, stderr = self.target_process.communicate()
            end_time = time.time_ns()
            self.stop_monitoring = True
            
            print(f"Target script completed in {round((end_time - start_time) / 1_000_000_000)}s")
            print(f"Tracked {len(self.target_pids)} process IDs: {sorted(list(self.target_pids))}")
            
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
    
    def stop_glances(self):
        """Stop the glances process"""
        if self.glances_process:
            print("Stopping glances...")
            self.glances_process.send_signal(signal.SIGINT)
            try:
                self.glances_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.glances_process.kill()
            time.sleep(2)
            print("Glances stopped")
    
    def save_log(self):
        """Filter the glances log for processes created by the target script and time period"""
        # Read
        logs = []
        with open(self.glances_output_file, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line.strip()))
        
        # Filter
        for l in logs:
            l["processlist"] = [p for p in l["processlist"] if p["pid"] in self.target_pids]
        logs = [l for l in logs if len(l["processlist"]) > 0]

        # Save
        log_output_path = Path(self.log_output_path)
        log_output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_output_path.open("w", encoding="utf-8") as f:
            for item in logs:
                f.write(json.dumps(item) + "\n")
        print(f"Log saved to {log_output_path.resolve()}")

        # Clean up
        os.remove(self.glances_output_file)
        self.glances_output_file = None
    
    def start_profiling(self):
        """Main method to run the complete monitoring process"""
        try:
            if not self.start_glances():
                return False
            self.run_target_script()
            self.stop_glances()
            self.save_log()
            return True
        except Exception as e:
            print(f"Error during monitoring: {e}")
            self.stop_glances()
            return False

def main():
    # Configuration
    target_script = "../agents/web/smol_agents.py"
    args = [
        "--agent_name", "BasicMlx",
        "--model_id", "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "--model_type", "mlx",
        "--trace_path", "./mlx_trace.json",
        "--prompt", "If the US keeps its 2024 growth rate, how many years will it take for the GDP to double?"
    ]
    process_filter = ".*python.*"
    log_output_path = "logs/basicmlx_log.jsonl"
    
    # Create and run the profiler
    profiler = Profiler(target_script, args, process_filter, log_output_path, frequency=0.5)
    success = profiler.start_profiling()
    
    if success:
        print("Profiling completed successfully!")
    else:
        print("Profiling failed or was interrupted")

if __name__ == "__main__":
    main()