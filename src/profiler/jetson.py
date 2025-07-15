"""
Use tegrastats via jtop to capture performance metrics from Nvidia Jetson devices.
tegrastats is much faster than using jtop directly, but it has a bit less information.
To access power info, run with sudo.
"""

import argparse
import json
import time
import traceback
import signal
import sys
from threading import Event

from jtop.core.tegrastats import Tegrastats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tegrastats logger')
    parser.add_argument("--output_path", type=str, required=True, help="Path to store output in JSONL format.")
    parser.add_argument("--interval", type=int, default=100, help="Logging interval (milliseconds).")
    args = parser.parse_args()
    output_path = args.output_path
    interval = args.interval

    stop_event = Event()

    def handle_signal(sig, frame):
        stop_event.set()

    # Catch Ctrl+C
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    with open(output_path, "w") as f:
        def write_stats(stats):
            if "CPU" not in stats or len(stats["CPU"].keys()) == 0:
                return
            stats["time"] = time.time_ns()
            f.write(json.dumps(stats) + "\n")

        ts = Tegrastats(write_stats, ["/usr/bin/tegrastats"])
        try:
            if not ts.open(interval=interval / 1000.0):
                raise Exception("Failed to start tegrastats")
            print("Tegrastats logging started. Press Ctrl+C to stop.")
            stop_event.wait()  # block here until interrupted
        except:
            print(traceback.format_exc())
        finally:
            ts.close()
            f.flush()
            print("Tegrastats logging stopped.")
