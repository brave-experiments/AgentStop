# Dzung Pham 2025

import json
import os
import platform
import plistlib
import re
import subprocess
from typing import Optional

"""Apple macOS integrated GPU extension unit for Glances GPU plugin.

The class grabs the stats using the ioreg tool. Only supports M-series laptops for now.
"""


class AppleIGPU:
    def __init__(self):
        pass

    def exit(self):
        pass

    def get_device_stats(self):
        stats = []
        if platform.system() != "Darwin":
            return stats
        
        device_stats = {
            "key": "gpu_id",
            "gpu_id": "apple0",
            "mem_raw": None,
            "proc": None,
            "temperature": None,
            "fan_speed": None,
        }

        try:
            ioreg = plistlib.loads(subprocess.check_output(["ioreg", "-r", "-n", "AGXAcceleratorG13X", "-a"]))
            perf = None
            for reg in ioreg:
                device_stats["name"] = reg.get("model") + " iGPU"
                perf = reg.get("PerformanceStatistics", None)
                if perf is not None:
                    in_use_sys_mem = perf.get("In use system memory", None)
                    alloc_sys_mem = perf.get("Alloc system memory", None)

                    device_stats["proc"] = perf.get("Device Utilization %", None)
                    device_stats["mem_raw"] = in_use_sys_mem
                    if in_use_sys_mem is not None and alloc_sys_mem is not None and alloc_sys_mem > 0:
                        device_stats["mem"] = in_use_sys_mem / alloc_sys_mem * 100

            try:
                temp = subprocess.check_output(["smctemp", "-g", "-i20", "-n4"], text=True, stderr=subprocess.DEVNULL)
                temp = float(temp.strip())
                device_stats["temperature"] = temp
            except:
                device_stats["temperature"] = None
        except:
            return stats
        
        stats.append(device_stats)
        return stats
