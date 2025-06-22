# efficient-agents

This repo contains measurement tools for profiling agent's system performance.

We modified `glances` slightly to support our use case (Apple M-series laptops for now):
- Modified `outputs/glances_stdout_json.py` to output everything at once as a single JSON object instead of plugin by plugin. Also added a Unix timestamp (ns).
- Added `plugins/gpu/cards/apple.py` to extract GPU info. (Also modified the gpu plugin `__init__.py` file.)
- Modified `plugins/sensors/__init__.py` to retrieve CPU/GPU temperature using smctemp.
- Modified `plugins/sensors/glances_batpercent.py` to retrieve various battery information.