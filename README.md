# efficient-agents

This repo contains measurement tools for profiling agent's system performance.

We modified `glances` slightly to support our usecase:
- Added a file called `plugins/gpu/cards/apple.py` to extract Apple Silicon's GPU info. Also modified the gpu plugin.
- Modified `outputs/glances_stdout_json.py` to output everything at once as a single JSON object instead of plugin by plugin. Also added a Unix timestamp (ns).