# efficient-agents

This repo contains measurement tools for profiling agent's system performance.

### Installation

#### Ollama

#### Llama.cpp

Use llama-swap.

#### Environment

Make sure to create and activate a virtual environment (we recommend `conda`).

#### Python packages

```
pip install fake_useragent llmlingua matplotlib ollama "smolagents[toolkit,litellm,telemetry]" pymupdf4llm wikipedia-api
```

#### Glances

We modified `glances` slightly to support our use case (Apple M-series laptops for now):
- Modified `outputs/glances_stdout_json.py` to output everything at once as a single JSON object instead of plugin by plugin. Also added a Unix timestamp (ns).
- Added `plugins/gpu/cards/apple.py` to extract GPU info via `ioreg -r -n AGXAcceleratorG13X -a`. (Also modified the gpu plugin `__init__.py` file.)
- Modified `plugins/sensors/__init__.py` to retrieve CPU/GPU temperature using `smctemp`.
- Modified `plugins/sensors/glances_batpercent.py` to retrieve various battery information via `ioreg -r -n AppleSmartBattery -a`

To install, navigate to the `glances` directory in our repo and run `pip install -e .`

#### smctemp

This is a command line program for retrieving CPU and GPU temperature on Mac. Install via `brew`:

```
brew tap narugit/tap
brew install narugit/tap/smctemp
```

#### iStats

This command line program is used for retrieving fan speed on Mac. Install via `gem install iStats`.

#### OpenInference

We use OpenInference's instrumentation to trace various agent frameworks. The installation steps above already covers smolagents. See [Arize-ai/openinference](https://github.com/Arize-ai/openinference/tree/main) for installation instructions specific to your target agent frameworks.