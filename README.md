# AgentStop

This repo contains the code and data for our paper "AgentStop: Terminating Local AI Agents Early to Save Energy in Consumer Device" (ACM CAIS '26).

### Requirements

Below are the hardware and software requirements to run and profile our agents.

Hardware:
- An Apple Silicon machine (M-series) or an NVIDIA Jetson with at least 24GB of unified RAM (32GB or more is recommended). The code has been tested on an Apple M1 Max (64GB RAM) and an NVIDIA Orin AGX (64GB RAM, JetPack 6.2.2, L4T Version 36.5.0). Other Linux-based machines capable of GPU inference should also be generally compatible, but will likely require some modifications to our code to be able to correctly extract power, thermal, fan speed, etc.
- At least 24GB of free disk space (mostly for the LLM models) for Q&A task. At least 120GB of disk and 48GB of RAM if you want to evaluate on SWE-Bench.
- 8 CPU cores or above are highly recommended.

Software:
- sudo access (to enable full access to power measurements)
- conda (to set up environment, Miniconda is easiest)
- VSCode or other apps that can run Jupyter Notebook (for analyzing profiling results and training AgentStop classifiers)

### Installation

#### Conda environment

Create a new conda environment with Python 3.10 or above, e.g.:

`conda create -n agstop python=3.10`

Activate your environment:

`conda activate agstop`

#### Download and install AgentStop

Clone this repo to your machine, cd into the folder, then run `install.sh` (`chmod +x install.sh` if needed).

#### Llama.cpp

This is our main LLM backend.
For Mac (Apple Silicon), you can follow the instructions at https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md.
We used Homebrew to install, e.g.: `brew install llama.cpp`.
Our llama.cpp version was b7770.

For NVIDIA Jetson devices, you will need to install from source.
Follow the official guide here: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda.
We also include our build commands here for reference (it will build a specific release):

```
RELEASE="b8762"

cd llama.cpp
git fetch --tags
git checkout $RELEASE

cmake -B build-$RELEASE \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_F16=ON

cmake --build build-$RELEASE --config Release -j$(nproc)
```

Make sure to add the binary to your PATH:
`export PATH=$HOME/llama.cpp/build-b8762/bin:$PATH`

To test if installation is successful, run `llama-server --version`. You should see something like this:
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 62827 MiB):
  Device 0: Orin, compute capability 8.7, VMM: yes, VRAM: 62827 MiB
version: 8762 (073bb2c20)
built with GNU 11.4.0 for Linux aarch64
```

#### Models

Create a folder to store your models, e.g., `/path/to/models`.
We mainly use Qwen3 model checkpoints from Ollama.
First, install Ollama for your device via `curl -fsSL https://ollama.com/install.sh | sh`.
Then start with `ollama start` and then pull models (e.g., `ollama pull qwen3:1.7b` or `qwen3:30b-a3b-instruct-2507-q4_K_M` or `qwen3:30b-coder`).
Ollama's model checkpoints need to be symlinked to the llamacpp model path that you chose.
We include a Python script `scripts/map_ollama_models.py` to do this automatically.
Example usage: `python map_ollama_models.py /path/to/models`

Ollama's models (such as Qwen3.5) might not always be compatible with Llama.cpp.
If you run into an error, you can download models from Unsloth instead.
We recommend sticking to Unsloth. Ollama was used in early work, so we still kept it to preserve consistency for our paper.

#### Llama-swap

We use llama-swap to switch between models more easily.
We used Homebrew to install (see instructions at https://github.com/mostlygeek/llama-swap#homebrew-install-macoslinux).

Once this is done, you will need to edit the `config/llama_swap.yaml` file in our project's folder to update the `llamacpp_path` macro to your llama.cpp model directory.

#### SWE-Bench (Optional)

To run profiling on SWE-Bench, follow the instruction here to install SWE-Bench: https://github.com/SWE-bench/SWE-bench#-set-up.
We use Docker to run the benchmark.
On Mac, you will also need to install `colima`: https://github.com/abiosoft/colima.
Next, run these commands to start a Colima Linux VM and configure docker to use colima:

```
colima start --cpu 8 --memory 16 --disk 120
docker context use colima
```

#### Jetson-stats (required for profiling NVIDIA Jetson)

Follow the instruction at https://github.com/rbonghi/jetson_stats/.
We used pip with sudo to install, e.g., `sudo pip install -U jetson-stats`.

#### Additional notes for Mac

##### Glances

We use `glances` (v4.3.1) to log a variety of stats.
We modified it slightly to support our use case (Apple M-series laptops):
- Modified `outputs/glances_stdout_json.py` to output everything at once as a single JSON object instead of plugin by plugin. Also added a Unix timestamp (ns).
- Added `plugins/gpu/cards/apple.py` to extract GPU info via `ioreg -r -n AGXAcceleratorG13X -a`. (Also modified the gpu plugin `__init__.py` file.)
- Modified `plugins/sensors/__init__.py` to retrieve CPU/GPU temperature using `smctemp`.
- Modified `plugins/sensors/glances_batpercent.py` to retrieve various battery information via `ioreg -r -n AppleSmartBattery -a`

If you make any changes, to (re-)install, navigate to the `glances-4.3.1` directory in our repo and run `pip install -e .`

##### smctemp

This is a command line program for retrieving CPU and GPU temperature on Mac. Install via `brew`:

```
brew tap narugit/tap
brew install narugit/tap/smctemp
```

##### iStats

This command line program is used for retrieving fan speed on Mac. Install via `gem install iStats`.

### Profiling

We included the task data and also our own profiling results in `experiment_data/`. Before proceeding, run the script `unzip_all.sh` in that folder.

#### Q&A tasks with FRAMES

First, you will need to create an .env file in the project repo and add a Brave Search API key:

`BRAVE_API_KEY=<API_KEY>`

Brave Search API keys can be obtained by signing up at: https://api-dashboard.search.brave.com/documentation/pricing.
It includes $5 free credit per month, which is worth 1000 queries and is sufficient for 100-200 tasks only. If you want to run the entire benchmark, you will need to pay extra.

If you want to evaluate the agent's output yourself, you will need to add a valid ANTHROPIC_API_KEY to the .env file. We use Claude Haiku to evaluate the agent's answers.

Now, to start profiling on FRAMES using Qwen3-30B-A3B, run the following:

```
cd scripts
./profile_frames_llama_cpp.sh
```

The raw log will be stored in `logs/frames/llamacpp_qwen3_30b`. For each task, you can find the compressed raw traces as well as the analysis data (power graphs, summaries, etc.).

To stop the profiling, you will need to manually kill all the processes (grep for python, powermetrics, glances, etc).

If you want to change the model or anything else, you can edit the script.

#### Coding task with SWE-Bench Verified

Make sure to install SWE-Bench first (follow the instruction above).
If you are on Mac, also make sure to install Colima and configure it correctly.

To start profiling on SWE-Bench Verified using Qwen3-30B-A3B-Coder, run:
```
cd scripts
./profile_swebench_llama_cpp.sh
```

The raw log will be stored in `logs/swebench/llamacpp_qwen3_coder_30b`. Similar to QA, for each task, you can find the compressed raw traces as well as the analysis data (power graphs, summaries, etc.).

To stop the profiling, you will need to manually kill all the processes (grep for python, powermetrics, glances, any docker containers still running).