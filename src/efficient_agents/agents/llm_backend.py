import ollama
import os
import psutil
import re
import requests
import shlex
import subprocess
import time

def process_exists(process_filter):
    """Check if there's any running process whose name matches the filter."""
    for proc in psutil.process_iter(["name"]):
        try:
            if bool(re.search(process_filter, proc.info.get("name", ""))):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def wait_for_http(url, timeout=60):
    start_time = time.time()
    while True:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {url}")
        time.sleep(0.5)

def wait_for_http_stop(url, timeout=60):
    start_time = time.time()
    while True:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code >= 500:
                return
        except requests.RequestException:
            return
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for {url} to stop.")
        time.sleep(0.5)

class LlmBackend:
    subclasses = {}

    def __init_subclass__(cls, key=None, **kwargs):
        super().__init_subclass__(**kwargs)
        reg_key = key or cls.__name__
        if reg_key in cls.subclasses:
            raise ValueError(f"Duplicate key '{reg_key}' registered")
        cls.subclasses[reg_key] = cls

    @classmethod
    def create(cls, key, *args, **kwargs):
        if key not in cls.subclasses:
            raise ValueError(f"No backend registered under key '{key}'")
        return cls.subclasses[key](*args, **kwargs)

    def __init__(self, process_filter):
        self.process_filter = process_filter

    def start(self, model_id, fresh=True):
        if fresh:
            self.stop()
        self._start(model_id)

    def _start(self, model_id):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def is_glances_process(self, process):
        raise NotImplementedError()


class LlmBackendOllama(LlmBackend, key="Ollama"):
    def __init__(self):
        super().__init__(process_filter=r"(o|O)llama")

    def _start(self, model_id):
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
        wait_for_http("http://127.0.0.1:11434/v1/models")
        print("Ollama started.")

        print(f"Preloading model {model_id}...")
        ollama.generate(model=model_id)
        print("Ollama started and preloaded.")

    def stop(self):
        models = ollama.ps()
        for model in models.models:
            model_id = model.model
            print(f"Stopping model {model_id}...")
            args = ["ollama", "stop", model_id]
            subprocess.run(args)
            time.sleep(1)

    def is_glances_process(self, process):
        target = process["name"]
        return bool(re.search(self.process_filter, target))


class LlmBackendLlamaSwap(LlmBackend, key="LlamaSwap"):
    def __init__(self, process_filter=r"(llama-swap|llama-server)"):
        super().__init__(process_filter=process_filter)
        self.addr = "http://127.0.0.1:8080"

    def _start(self, model_id):
        print("Starting llama-swap...")
        if not process_exists(self.process_filter):
            subprocess.Popen(
                ["llama-swap", "-config", "../config/llama_swap.yaml", "-watch-config"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wait_for_http(f"{self.addr}/health")
            print("llama-swap started.")
        else:
            print("llama-swap already started.")

        print(f"Preloading model {model_id}...")
        response = requests.post(
            f"{self.addr}/v1/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_id,
                "prompt": "",
                "max_tokens": 0, 
            }
        )
        response.raise_for_status()

        running = requests.get(f"{self.addr}/running").json()["running"]
        is_ready = False
        for run in running:
            if run["model"] == model_id and run["state"] == "ready":
                is_ready = True
                break
        if not is_ready:
            raise Exception(f"Model {model_id} is not running!")
        
        print("Model preloaded.")

    def stop(self):
        if process_exists(self.process_filter):
            print("Unloading llama-swap...")
            req = requests.get(f"{self.addr}/unload")
            req.raise_for_status()

    def is_glances_process(self, process):
        target = process["name"]
        return bool(re.search(self.process_filter, target))


class LlmBackendLlamaCpp(LlmBackendLlamaSwap, key="LlamaCpp"):
    def __init__(self):
        super().__init__(process_filter=r"(llama-swap|llama-server)")


class LlmBackendJetsonMLC(LlmBackend, key="JetsonMLC"):
    def __init__(self, container_name="mlc_container"):
        super().__init__(process_filter=r"mlc_llm")
        self.container_name = container_name

    def _start(self, model_id):
        print("Starting MLC server...")
        container_script = f'mlc_llm serve HF://{model_id} --host 0.0.0.0 --device cuda --mode interactive --overrides "max_total_seq_length=40960" > /dev/null 2>&1 &'
        script = f"docker exec {self.container_name} sh -c '{container_script}'"
        subprocess.run(shlex.split(script))
        wait_for_http("http://0.0.0.0:8000/v1/models")
        print("MLC server started.")

    def stop(self):
        print("Stopping MLC server...")
        subprocess.run(shlex.split(f"docker exec {self.container_name} pkill -f mlc_llm"))
        wait_for_http_stop("http://0.0.0.0:8000/v1/models")
        print("MLC server stopped.")
        
    def is_glances_process(self, process):
        target = " ".join(process["cmdline"])
        return bool(re.search(self.process_filter, target))