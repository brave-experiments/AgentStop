import ollama
import os
import re
import requests
import shlex
import subprocess
import time

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

    def _start(model_id):
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
            subprocess.run(["ollama", "stop", model_id])
            time.sleep(1)

    def is_glances_process(self, process):
        target = process["name"]
        return bool(re.search(self.process_filter, target))


class LlmBackendJetsonMLC(LlmBackend, key="JetsonMLC"):
    def __init__(self, container_name="mlc_container"):
        super().__init__(process_filter=r"mlc_llm")
        self.container_name = container_name

    def _start(self, model_id):
        print("Starting MLC server...")
        container_script = f"mlc_llm serve HF://{model_id} --host 0.0.0.0 --device cuda --mode interactive > /dev/null 2>&1 &"
        script = f"docker exec {self.container_name} sh -c '{container_script}'"
        subprocess.run(shlex.split(script))
        wait_for_http("http://0.0.0.0:8000/v1/models")
        print("MLC server started.")

    def stop(self):
        print("Stopping MLC server...")
        subprocess.run(shlex.split(f"docker exec {self.container_name} pkill -f mlc_llm"))
        time.sleep(1)
        
    def is_glances_process(self, process):
        target = " ".join(process["cmdline"])
        return bool(re.search(self.process_filter, target))