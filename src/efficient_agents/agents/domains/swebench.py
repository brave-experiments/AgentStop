import json
import subprocess

from efficient_agents.agents.base import LogProbsCascadeAgent
from importlib import resources
from jinja2 import Template
from smolagents.local_python_executor import CodeOutput, PythonExecutor

timeout_template = """The last command <command>{{command}}</command> timed out and has been killed.
The output of the command was:
{% if output | length < 10000 -%}
<output>
{{output}}
</output>
{%- else -%}
<warning>Output was too long and has been truncated.</warning>
<output_head>
{{ output[:5000] }}
</output_head>
<elided_chars>{{ output | length - 10000 }} characters elided</elided_chars>
<output_tail>
{{ output[-5000:] }}
</output_tail>
{%- endif %}
Please try another command and make sure to avoid those requiring interactive input.
"""

full_output_template = """<returncode>{{output.returncode}}</returncode>
<output>
{{ output.output -}}
</output>
"""

# Run bash commands in Docker environment
# Based on https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/docker.py
class BashExecutor(PythonExecutor):
    def __init__(self, docker_id, config):
        super().__init__()
        assert docker_id is not None
        self.docker_id = docker_id
        self.timeout = config["environment"]["timeout"]
        self.env = config["environment"]["env"]
        self.timeout_template = Template(timeout_template)
        self.output_template = Template(config["agent"]["action_observation_template"])
        self.full_output_template = Template(full_output_template)

    def __call__(self, code: str) -> CodeOutput:
        cmd = ["docker", "exec"]
        for k, v in self.env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.extend([self.docker_id,  "timeout", "-k", str(self.timeout + 10) + "s", str(self.timeout) + "s", "bash", "-lc", code])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            is_finished = self.is_finished(result.stdout)
            output_obj = {"returncode": result.returncode, "output": result.stdout}
            if not is_finished:
                output = self.output_template.render(output=output_obj)
            else:
                output = self.full_output_template.render(output=output_obj)
            return CodeOutput(output=output if is_finished else None, logs=output, is_final_answer=is_finished)
        except subprocess.TimeoutExpired as e:
            output = ""
            if e.stdout:
                output = e.stdout.decode("utf-8", errors="replace")
            msg = self.timeout_template.render(command=code, output=output)
            raise Exception(msg)

    def is_finished(self, output):
        lines = output.lstrip().splitlines(keepends=True)
        return len(lines) > 0 and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

    def send_variables(self, variables):
        pass

    def send_tools(self, tools):
        pass


class SweBenchLogProbsAgent(LogProbsCascadeAgent, key="swebench_logprobs"):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_tokens=4096, **kwargs)

        assert kwargs["docker_id"] is not None
        self.docker_id = kwargs["docker_id"]
        with resources.open_text("efficient_agents.config", "mini_swe_agent.json") as f:
            config = json.load(f)
        self.agent.python_executor = BashExecutor(self.docker_id, config)
        self.agent.prompt_templates["system_prompt"] = config["agent"]["system_template"]
        self.agent.code_block_tags = ("```bash", "```")
        self.task_template = Template(config["agent"]["instance_template"])
    
    def run(self, prompt, **kwargs):
        task = self.task_template.render(task=prompt)
        return super().run(task, **kwargs)
