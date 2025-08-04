"""
Analyze glances log and agent's trace.
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import ollama
import pandas as pd
import re
import seaborn as sns
import sys

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes
)
from tqdm import tqdm
from collections import deque
from pathlib import Path

SEC_TO_NANOSEC = 1_000_000_000
GB_TO_BYTE = 1024 * 1024 * 1024
KB_TO_BYTE = 1024

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
FIRST_TOKEN_TS = "first_token_ts"
PREFILL_COUNT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
GENERATION_COUNT = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
CACHED_COUNT = "cached_count"
PREFILL_TPS = "prefill_tps"
GENERATION_TPS = "generation_tps"

def avg_med_std(l):
    res = []
    for values, label in l:
        res.extend([
            (f"avg_{label}", values.mean()),
            (f"med_{label}", values.median()),
            (f"std_{label}", values.std()),
        ])
    return res

"""
Approximates total energy usage using trapezoidal integration.
ts is assumed to be an array of timestamps (nanoseconds).
If ts_is_elapsed is set to True, ts would be treated as elapsed time.
power_data is assumed to be an array of tuple (power, label, token_count)
power is assumed to be an array of floats (milliwatt)
This is because timestamp can be unevenly spaced and power can fluctuate widely.
Returns total energy in mWh (equivalent to 3.6 joules)
"""
def energy(ts, power_data, ts_is_elapsed=False):
    assert len(power_data) > 0 and len(ts) == len(power_data[0][0])
    ts = np.array(ts) / SEC_TO_NANOSEC / 3600
    if ts_is_elapsed:
        ts = ts.cumsum()

    res = []
    for power, label, token_count in power_data:
        e = np.trapz(np.array(power), x=ts)
        res.append((f"energy_{label}_mWh", e))
        if token_count is not None:
            res.append((f"energy_{label}_mWh_per_token", e / token_count))
    return res

class DeviceType:
    APPLE_LAPTOP = "apple_laptop"
    JETSON = "jetson"
    ALL = [APPLE_LAPTOP, JETSON]

class Analyzer:
    def __init__(
        self,
        device_type,
        glances_log_path,
        agent_trace_path,
        power_log_path=None,
        model_id=None, # LLM model for analyzing
        full_execution=False,
        output_dir="./analysis_logs",
        output_ext=["png"],
        display_plots=False,
        display_summary=False,
    ):
        assert device_type in [DeviceType.APPLE_LAPTOP, DeviceType.JETSON]
        self.device_type = device_type
        with Path(glances_log_path).open("r", encoding="utf-8") as f:
            self.glances_log = [json.loads(l) for l in f if l.strip()]
        with Path(agent_trace_path).open("r", encoding="utf-8") as f:
            self.agent_trace = sorted(json.load(f), key=lambda t: t["start_time"])
        
        self.power_df = None
        if power_log_path is not None:
            with Path(power_log_path).open("r", encoding="utf-8") as f:
                self.power_log = [json.loads(l) for l in f if l.strip()]
        
        if model_id is not None: # Make sure model_id is correct
            ollama.show(model=model_id)
        self.model_id = model_id

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.full_execution = full_execution
        self.processed_agent_trace = None
        self.agent_task = None
        self.agent_output = None
        self.display_plots = display_plots
        self.display_summary = display_summary

    def process_glances_log(self):
        data = [{
            "timestamp": l["timestamp"],
            "gpu_mem": l["gpu"][0]["mem_raw"] if len(l["gpu"]) > 0 else 0,
            "gpu_mem_pct": l["gpu"][0]["mem"] if len(l["gpu"]) > 0 else 0,
            "gpu_usage": l["gpu"][0]["proc"] if len(l["gpu"]) > 0 else 0,
            "gpu_temp": l["gpu"][0]["temperature"] if len(l["gpu"]) > 0 else 0,
            "cpu_total_pct": l["cpu"]["total"],
            "smctemp_cpu": next((s["value"] for s in l["sensors"] if s["label"] == "smctemp"), 0),
            "battery_percent": next((s["value"] for s in l["sensors"] if s["label"] == "Battery"), 0),
            "battery_discharge": next((-min(s["value"], 0) for s in l["sensors"] if s["label"] == "Battery Current"), 0),
            "battery_temp": next(((s["value"] / 10 - 273.15) for s in l["sensors"] if s["label"] == "Battery Temperature"), 0),
            "fan_max_speed": max((s["value"] for s in l["sensors"] if s["label"].startswith("Fan")), default=0),
            "processes_count": len([p for p in l["processlist"] if p["status"] != "Z"]),
            "processes_cpu_pct": sum(p["cpu_percent"] for p in l["processlist"] if p["status"] != "Z"),
            "processes_num_threads": sum(p["num_threads"] for p in l["processlist"] if p["status"] != "Z"),
            "processes_mem": sum(p["memory_info"]["rss"] for p in l["processlist"] if p["status"] != "Z"),
            "mem_used": l["mem"]["used"],
            "memswap_used": l["memswap"]["used"],
            "diskio_read_bytes": sum(d["read_bytes"] for d in l["diskio"]),
            "diskio_write_bytes": sum(d["write_bytes"] for d in l["diskio"]),
            "diskio_read_bytes_per_sec": sum(d.get("read_bytes_rate_per_sec", 0) for d in l["diskio"]),
            "diskio_write_bytes_per_sec": sum(d.get("write_bytes_rate_per_sec", 0) for d in l["diskio"]),
            "network_sent_all": sum(itf["bytes_sent"] for itf in l["network"]) / KB_TO_BYTE,
            "network_recv_all": sum(itf["bytes_recv"] for itf in l["network"]) / KB_TO_BYTE,
        } for l in self.glances_log]
        df = pd.DataFrame.from_dict(data)

        start_time = df["timestamp"].iloc[0]
        df["timestamp_plot"] = (df["timestamp"] - start_time) / SEC_TO_NANOSEC
        df["gpu_mem_plot"] = df["gpu_mem"] / GB_TO_BYTE
        df["processes_mem_plot"] = df["processes_mem"] / GB_TO_BYTE
        df["mem_used_plot"] = df["mem_used"] / GB_TO_BYTE
        df["memswap_used_plot"] = df["memswap_used"] / GB_TO_BYTE
        
        df["diskio_read_bytes_plot"] = df["diskio_read_bytes"] / GB_TO_BYTE
        df["diskio_write_bytes_plot"] = df["diskio_write_bytes"] / GB_TO_BYTE
        df["diskio_read_bytes_per_sec_plot"] = df["diskio_read_bytes_per_sec"] / GB_TO_BYTE
        df["diskio_write_bytes_per_sec_plot"] = df["diskio_write_bytes_per_sec"] / GB_TO_BYTE

        self.glances_df = df

    def process_power_log(self):
        if self.glances_df is None:
            raise Exception("Needs processed glances to process power log")

        if self.device_type == DeviceType.APPLE_LAPTOP:
            self.process_powermetrics_log()
        elif self.device_type == DeviceType.JETSON:
            self.process_tegrastats_log()

    def process_powermetrics_log(self):
        glances_start_time = self.glances_df["timestamp"].iloc[0]
        glances_end_time = self.glances_df["timestamp"].iloc[-1]
        data = [
            {
                "timestamp": l["timestamp"],
                "elapsed_ns": l["elapsed_ns"],
                "cpu_energy": l["processor"]["cpu_energy"],
                "cpu_power": l["processor"]["cpu_power"],
                "gpu_energy": l["processor"]["gpu_energy"],
                "gpu_power": l["processor"]["gpu_power"],
                "combined_power": l["processor"]["combined_power"],
            } for l in self.power_log
            if glances_start_time <= l["timestamp"] <= glances_end_time
        ]
        df = pd.DataFrame.from_dict(data)

        # Need to better approximate timestamps since the precision from powermetrics is 1 sec
        glances_start_time = self.glances_df["timestamp"].iloc[0]
        elapsed_times = df["elapsed_ns"].to_list()
        elapsed_times[0] = 0
        coarse_times = (df["timestamp"] - glances_start_time).to_list()
        n = df.shape[0]
        corrected_time = [None] * n

        # Find index of first change in timestamp
        for i in range(n - 1):
            if coarse_times[i] != coarse_times[i + 1]:
                first_ts_change_idx = i + 1
                break
        
        # For each timestamp change, snap the actual time to the new timestamp
        # and add elapsed_ns until next timestamp change
        i = first_ts_change_idx
        while i < n:
            cur_time = coarse_times[i]
            prev_time = coarse_times[i - 1]
            if cur_time == prev_time:
                i += 1
                continue
            corrected_time[i] = cur_time
            j = i + 1
            elapsed = 0
            while j < n and coarse_times[j] == cur_time:
                elapsed += elapsed_times[j]
                assert elapsed < SEC_TO_NANOSEC, "Elapsed timed must be < 1s during the same timestamp!"
                corrected_time[j] = cur_time + elapsed
                j += 1
            i = j
        
        # Fix the time for the first few entries before the first ever ts change
        for i in range(first_ts_change_idx - 1, -1, -1):
            corrected_time[i] = corrected_time[i + 1] - elapsed_times[i + 1]

        df["timestamp_plot"] = np.array(corrected_time) / SEC_TO_NANOSEC
        self.power_df = df

    def process_tegrastats_log(self):
        glances_start_time = self.glances_df["timestamp"].iloc[0]
        glances_end_time = self.glances_df["timestamp"].iloc[-1]
        
        sample = self.power_log[0]

        data = []
        for l in self.power_log:
            ts = l["timestamp"]
            if ts < glances_start_time:
                continue
            elif ts > glances_end_time:
                break

            d = {
                "timestamp": ts,
                "gpu_usage": l["GR3D"]["val"],
                "gpu_freq": l["GR3D"]["frq"] / 1000.0,
            }

            for k, v in l["WATT"].items():
                if k != "NC": # Skip "Not Connected"
                    d[f"power_{k}"] = v["cur"]
            
            for k, v in l["TEMP"].items():
                d[f"temp_{k}"] = max(v, 0.0)
            
            data.append(d)

        df = pd.DataFrame.from_dict(data)
        df["timestamp_plot"] = (df["timestamp"] - glances_start_time) / SEC_TO_NANOSEC
        self.power_df = df

    def describe_agent_action(self, text):
        if self.model_id is None:
            return ""
        
        prompt = f"""You are an expert in analyzing computer programs and AI behaviors.
You will be provided with the output message of some AI assistant.
Your task is to describe succinctly in only 5-7 words what the AI assistant is doing.
Do not include specific details, just focus on the general action.
Use abbreviations as much as possible.

Here are some examples of good descriptions:
<examples>
Searching for <SOMETHING>.
Planning for <TASK>.
Reasoning about <TASK>.
Calling <TOOL>.
<examples>

Here's the message of the AI assistant:
<message>
{text}
</message>

Write your short description here:
"""
        res = ollama.generate(
            model=self.model_id,
            prompt=prompt,
            think=False,
            options={"temperature": 0.0},
        )
        return res.response

    def process_agent_trace(self):
        id_to_child_id = {t["span_id"]: [] for t in self.agent_trace}
        root_id = None
        for t in self.agent_trace:
            parent_id = t["parent_span_id"]
            if parent_id is not None:
                id_to_child_id[parent_id].append(t["span_id"])
            else:
                root_id = t["span_id"]
        
        id_to_level = {t["span_id"]: None for t in self.agent_trace}
        id_to_level[root_id] = 0
        queue = deque([root_id])
        while len(queue) > 0:
            cur_node = queue.popleft()
            level = id_to_level[cur_node]
            children = id_to_child_id[cur_node]
            if len(children) > 0:
                for child in children:
                    id_to_level[child] = level + 1
                queue.extend(children)

        id_to_trace = {t["span_id"]: t for t in self.agent_trace}
        
        prefix = None
        base_prefill_tps = None
        cur_model = None
        start_time = self.glances_df["timestamp"].iloc[0]
        processed = []
        for t in tqdm(self.agent_trace):
            attributes = t["attributes"]
            kind = attributes[SPAN_KIND]
            if kind == AGENT:
                agent_task = attributes.get(INPUT_VALUE, None)
                if agent_task is not None:
                    try:
                        agent_task = json.loads(agent_task).get("task", None)
                    except:
                        pass
                
                agent_output = attributes.get(OUTPUT_VALUE, None)
                if agent_output is not None:
                    pattern = r"output=('[^']*'|\"[^\"]*\"|\[[^\]]*\]|\{[^}]*\}|\([^)]*\)|[^,)]+)"
                    match = re.search(pattern, agent_output)
                    if match:
                        agent_output = match.group(1)
                
                self.agent_task = agent_task
                self.agent_output = agent_output


            if (t["end_time"] - t["start_time"]) / SEC_TO_NANOSEC < 0.1:
                continue

            attributes = t["attributes"]
            kind = attributes[SPAN_KIND]
            desc = None
            extra_data = {}

            if kind == LLM:
                # Copy attributes from parent LLM span
                parent_id = t["parent_span_id"]
                if parent_id is not None:
                    parent_attr = id_to_trace[parent_id]["attributes"]
                    if parent_attr[SPAN_KIND] == LLM:
                        for k, v in parent_attr.items():
                            if k not in attributes:
                                attributes[k] = v

                # Check if parent of LLM to save some work
                is_parent_of_LLM = False
                for child_id in id_to_child_id[t["span_id"]]:
                    child_attr = id_to_trace[child_id]["attributes"]
                    if child_attr[SPAN_KIND] == LLM:
                        is_parent_of_LLM = True
                        break

                # Create description text for span
                agent_desc = ""
                token_desc = ""
                model_name = attributes.get(LLM_MODEL_NAME, None)
                input_msg = attributes.get("input.value", None)
                output_msg = attributes.get("output.value", None)
                input_tokens = attributes.get(PREFILL_COUNT, None)
                output_tokens = attributes.get(GENERATION_COUNT, None)
                pf_tps = attributes.get(PREFILL_TPS, None)
                gen_tps = attributes.get(GENERATION_TPS, None)

                if pf_tps is not None and input_msg is not None:
                    input_msg = str(json.loads(input_msg)["messages"])
                    pf_time = input_tokens * 1.0 / pf_tps
                    cached_tokens = 0

                    if base_prefill_tps is None or (cur_model is not None and cur_model != model_name):
                        prefix = [{"text": input_msg, "tokens": input_tokens}]
                        base_prefill_tps = pf_tps
                        cur_model = model_name

                    elif pf_tps > base_prefill_tps * 1.1: # Likely cached
                        for p in prefix:
                            if input_msg.startswith(p["text"][:-1]): # Ignore the closing ]
                                cached_tokens = p["tokens"]
                                pf_tps = (input_tokens - cached_tokens) / pf_time
                                p["text"] = input_msg
                                p["tokens"] = input_tokens
                                break
                    else:
                        prefix.append({"text": input_msg, "tokens": input_tokens})
                
                if output_msg is not None and not is_parent_of_LLM:
                    agent_desc = self.describe_agent_action(output_msg)
                if input_tokens is not None and output_tokens is not None:
                    if pf_tps is not None and gen_tps is not None:
                        token_desc = f"Tkn: {input_tokens} in ({cached_tokens} ca.) ({round(pf_tps, 1)}/s), {output_tokens} out ({round(gen_tps, 1)}/s)"
                    else:
                        token_desc = f"Tkn: {input_tokens} in, {output_tokens} out"
                desc = f"{agent_desc}\n[{token_desc}]".strip()

                # Add extra data
                if FIRST_TOKEN_TS in attributes:
                    extra_data.update({
                        FIRST_TOKEN_TS: (attributes[FIRST_TOKEN_TS] - start_time) / SEC_TO_NANOSEC,
                        PREFILL_COUNT: input_tokens,
                        CACHED_COUNT: cached_tokens,
                        GENERATION_COUNT: output_tokens,
                        PREFILL_TPS: pf_tps,
                        GENERATION_TPS: gen_tps,
                    })
            
            elif kind == TOOL:
                if "search" in t["name"].lower():
                    input_value = attributes.get("input.value", None)
                    if input_value is not None:
                        try:
                            input_value = json.loads(input_value)
                            desc = input_value.get("query", None)
                            if desc is None and "kwargs" in input_value:
                                desc = input_value["kwargs"].get("query", None)
                            if desc is None and "args" in input_value:
                                args = input_value["args"]
                                if type(args) is str:
                                    desc = args
                                elif type(args) is list and len(args) > 0:
                                    desc = args[0]
                        except:
                            pass
            
            span = {
                "name": attributes.get(LLM_MODEL_NAME, t["name"]),
                "level": id_to_level[t["span_id"]],
                "start_time": (t["start_time"] - start_time) / SEC_TO_NANOSEC,
                "end_time": (t["end_time"] - start_time) / SEC_TO_NANOSEC,
                "kind": kind,
                "desc": desc.strip() if desc is not None else "",
                "extra_data": extra_data,
            }
            processed.append(span)

        self.processed_agent_trace = processed


    def get_topmost_spans(self):
        spans = self.processed_agent_trace
        top_spans = []
        i = 0
        while i < len(spans):
            top = spans[i]
            if i == len(spans) - 1:
                top_spans.append(top)
                break

            for j in range(i + 1, len(spans)):
                cur = spans[j]
                if cur["start_time"] >= top["end_time"]: # Not overlapping
                    i = j
                    break
                elif cur["level"] > top["level"]: # Higher level
                    top = cur
                i = j + 1

            top_spans.append(top)

        return top_spans

    def wrap_text_to_axis_width(self, ax, text, x_start, x_end, fontsize=None):
        """
        Wrap text so that it fits within the horizontal space between x_start and x_end in an axes.
        
        Parameters:
        - ax: matplotlib Axes
        - text: str, the original unwrapped text
        - x_start: float, starting x-value (data units)
        - x_end: float, ending x-value (data units)
        - fontsize: optional dict, font settings (same as used in ax.text)
        
        Returns:
        - wrapped_text: str, text with line breaks inserted to fit within the pixel width
        """

        renderer = ax.figure.canvas.get_renderer()
        max_width_px = ax.transData.transform((x_end, 0))[0] - ax.transData.transform((x_start, 0))[0]

        # Estimate the width of each word and wrap accordingly
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            text_obj = ax.text(0, 0, test_line, fontsize=fontsize)
            text_width = text_obj.get_window_extent(renderer=renderer).width
            text_obj.remove()

            if text_width <= max_width_px:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def plot_trace_execution_timeline_full(self, ax):
        # Color
        agent_trace = self.processed_agent_trace
        stage_names = list({t["name"] for t in agent_trace})
        palette = sns.color_palette("Set2", len(stage_names))
        stage_names_ordered = [
            (
                name,
                min(t["level"] for t in agent_trace if t["name"] == name),
                min(t["start_time"] for t in agent_trace if t["name"] == name)
            ) for name in stage_names
        ]
        stage_names_ordered = sorted(stage_names_ordered, key=lambda e: (e[1], e[2]))
        stage_to_color = {stage: palette[i] for i, stage in enumerate([name for (name, _, _) in stage_names_ordered])}

        # Spacing and level positions
        max_level = max(t["level"] for t in agent_trace)
        spacing = 0.3
        bar_height = 0.2
        level_positions = [i * spacing for i in range(max_level + 1)]

        # Config axis
        ax.set_ylim(-spacing, level_positions[-1] + spacing)
        ax.set_yticks(level_positions)
        ax.set_yticklabels([f"{i}" for i in range(max_level + 1)])
        ax.set_ylabel("Call Stack Level")

        # Plot stages
        grouped_agent_trace = [
            [t for t in agent_trace if t["level"] == level]
            for level in range(max_level + 1)
        ]
        for level_group in grouped_agent_trace:
            for i, t in enumerate(level_group):
                stage = t["name"]
                level = t["level"]
                t_start = t["start_time"]
                t_end = t["end_time"]
                duration = max(t_end - t_start, 0.2)
                y_pos = level_positions[level]
                color = stage_to_color[stage]
                
                # Draw the bar
                ax.barh(y_pos, duration, left=t_start, height=bar_height, color=color, edgecolor=None)

                # Draw border
                if i > 0 and (t_start - level_group[i-1]["end_time"]) < 0.1:
                    ax.vlines(t_start, y_pos - bar_height / 2 + 0.001, y_pos + bar_height / 2, color='black', linewidth=0.5)

                # Draw the text if the bar is long enough
                bar_pixel_width = ax.transData.transform((t_end, 0))[0] - ax.transData.transform((t_start, 0))[0]
                text_obj = ax.text(0, 0, stage, fontsize=8)
                text_pixel_width = text_obj.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
                text_obj.remove()
                if text_pixel_width <= bar_pixel_width * 1.25:
                    ax.text((t_start + t_end) / 2, y_pos, stage, ha='center', va='center', fontsize=8)

        # Legend
        legend_handles = [mpatches.Patch(color=color, label=stage) for stage, color in stage_to_color.items()]
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=7, fontsize=8, frameon=False)

    def plot_trace_execution_timeline_summarized(self, ax, other_axes=None):
        # Color
        agent_trace = self.get_topmost_spans()
        stage_names = list({t["name"] for t in agent_trace})
        palette = sns.color_palette("Set2", len(stage_names) + 2)
        stage_names_ordered = [
            (
                name,
                min(t["level"] for t in agent_trace if t["name"] == name),
                min(t["start_time"] for t in agent_trace if t["name"] == name)
            ) for name in stage_names
        ]
        stage_names_ordered = sorted(stage_names_ordered, key=lambda e: (e[1], e[2]))
        stage_to_color = {"Init": palette[0]}
        for i, stage in enumerate([name for (name, _, _) in stage_names_ordered]):
            stage_to_color[stage] = palette[i+1]
        stage_to_color["End"] = palette[-1]

        # Spacing and level positions
        max_level = 3
        spacing = 0.3
        bar_height = 0.2
        level_positions = [i * spacing for i in range(max_level)]
        fontsize = 8
        tool_height_idx = 0

        # Config axis
        ax.set_ylim(-spacing, level_positions[-1] + spacing)
        ax.set_yticks(level_positions)
        other_label = "Other"
        ax.set_yticklabels([other_label, LLM, "Tool"], rotation=60, va="center")
        ax.tick_params(axis="y", pad=0)
        ax.set_ylabel("Execution Type")
        type_to_level = {
            other_label: 0,
            CHAIN: 0,
            LLM: 1,
            TOOL: 2,
        }

        # Plot stages
        custom_init_trace = {
            "name": "Init",
            "start_time": 0,
            "end_time": agent_trace[0]["start_time"],
            "kind": other_label,
        }
        custom_end_trace = {
            "name": "End",
            "start_time": agent_trace[-1]["end_time"],
            "end_time": self.glances_df["timestamp_plot"].iloc[-1],
            "kind": other_label,
        }
        grouped_agent_trace = [
            [t for t in agent_trace if t["kind"] == kind]
            for kind in [LLM, TOOL, CHAIN]
        ]
        grouped_agent_trace.append([custom_init_trace, custom_end_trace])
        for level_group in grouped_agent_trace:
            for i, t in enumerate(level_group):
                stage = t["name"]
                t_start = t["start_time"]
                t_end = t["end_time"]
                kind = t["kind"]
                extra_data = t.get("extra_data", None)
                desc = t.get("desc", None)

                duration = max(t_end - t_start, 0.2)
                y_pos = level_positions[type_to_level[kind]]
                color = stage_to_color[stage]
                vline_start = y_pos - bar_height / 2 + 0.001
                vline_end = y_pos + bar_height / 2
                
                # Draw the bar
                ax.barh(y_pos, duration, left=t_start, height=bar_height, color=color, edgecolor=None)

                # Draw border
                if i > 0 and (t_start - level_group[i-1]["end_time"]) < 0.1:
                    ax.vlines(t_start, vline_start, vline_end, color="black", linewidth=0.5)

                # Draw stage name inside bar if the bar is long enough
                bar_center = (t_start + t_end) / 2
                bar_pixel_width = ax.transData.transform((t_end, 0))[0] - ax.transData.transform((t_start, 0))[0]
                text_obj = ax.text(0, 0, stage, fontsize=fontsize)
                text_pixel_width = text_obj.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
                text_obj.remove()
                if text_pixel_width <= bar_pixel_width * 1.25:
                    ax.text(bar_center, y_pos, stage, ha='center', va='center', fontsize=fontsize)

                # Draw extra annotation text outside of bar:
                if desc is not None and len(desc) > 0:
                    if kind == LLM:
                        text = self.wrap_text_to_axis_width(ax, desc, t_start, t_start + duration, fontsize=6)
                        ax.text(bar_center, y_pos - bar_height, text, ha="center", va="top", fontsize=6)
                    elif kind == TOOL:
                        text = desc[:25] + ("..." if len(desc) > 25 else "")
                        y_pos_adjusted = y_pos + bar_height + 0.5 * bar_height * (tool_height_idx % 2) # Alternate
                        tool_height_idx += 1
                        ax.text(bar_center, y_pos_adjusted, text, ha="center", va="top", fontsize=fontsize)

        # Legend
        legend_handles = [mpatches.Patch(color=color, label=stage) for stage, color in stage_to_color.items()]
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=7, frameon=False)

        # Ax grid vertical lines
        if other_axes is not None:
            for axis in other_axes + [ax]:
                for t in [custom_init_trace] + agent_trace + [custom_end_trace]:
                    if t["kind"] == LLM and FIRST_TOKEN_TS in t["extra_data"]:
                        mid_ts = t["extra_data"][FIRST_TOKEN_TS]
                        axis.axvspan(t["start_time"], mid_ts, color=stage_to_color[t["name"]], alpha=0.2, zorder=999)
                        axis.axvspan(mid_ts, t["end_time"], color=stage_to_color[t["name"]], alpha=0.2, hatch="\\", zorder=999)
                    else:
                        axis.axvspan(t["start_time"], t["end_time"], color=stage_to_color[t["name"]], alpha=0.2, zorder=999)

        ax.set_xlim(-custom_end_trace["end_time"] * 0.01, custom_end_trace["end_time"] * 1.01)

    def plot_metrics(
        self,
        save_name,
        title,
        y_axis_label,
        metrics,
        power=False,
        second_y_axis_label=None,
        second_subplot=False,
        second_metrics=None,
        second_metrics_power=False,
    ):
        if second_metrics is not None and second_subplot:
            fig, (ax, ax_second, ax_stage) = plt.subplots(
                3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1]}
            )
        else:
            fig, (ax, ax_stage) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
            )
        linewidth = 0.75

        df = self.glances_df if not power else self.power_df

        # Plot first y
        for m in metrics:
            if len(m) == 2:
                metric, label = m
                plot_kwargs = {}
            elif len(m) == 3:
                metric, label, plot_kwargs = m
            else:
                raise Exception("Invalid metrics input")
            ax.plot("timestamp_plot", metric, data=df, label=label, linewidth=linewidth, **plot_kwargs)
        
        ax.set_ylabel(y_axis_label)
        ax.tick_params(axis="y")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        
        # Plot second y
        if second_metrics is not None:
            second_df = self.glances_df if not second_metrics_power else self.power_df
            if not second_subplot:
                ax_second = ax.twinx()
            else:
                ax_second.grid(True, alpha=0.3, axis="y")
            for m in second_metrics:
                if len(m) == 2:
                    metric, label = m
                    plot_kwargs = {}
                elif len(m) == 3:
                    metric, label, plot_kwargs = m
                ax_second.plot("timestamp_plot", metric, data=second_df, label=label, linewidth=linewidth, **plot_kwargs)

            ax_second.set_ylabel(second_y_axis_label)
            ax_second.tick_params(axis="y")
        
        # Plot legend(s)
        if second_metrics is not None:
            if second_subplot:
                if len(metrics) > 1:
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(metrics), frameon=False)
                if len(second_metrics) > 1:
                    ax_second.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(second_metrics), frameon=False)            
            else:
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=len(metrics), frameon=False, columnspacing=0.8)
                ax_second.legend(loc="upper right", bbox_to_anchor=(1.0, 0.0), ncol=len(second_metrics), frameon=False, columnspacing=0.8)
        elif len(metrics) > 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(metrics), frameon=False)

        # Plot stage timeline
        if self.full_execution:
            self.plot_trace_execution_timeline_full(ax_stage)
        else:
            other_axes=[ax] if (second_metrics is None or not second_subplot) else [ax, ax_second]
            self.plot_trace_execution_timeline_summarized(ax_stage, other_axes)
        ax_stage.set_xlabel("Time Elapsed (sec)")

        # Draw input and output boxes
        if isinstance(self.agent_task, str):
            x_start, x_end = ax.get_xlim()
            x_end = x_start + 0.25 * (x_end - x_start)
            task_text = self.wrap_text_to_axis_width(ax, f"Task: {self.agent_task}", x_start, x_end, fontsize=8)
            if self.agent_output is not None:
                output_text = f"Answer: {self.agent_output[:100]}{'...' if len(self.agent_output) > 100 else ''}"
                output_text = self.wrap_text_to_axis_width(ax, output_text, x_start, x_end, fontsize=8)
            else:
                output_text = ""
            text = f"{task_text}\n\n{output_text}".strip()
            ax.text(
                0.01, 0.99,  # x and y in axes fraction coordinates (0 to 1)
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=8,
            )

        plt.tight_layout()
        for ext in self.output_ext:
            if ext == "png":
                plt.savefig(f"{self.output_dir}/{save_name}.{ext}", dpi=300)
            else:
                plt.savefig(f"{self.output_dir}/{save_name}.{ext}")
        
        if self.display_plots:
            plt.show()
        plt.close()

    def plot_gpu_metrics(self):
        if self.device_type == DeviceType.JETSON:
            return self.plot_metrics(
                "gpu",
                title="GPU Utilization and Frequency Over Time",
                y_axis_label="GPU Utilization (%)",
                metrics=[("gpu_usage", "GPU Utilization (%)")],
                power=True,
                second_y_axis_label="GPU Frequency (MHz)",
                second_metrics=[("gpu_freq", "GPU Frequency (MHz)", {"color": "green", "linestyle": "-."})],
                second_metrics_power=True,
            )
        
        self.plot_metrics(
            "gpu",
            title="GPU Memory and Utilization Over Time",
            y_axis_label="GPU Memory (GB)",
            metrics=[("gpu_mem_plot", "GPU Memory (GB)")],
            second_y_axis_label="GPU Utilization (%)",
            second_metrics=[("gpu_usage", "GPU Utilization (%)", {"color": "green", "linestyle": "-."})],
        )

    def plot_cpu_metrics(self):
        self.plot_metrics(
            "cpu",
            title="CPU Usage Over Time",
            y_axis_label="Percentage (%)",
            metrics=[
                ("cpu_total_pct", "Total", {"linestyle": "--", "alpha": 0.7}),
                ("processes_cpu_pct", "Process")
            ],
        )

    def plot_cpu_and_gpu_metrics(self):
        if self.device_type == DeviceType.JETSON:
            return self.plot_metrics(
                "cpu_and_gpu",
                title="CPU and GPU Utilization and Frequency Over Time",
                y_axis_label="CPU Utilization (%)",
                metrics=[("processes_cpu_pct", "CPU Utilization (%)")],
                second_y_axis_label="GPU Utilization (MHz)",
                second_metrics=[("gpu_usage", "GPU Utilization (%)", {"color": "green", "linestyle": "-."})],
                second_metrics_power=True,
            )
        
        self.plot_metrics(
            "cpu_and_gpu",
            title="CPU Process and GPU Memory Usage Over Time",
            y_axis_label="CPU Process (%)",
            metrics=[("processes_cpu_pct", "Process")],
            second_y_axis_label="GPU Memory (GB)",
            second_subplot=True,
            second_metrics=[("gpu_mem_plot", "GPU Memory (GB)", {"color": "green"})],
        )

    def plot_mem_metrics(self):
        self.plot_metrics(
            "mem",
            title="Memory Usage Over Time",
            y_axis_label="GB",
            metrics=[
                ("processes_mem_plot", "Process Memory"),
                ("mem_used_plot", "System Memory Used"),
                ("memswap_used_plot", "System Memswap Used"),
            ],
        )

    def plot_diskio_metrics(self):
        self.plot_metrics(
            "diskio",
            title="Disk IO Over Time",
            y_axis_label="GB",
            metrics=[
                ("diskio_read_bytes_plot", "Read"),
                ("diskio_write_bytes_plot", "Write"),
                ("diskio_read_bytes_per_sec_plot", "Read throughput (per sec)", {"linestyle": "--", "alpha": 0.7}),
                ("diskio_write_bytes_per_sec_plot", "Write throughput (per sec)", {"linestyle": "--", "alpha": 0.7}),
            ],
        )

    def plot_concurrency_metrics(self):
        self.plot_metrics(
            "concurrency",
            title="Number of Processes and Threads Over Time",
            y_axis_label="Count",
            metrics=[
                ("processes_count", "Processes"),
                ("processes_num_threads", "Threads"),
            ],
        )

    def plot_network_metrics(self):
        self.plot_metrics(
            "network",
            title="Network Traffic Across All Interfaces Over Time",
            y_axis_label="Sent (KB)",
            metrics=[("network_sent_all", "Sent")],
            second_y_axis_label="Received (KB)",
            second_metrics=[("network_recv_all", "Received", {"color": "brown"})],
            second_subplot=True,
        )

    def get_temperature_metrics(self):
        if self.device_type == DeviceType.APPLE_LAPTOP:
            return [
                ("smctemp_cpu", "CPU Temperature"),
                ("gpu_temp", "GPU Temperature"),
                ("battery_temp", "Battery Temperature"),
            ]
        elif self.device_type == DeviceType.JETSON:
            return [
                (c, c[5:]) for c in self.power_df.columns.to_list() if (
                    c.startswith("temp_") and
                    c[5:] in ["CPU", "GPU", "Tboard", "Tdiode"]
                )
            ]

    def plot_temp_metrics(self):
        self.plot_metrics(
            "temp",
            title="Temperature Over Time",
            y_axis_label="Celcius",
            metrics=self.get_temperature_metrics(),
            power=self.device_type == DeviceType.JETSON,
        )

    def plot_temp_and_fan_metrics(self):
        self.plot_metrics(
            "temp_and_fan",
            title="Temperature and Fan Speed Over Time",
            y_axis_label="Celcius",
            metrics=self.get_temperature_metrics(),
            power=self.device_type == DeviceType.JETSON,
            second_y_axis_label="RPM",
            second_metrics=[
                ("fan_max_speed", "Fan Speed (max)", {"color": "brown", "linestyle": "--"}),
            ],
        )

    def plot_battery_metrics(self):
        if self.device_type == DeviceType.APPLE_LAPTOP:
            self.plot_metrics(
                "battery",
                title="Battery Capacity and Discharge Over Time",
                y_axis_label="Capacity (%)",
                metrics=[("battery_percent", "Capacity")],
                second_y_axis_label="Discharge (mA)",
                second_metrics=[("battery_discharge", "Discharge", {"color": "brown"})],
            )

    def get_power_metrics(self):
        if self.device_type == DeviceType.APPLE_LAPTOP:
            return [
                ("cpu_power", "CPU Power"),
                ("gpu_power", "GPU Power"),
            ]
        elif self.device_type == DeviceType.JETSON:
            return [
                (c, c[6:]) for c in self.power_df.columns.to_list() if c.startswith("power_")
            ]

    def plot_power_metrics(self):
        self.plot_metrics(
            "power",
            title="Power Usage Over Time",
            y_axis_label="mW",
            metrics=self.get_power_metrics(),
            power=True,
        )

    def plot_power_and_temp_metrics(self):
        temp_metrics = self.get_temperature_metrics()
        colors = sns.color_palette("Set1", len(temp_metrics))
        temp_metrics = [
            (*m, {"linestyle": "--", "color": colors[i]}) for i, m in enumerate(temp_metrics)
        ]
        self.plot_metrics(
            "power_and_temp",
            title="Power and Temperature Over Time",
            y_axis_label="mW",
            metrics=self.get_power_metrics(),
            power=True,
            second_y_axis_label="Temperature (Celcius)",
            second_metrics=temp_metrics,
            second_metrics_power=self.device_type==DeviceType.JETSON,
        )

    def plot_power_and_battery_metrics(self):
        if self.device_type != DeviceType.APPLE_LAPTOP:
            return
        self.plot_metrics(
            "power_and_battery",
            title="Power and Battery Discharge Over Time",
            y_axis_label="mW",
            metrics=self.get_power_metrics(),
            power=True,
            second_y_axis_label="mA",
            second_metrics=[
                ("battery_discharge", "Battery Discharge", {"linestyle": "-", "color": "brown"}),
            ],
            second_metrics_power=False,
        )

    def summarize_stats(self):
        agent_trace = self.get_topmost_spans()
        df = self.glances_df
        ts = df["timestamp_plot"]
        pwr_df = self.power_df

        num_LLM_calls = sum(1 for step in agent_trace if step["kind"] == "LLM")
        stats = {
            "agent_output": self.agent_output,
            "num_LLM_calls": num_LLM_calls,
            "duration_sec": ts.iloc[-1] - ts.iloc[0],
            "peak_process_mem_gb": df["processes_mem_plot"].max(),
            "peak_system_mem_gb": df["mem_used_plot"].max(),
        }

        if self.device_type == DeviceType.APPLE_LAPTOP:
            stats.update({
                "peak_gpu_mem_gb": df["gpu_mem_plot"].max(),
                "peak_gpu_temp_celcius": df["gpu_temp"].max(),
                "peak_cpu_temp_celcius": df["smctemp_cpu"].max(),
                "peak_battery_temp_celcius": df["battery_temp"].max(),
                "battery_charge_drop_pct": np.ptp(df["battery_percent"]),
            })

            if pwr_df is not None:
                power_stats = [
                    (pwr_df["cpu_power"], "cpu", None),
                    (pwr_df["gpu_power"], "gpu", None),
                    (pwr_df["combined_power"], "total", None),
                ]
                for k, v in energy(pwr_df["elapsed_ns"], power_stats, ts_is_elapsed=True):
                    stats[k] = v
        elif self.device_type == DeviceType.JETSON and pwr_df is not None:
            power_stats = [(pwr_df[m], l, None) for m, l in self.get_power_metrics()]
            for k, v in energy(pwr_df["timestamp"], power_stats):
                stats[k] = v
            stats.update({
                f"peak_{m}_celcius": pwr_df[m].max() for m, _ in self.get_temperature_metrics()
            })

        for k, v in stats.items():
            if type(v) is np.int64:
                stats[k] = int(v)
        
        stats_txt = json.dumps(stats, indent=4)
        with open(f"{self.output_dir}/summary.json", "w") as f:
            f.write(stats_txt)
        if self.display_summary:
            print("*** Summary ***")
            print(stats_txt)

    def summarize_stats_per_step(self):
        agent_trace = self.get_topmost_spans()
        stats = []

        for step in agent_trace:
            start_time, end_time = step["start_time"], step["end_time"]
            start_end = [(start_time, end_time, None, None)]
            extra_data = step.get("extra_data", None)
            stat = {
                "step_name": step["name"],
                "kind": step["kind"],
                "start_time": start_time,
                "end_time": end_time,
                "desc": step["desc"],
            }

            if step["kind"] == LLM and FIRST_TOKEN_TS in extra_data:
                first_token_ts = extra_data[FIRST_TOKEN_TS]
                prefill_count = extra_data[PREFILL_COUNT]
                gen_count = extra_data[GENERATION_COUNT]
                cached_count = extra_data[CACHED_COUNT]
                start_end.extend([
                    (start_time, first_token_ts, "prefill", prefill_count - cached_count),
                    (first_token_ts, end_time, "generation", gen_count),
                ])
                stat.update({
                    PREFILL_TPS: extra_data[PREFILL_TPS],
                    GENERATION_TPS: extra_data[GENERATION_TPS],
                    PREFILL_COUNT: prefill_count,
                    CACHED_COUNT: cached_count,
                    GENERATION_COUNT: gen_count,
                })

            for (start_time, end_time, label, token_count) in start_end:
                df = self.glances_df
                ts = df["timestamp_plot"]
                df = df[(ts >= start_time) & (ts < end_time)]
                prefix = (f"{label}_" if label is not None else "")

                metrics = [
                    ("duration_sec", end_time - start_time),
                    *avg_med_std([(df["processes_cpu_pct"], "process_cpu_pct")]),
                    ("avg_process_mem_gb", df["processes_mem_plot"].mean()),
                    ("avg_system_mem_gb", df["mem_used_plot"].mean()),
                ]
                
                if self.device_type == DeviceType.APPLE_LAPTOP:
                    metrics.extend([
                        ("avg_gpu_usage_pct", df["gpu_usage"].mean()),
                        ("avg_gpu_mem_gb", df["gpu_mem_plot"].mean()),
                        ("avg_temp_gpu_celcius", df["gpu_temp"].mean()),
                        ("avg_temp_cpu_celcius", df["smctemp_cpu"].mean()),
                        ("avg_temp_battery_celcius", df["battery_temp"].mean()),
                    ])

                    if self.power_df is not None:
                        power_df = self.power_df
                        ts = power_df["timestamp_plot"]
                        power_df = power_df[(ts >= start_time) & (ts < end_time)]
                        cpu_power = power_df["cpu_power"]
                        gpu_power = power_df["gpu_power"]
                        combined_power = power_df["combined_power"]

                        power_stats = [
                            (cpu_power, "power_cpu_mW"),
                            (gpu_power, "power_gpu_mW"),
                            (combined_power, "power_combined_mW")
                        ]
                        energy_stats = [
                            (cpu_power, "cpu", token_count),
                            (gpu_power, "gpu", token_count),
                            (combined_power, "total", token_count),
                        ]

                        metrics.extend([
                            *avg_med_std(power_stats),
                            *energy(power_df["elapsed_ns"], energy_stats, ts_is_elapsed=True),
                        ])
                
                elif self.device_type == DeviceType.JETSON and self.power_df is not None:
                    power_df = self.power_df
                    ts = power_df["timestamp_plot"]
                    power_df = power_df[(ts >= start_time) & (ts < end_time)]
                    energy_stats = [
                        (power_df[m], l, token_count) for m, l in self.get_power_metrics()
                    ]
                    metrics.extend(
                        [("avg_gpu_usage_pct", power_df["gpu_usage"].mean())] +
                        [(f"avg_{m}_celcius", power_df[m].mean()) for m, _ in self.get_temperature_metrics()] +
                        [e for t in [avg_med_std(power_df[m], m) for m, _ in self.get_power_metrics()] for e in t] +
                        energy(power_df["timestamp"], energy_stats)
                    )
                
                for (k, v) in metrics:
                    stat[prefix + k] = v

            stats.append(stat)

        for stat in stats:
            for k, v in stat.items():
                if type(v) is np.int64:
                    stat[k] = int(v)

        stats_txt = json.dumps(stats, indent=4)
        with open(f"{self.output_dir}/step_summary.json", "w") as f:
            f.write(stats_txt)
        if self.display_summary:
            print("*** Step-by-step summary ***")
            print(stats_txt)

    def analyze(self):
        print("Processing glances...")
        self.process_glances_log()
        print("Processing trace...")
        self.process_agent_trace()
        if self.power_log is not None:
            print("Processing power...")
            self.process_power_log()

        print("Plotting...")
        self.plot_gpu_metrics()
        self.plot_cpu_metrics()
        self.plot_cpu_and_gpu_metrics()
        self.plot_mem_metrics()
        self.plot_diskio_metrics()
        self.plot_concurrency_metrics()
        self.plot_network_metrics()
        self.plot_temp_metrics()
        self.plot_temp_and_fan_metrics()
        self.plot_battery_metrics()

        if self.power_df is not None:
            self.plot_power_metrics()
            self.plot_power_and_temp_metrics()
            self.plot_power_and_battery_metrics()

        self.summarize_stats()
        self.summarize_stats_per_step()

if __name__ == "__main__":
    example_text = """
Examples:

python -m profiler.analyze \\
--glances_log_path ./logs/smolagent_glances.jsonl \\
--agent_trace_path ./logs/smolagent_trace.json \\
--power_log_path ./logs/smolagent_powermetrics.jsonl \\
--device_type mac
--model_id qwen3:32b \\
--output_dir ./analysis_logs \\
--output_ext png pdf
"""
    parser = argparse.ArgumentParser(
        description="Analyze an execution's trace and along with its system resource usage",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device_type", type=str, required=True, choices=DeviceType.ALL, help="Type of device.")
    parser.add_argument("--glances_log_path", type=str, required=True, help="Path to the glances log.")
    parser.add_argument("--agent_trace_path", type=str, required=True, help="Path to the agent's trace.")
    parser.add_argument("--power_log_path", type=str, default=None, help="Path to the power measurement log.")
    parser.add_argument("--model_id", type=str, default=None, help="Ollama model id for analyzing.")
    parser.add_argument("--full_execution", action=argparse.BooleanOptionalAction, help="Whether to print all agent's execution steps.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to write analyses, figures, etc.")
    parser.add_argument("--output_ext", type=str, nargs="+", choices=["png", "pdf", "svg"], default=["png"], help="File type for saving (e.g., png, pdf, svg).")
    parser.add_argument("--display_plots", action=argparse.BooleanOptionalAction, help="Whether to display the plots.")
    parser.add_argument("--display_summary", action=argparse.BooleanOptionalAction, help="Whether to print the summaries.")

    args = parser.parse_args()

    analyzer = Analyzer(
        args.device_type,
        args.glances_log_path,
        args.agent_trace_path,
        power_log_path=args.power_log_path,
        model_id=args.model_id,
        full_execution=args.full_execution,
        output_dir=args.output_dir,
        output_ext=args.output_ext,
        display_plots=args.display_plots,
        display_summary=args.display_summary,
    )
    analyzer.analyze()