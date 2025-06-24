"""
Analyze glances log and agent's trace
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import ollama
import pandas as pd
import seaborn as sns
import sys
from tqdm import tqdm
from collections import deque
from pathlib import Path

SEC_TO_NANOSEC = 1_000_000_000
GB_TO_BYTE = 1024 * 1024 * 1024

TRACE_KIND_FIELD = "openinference.span.kind"
LLM = "LLM"
POWER_LOG_TYPE_MAC = "mac"
FIRST_TOKEN_TS = "first_token_ts"

class Analyzer:
    def __init__(
        self,
        glances_log_path,
        agent_trace_path,
        power_log_path=None,
        power_log_type=None,
        model_id=None, # LLM model for analyzing
        full_execution=False,
        output_dir="./analysis_logs", output_ext=["png"],
    ):
        with Path(glances_log_path).open("r", encoding="utf-8") as f:
            self.glances_log = [json.loads(l) for l in f if l.strip()]
        with Path(agent_trace_path).open("r", encoding="utf-8") as f:
            self.agent_trace = json.load(f)
        
        self.power_df = None
        if power_log_path is not None:
            self.power_log_type = power_log_type
            if power_log_type == POWER_LOG_TYPE_MAC:
                with Path(power_log_path).open("r", encoding="utf-8") as f:
                    self.power_log = [json.loads(l) for l in f if l.strip()]
            else:
                raise NotImplemented("Only power log from macOS's powermetrics is supported for now")
        
        if model_id is not None: # Make sure model_id is correct
            ollama.show(model=model_id)
        self.model_id = model_id

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.full_execution = full_execution

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
            "battery_temp": next(((s["value"] / 10 - 273.15) for s in l["sensors"] if s["label"] == "Battery Virtual Temperature"), 0),
            "fan_max_speed": max(s["value"] for s in l["sensors"] if s["label"].startswith("Fan")),
            "processes_count": len(l["processlist"]),
            "processes_cpu_pct": sum(p["cpu_percent"] for p in l["processlist"]),
            "processes_num_threads": sum(p["num_threads"] for p in l["processlist"]),
            "processes_mem": sum(p["memory_info"]["rss"] for p in l["processlist"]),
            "memswap_used": l["memswap"]["used"],
            "diskio_read_bytes": sum(d["read_bytes"] for d in l["diskio"]),
            "diskio_write_bytes": sum(d["write_bytes"] for d in l["diskio"]),
            "diskio_read_bytes_per_sec": sum(d.get("read_bytes_rate_per_sec", 0) for d in l["diskio"]),
            "diskio_write_bytes_per_sec": sum(d.get("write_bytes_rate_per_sec", 0) for d in l["diskio"]),
        } for l in self.glances_log]
        df = pd.DataFrame.from_dict(data)

        start_time = df["timestamp"].iloc[0]
        df["timestamp_plot"] = (df["timestamp"] - start_time) / SEC_TO_NANOSEC
        df["gpu_mem_plot"] = df["gpu_mem"] / GB_TO_BYTE
        df["processes_mem_plot"] = df["processes_mem"] / GB_TO_BYTE
        df["memswap_used_plot"] = df["memswap_used"] / GB_TO_BYTE
        
        df["diskio_read_bytes_plot"] = df["diskio_read_bytes"] / GB_TO_BYTE
        df["diskio_write_bytes_plot"] = df["diskio_write_bytes"] / GB_TO_BYTE
        df["diskio_read_bytes_per_sec_plot"] = df["diskio_read_bytes_per_sec"] / GB_TO_BYTE
        df["diskio_write_bytes_per_sec_plot"] = df["diskio_write_bytes_per_sec"] / GB_TO_BYTE

        self.glances_df = df

    def process_power_log(self):
        if self.power_log_type != POWER_LOG_TYPE_MAC:
            raise NotImplemented("Only power log from macOS's powermetrics is supported for now")

        data = [{
            "timestamp": l["timestamp"],
            "elapsed_ns": l["elapsed_ns"],
            "cpu_energy": l["processor"]["cpu_energy"],
            "cpu_power": l["processor"]["cpu_power"],
            "gpu_energy": l["processor"]["gpu_energy"],
            "gpu_power": l["processor"]["gpu_power"],
            "combined_power": l["processor"]["combined_power"],
            "all_processes_energy_impact": l["all_tasks"]["energy_impact"],
            "processes_energy_impact": sum(p["energy_impact"] for p in l["tasks"]),
            "processes_diskio_read_bytes": sum(p["diskio_bytesread"] for p in l["tasks"]),
            "processes_diskio_write_bytes": sum(p["diskio_byteswritten"] for p in l["tasks"]),
            "processes_diskio_read_bytes_per_sec": sum(p["diskio_bytesread_per_s"] for p in l["tasks"]),
            "processes_diskio_write_bytes_per_sec": sum(p["diskio_byteswritten_per_s"] for p in l["tasks"]),
            "processes_network_sent_bytes": sum(p["bytes_sent"] for p in l["tasks"]),
            "processes_network_received_bytes": sum(p["bytes_received"] for p in l["tasks"]),
            "processes_network_sent_bytes_per_sec": sum(p["bytes_sent_per_s"] for p in l["tasks"]),
            "processes_network_received_bytes_per_sec": sum(p["bytes_received_per_s"] for p in l["tasks"]),
            "processes_network_sent_packets": sum(p["packets_sent"] for p in l["tasks"]),
            "processes_network_received_packets": sum(p["packets_received"] for p in l["tasks"]),
            "processes_network_sent_packets_per_sec": sum(p["packets_sent_per_s"] for p in l["tasks"]),
            "processes_network_received_packets_per_sec": sum(p["packets_received_per_s"] for p in l["tasks"]),
        } for l in self.power_log]
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
        res = ollama.generate(model=self.model_id, prompt=prompt, options={"temperature": 0.0})
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
        
        start_time = self.glances_df["timestamp"].iloc[0]

        processed = []
        for t in tqdm(self.agent_trace):
            if (t["end_time"] - t["start_time"]) / SEC_TO_NANOSEC < 0.1:
                continue

            attributes = t["attributes"]
            kind = attributes[TRACE_KIND_FIELD]
            desc = None
            extra_data = {}

            if kind == LLM:
                # Skip processing child LLM span
                parent_id = t["parent_span_id"]
                if parent_id is not None:
                    parent_attr = id_to_trace[parent_id]["attributes"]
                    if parent_attr[TRACE_KIND_FIELD] == LLM:
                        continue

                # Copy attributes from child LLM span (if any)
                for child_id in id_to_child_id[t["span_id"]]:
                    child_attr = id_to_trace[child_id]["attributes"]
                    if child_attr[TRACE_KIND_FIELD] == LLM:
                        for k, v in child_attr.items():
                            if k not in attributes:
                                attributes[k] = v

                # Create description text for span
                agent_desc = ""
                token_desc = ""
                output_msg = attributes.get("output.value", None)
                input_tokens = attributes.get("llm.token_count.prompt", None)
                output_tokens = attributes.get("llm.token_count.completion", None)
                prefill_tps = attributes.get("prefill_tps", None)
                generation_tps = attributes.get("generation_tps", None)
                
                if output_msg is not None:
                    agent_desc = self.describe_agent_action(output_msg)
                if input_tokens is not None and output_tokens is not None:
                    token_desc = f"Tkn: {input_tokens} in, {output_tokens} out"
                if prefill_tps is not None and generation_tps is not None:
                    prefill_tps = round(prefill_tps, 1)
                    generation_tps = round(generation_tps, 1)
                    desc_str = f"{prefill_tps}/s pf, {generation_tps}/s gn"
                    if len(token_desc) > 0:
                        token_desc += f", {desc_str}"
                    else:
                        token_desc = f"Tkn: {desc_str}"
                desc = f"{agent_desc}\n({token_desc})"

                # Add extra data
                if FIRST_TOKEN_TS in attributes:
                    extra_data[FIRST_TOKEN_TS] = (attributes[FIRST_TOKEN_TS] - start_time) / SEC_TO_NANOSEC
            
            elif kind == "TOOL":
                if "search" in t["name"].lower():
                    input_value = attributes.get("input.value", None)
                    if input_value is not None:
                        try:
                            input_value = json.loads(input_value)
                            desc = input_value.get("query", None)
                            if desc is None and "kwargs" in input_value:
                                desc = input_value["kwargs"].get("query", None)
                        except:
                            pass
            
            span = {
                "name": t["name"],
                "level": id_to_level[t["span_id"]],
                "start_time": (t["start_time"] - start_time) / SEC_TO_NANOSEC,
                "end_time": (t["end_time"] - start_time) / SEC_TO_NANOSEC,
                "kind": kind,
                "desc": desc.strip() if desc is not None else "",
                "extra_data": extra_data,
            }
            processed.append(span)

        self.processed_agent_trace = processed


    def get_sorted_topmost_spans(self):
        spans = sorted(self.processed_agent_trace, key=lambda t: t["start_time"])
        top_spans = []
        i = 0
        while i < len(spans):
            top = spans[i]
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
            sorted([t for t in agent_trace if t["level"] == level], key=lambda t: t["start_time"])
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
        agent_trace = self.get_sorted_topmost_spans()
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

        # Config axis
        ax.set_ylim(-spacing, level_positions[-1] + spacing)
        ax.set_yticks(level_positions)
        other_label = "Other"
        ax.set_yticklabels([other_label, LLM, "Tool"], rotation=60, va="center")
        ax.tick_params(axis="y", pad=0)
        ax.set_ylabel("Execution Type")
        type_to_level = {
            other_label: 0,
            LLM: 1,
            "TOOL": 2,
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
            sorted([t for t in agent_trace if t["kind"] == kind], key=lambda t: t["start_time"])
            for kind in [LLM, "TOOL"]
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
                    elif kind == "TOOL":
                        ax.text(bar_center, y_pos + bar_height, desc[:50], ha="center", va="top", fontsize=fontsize)

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
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=len(metrics), frameon=False)
                ax_second.legend(loc="upper right", bbox_to_anchor=(1.0, 0.0), ncol=len(second_metrics), frameon=False)
        elif len(metrics) > 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(metrics), frameon=False)

        # Plot stage timeline
        if self.full_execution:
            self.plot_trace_execution_timeline_full(ax_stage)
        else:
            other_axes=[ax] if (second_metrics is None or not second_subplot) else [ax, ax_second]
            self.plot_trace_execution_timeline_summarized(ax_stage, other_axes)
        ax_stage.set_xlabel("Time Elapsed (sec)")

        plt.tight_layout()
        for ext in self.output_ext:
            if ext == "png":
                plt.savefig(f"{self.output_dir}/{save_name}.{ext}", dpi=300)
            else:
                plt.savefig(f"{self.output_dir}/{save_name}.{ext}")
        plt.show()

    def plot_gpu_metrics(self):
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
            metrics=[("processes_mem_plot", "Process Memory"), ("memswap_used_plot", "System Memswap")],
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

    def plot_temp_metrics(self):
        self.plot_metrics(
            "temp",
            title="CPU, GPU, and Battery Temperature Over Time",
            y_axis_label="Celcius",
            metrics=[
                ("smctemp_cpu", "CPU Temperature"),
                ("gpu_temp", "GPU Temperature"),
                ("battery_temp", "Battery Temperature"),
            ],
        )

    def plot_temp_and_fan_metrics(self):
        self.plot_metrics(
            "temp",
            title="CPU/GPU/Battery Temperature and Fan Speed Over Time",
            y_axis_label="Celcius",
            metrics=[
                ("smctemp_cpu", "CPU Temperature"),
                ("gpu_temp", "GPU Temperature"),
                ("battery_temp", "Battery Temperature"),
            ],
            second_y_axis_label="RPM",
            second_metrics=[
                ("fan_max_speed", "Fan Speed (max)", {"color": "brown"}),
            ],
        )

    def plot_battery_metrics(self):
        self.plot_metrics(
            "battery",
            title="Battery Capacity and Discharge Over Time",
            y_axis_label="Capacity (%)",
            metrics=[("battery_percent", "Capacity")],
            second_y_axis_label="Discharge (mA)",
            second_metrics=[("battery_discharge", "Discharge", {"color": "brown"})],
        )

    def plot_power_metrics(self):
        self.plot_metrics(
            "power",
            title="CPU/GPU Power and Process Energy Impact Over Time",
            y_axis_label="mW",
            metrics=[
                ("cpu_power", "CPU Power"),
                ("gpu_power", "GPU Power"),
                ("combined_power", "Combined Power"),
            ],
            power=True,
            second_y_axis_label="Energy Impact Score",
            second_metrics=[
                ("all_processes_energy_impact", "Energy Impact", {"color": "grey"}),
            ],
            second_metrics_power=True,
        )

    def plot_power_and_temp_metrics(self):
        self.plot_metrics(
            "power_and_temp",
            title="CPU/GPU Power and Temperature Over Time",
            y_axis_label="mW",
            metrics=[
                ("cpu_power", "CPU Power"),
                ("gpu_power", "GPU Power"),
            ],
            power=True,
            second_y_axis_label="Temperature (Celcius)",
            second_metrics=[
                ("smctemp_cpu", "CPU Temperature", {"linestyle": "-", "color": "green"}),
                ("gpu_temp", "GPU Temperature", {"linestyle": "-", "color": "brown"}),
            ],
            second_metrics_power=False,
        )

    def plot_power_and_battery_metrics(self):
        self.plot_metrics(
            "power_and_battery",
            title="CPU/GPU Power and Battery Discharge Over Time",
            y_axis_label="mW",
            metrics=[
                ("cpu_power", "CPU Power"),
                ("gpu_power", "GPU Power"),
            ],
            power=True,
            second_y_axis_label="mA",
            second_metrics=[
                ("battery_discharge", "Battery Discharge", {"linestyle": "-", "color": "brown"}),
            ],
            second_metrics_power=False,
        )

    def analyze(self):
        print("Processing glances...")
        self.process_glances_log()
        print("Processing trace...")
        self.process_agent_trace()
        if self.power_log is not None:
            print("Processing power...")
            self.process_power_log()

        self.plot_gpu_metrics()
        self.plot_cpu_metrics()
        self.plot_cpu_and_gpu_metrics()
        self.plot_mem_metrics()
        self.plot_diskio_metrics()
        self.plot_concurrency_metrics()
        self.plot_temp_metrics()
        self.plot_temp_and_fan_metrics()
        self.plot_battery_metrics()

        if self.power_df is not None:
            self.plot_power_metrics()
            self.plot_power_and_temp_metrics()
            self.plot_power_and_battery_metrics()

        self.summarize_stats()
    
    def summarize_stats(self):
        print("*** Summary ***")
        print(f"Peak GPU memory: {self.glances_df['gpu_mem_plot'].max()} GB")
        print(f"Peak RAM: {self.glances_df['processes_mem_plot'].max()} GB")
        print(f"Peak GPU temp: {self.glances_df['gpu_temp'].max()} C")
        print(f"Peak CPU temp: {self.glances_df['smctemp_cpu'].max()} C")
        print(f"Battery temp: {self.glances_df['battery_temp'].max()} C")
        print(f"Battery charge drop: {self.glances_df['battery_percent'].max() - self.glances_df['battery_percent'].min()} %")

        if self.power_df is not None:
            time_diff = self.power_df["elapsed_ns"] / SEC_TO_NANOSEC / 3600
            total_energy = self.power_df["combined_power"] * time_diff
            print(f"Total energy spent: {total_energy.sum()} mWh")

if __name__ == "__main__":
    example_text = """
Examples:

python analyze.py \\
--glances_log_path ./logs/smolagent_glances.jsonl \\
--agent_trace_path ./logs/smolagent_trace.json \\
--power_log_path ./logs/smolagent_powermetrics.jsonl \\
--power_log_type mac
--model_id qwen2.5-coder:32b \\
--output_dir ./analysis_logs \\
--output_ext png pdf
"""
    parser = argparse.ArgumentParser(
        description="Analyze an execution's trace and along with its system resource usage",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--glances_log_path", type=str, required=True, help="Path to the glances log.")
    parser.add_argument("--agent_trace_path", type=str, required=True, help="Path to the agent's trace.")
    parser.add_argument("--power_log_path", type=str, default=None, help="Path to the power measurement log.")
    parser.add_argument(
        "--power_log_type", type=str,
        required="--power_log_path" in sys.argv,
        choices=[POWER_LOG_TYPE_MAC],
        help="Type of power measurement."
    )
    parser.add_argument("--model_id", type=str, default=None, help="Ollama model id for analyzing.")
    parser.add_argument("--full_execution", action=argparse.BooleanOptionalAction, help="Whether to print all agent's execution steps.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to write analyses, figures, etc.")
    parser.add_argument("--output_ext", type=str, nargs="+", choices=["png", "pdf", "svg"], default=["png"], help="File type for saving (e.g., png, pdf, svg).")

    args = parser.parse_args()

    analyzer = Analyzer(
        args.glances_log_path,
        args.agent_trace_path,
        power_log_path=args.power_log_path,
        power_log_type=args.power_log_type,
        model_id=args.model_id,
        full_execution=args.full_execution,
        output_dir=args.output_dir,
        output_ext=args.output_ext,
    )
    analyzer.analyze()