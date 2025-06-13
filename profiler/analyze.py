#!/usr/bin/env python3
"""
Analyze glances log and agent's trace
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from collections import deque
from pathlib import Path

SEC_TO_NANOSEC = 1_000_000_000
GB_TO_BYTE = 1024 * 1024 * 1024

class Analyzer:
    def __init__(self, glances_log_path, agent_trace, output_dir="./analysis_logs"):
        with Path(glances_log_path).open("r", encoding="utf-8") as f:
            glances_log = [json.loads(l) for l in f if l.strip()]
        with Path(agent_trace_path).open("r", encoding="utf-8") as f:
            agent_trace = json.load(f)
            
        self.glances_log = glances_log
        self.agent_trace = agent_trace

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def process_glances_log(self):
        data = [{
            "timestamp": l["timestamp"],
            "gpu_mem": l["gpu"][0]["mem_raw"],
            "gpu_mem_pct": l["gpu"][0]["mem"],
            "gpu_usage": l["gpu"][0]["proc"],
            "cpu_total_pct": l["cpu"]["total"],
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
        
        start_time = self.glances_df["timestamp"].iloc[0]
        self.processed_agent_trace = [
            (
                t["name"],
                id_to_level[t["span_id"]],
                (t["start_time"] - start_time) / SEC_TO_NANOSEC,
                (t["end_time"] - start_time) / SEC_TO_NANOSEC
            ) for t in self.agent_trace if (t["end_time"] - t["start_time"]) / SEC_TO_NANOSEC >= 0.1
        ]

    def plot_trace_execution_timeline(self, ax):
        # Color
        stage_names = list({t[0] for t in self.processed_agent_trace})
        palette = sns.color_palette("Set2", len(stage_names))
        stage_names_ordered = [
            (
                name,
                min(level for (stage, level, _, _) in self.processed_agent_trace if stage == name),
                min(start for (stage, _, start, _) in self.processed_agent_trace if stage == name)
            ) for name in stage_names
        ]
        stage_names_ordered = sorted(stage_names_ordered, key=lambda e: (e[1], e[2]))
        stage_to_color = {stage: palette[i] for i, stage in enumerate([name for (name, _, _) in stage_names_ordered])}

        # Spacing and level positions
        max_level = max(t[1] for t in self.processed_agent_trace)
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
            sorted([t for t in self.processed_agent_trace if t[1] == level], key=lambda t: t[2])
            for level in range(max_level + 1)
        ]
        for level_group in grouped_agent_trace:
            for i, (stage, level, t_start, t_end) in enumerate(level_group):
                duration = max(t_end - t_start, 0.2)
                y_pos = level_positions[level]
                color = stage_to_color[stage]
                
                # Draw the bar
                ax.barh(y_pos, duration, left=t_start, height=bar_height, color=color, edgecolor=None)

                # Draw border
                if i > 0 and (t_start - level_group[i-1][3]) < 0.1:
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
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=7, fontsize=8, frameon=False)

    def plot_metrics(self, save_name, title, y_axis_label, metrics, second_y_axis_label=None, second_metrics=None):
        fig, (ax, ax_stage) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
        )

        # Plot left y-axis
        for m in metrics:
            if len(m) == 2:
                metric, label = m
                ax.plot("timestamp_plot", metric, data=self.glances_df, label=label)
            elif len(m) == 3:
                metric, label, plot_kwargs = m
                ax.plot("timestamp_plot", metric, data=self.glances_df, label=label, **plot_kwargs)
            else:
                raise Exception("Invalid metrics input")
        
        ax.set_ylabel(y_axis_label)
        ax.tick_params(axis="y")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(metrics) > 1:
            ax.legend()

        # Plot right y-axis
        if second_y_axis_label is not None:
            ax_right = ax.twinx()
            for (metric, label) in second_metrics:
                ax_right.plot("timestamp_plot", metric, data=self.glances_df, color="green", linestyle="-.", label=label)

            ax_right.set_ylabel(second_y_axis_label, color="green")
            ax_right.tick_params(axis="y", labelcolor="green")
            if len(second_metrics) > 1:
                ax_right.legend()

        # Plot stage timeline
        self.plot_trace_execution_timeline(ax_stage)
        ax_stage.set_xlabel("Time Elapsed (sec)")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{save_name}.png", dpi=300)
        plt.show()


    def plot_gpu_metrics(self):
        self.plot_metrics(
            "gpu",
            title="GPU Memory and Utilization Over Time",
            y_axis_label="GPU Memory (GB)",
            metrics=[("gpu_mem_plot", "GPU Memory (GB)")],
            second_y_axis_label="GPU Utilization (%)",
            second_metrics=[("gpu_usage", "GPU Utilization (%)")]
        )

    def plot_cpu_metrics(self):
        self.plot_metrics(
            "cpu",
            title="CPU Usage Over Time",
            y_axis_label="Percentage (%)",
            metrics=[("processes_cpu_pct", "Process"), ("cpu_total_pct", "Total")]
        )

    def plot_mem_metrics(self):
        self.plot_metrics(
            "mem",
            title="Memory Usage Over Time",
            y_axis_label="GB",
            metrics=[("processes_mem_plot", "Process Memory"), ("memswap_used_plot", "System Memswap")]
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
            ]
        )

    def plot_concurrency_metrics(self):
        self.plot_metrics(
            "concurrency",
            title="Number of Processes and Threads Over Time",
            y_axis_label="Count",
            metrics=[
                ("processes_count", "Processes"),
                ("processes_num_threads", "Threads"),
            ]
        )


    def analyze(self):
        self.process_glances_log()
        self.process_agent_trace()

        self.plot_gpu_metrics()
        self.plot_cpu_metrics()
        self.plot_mem_metrics()
        self.plot_diskio_metrics()
        self.plot_concurrency_metrics()
    

if __name__ == "__main__":
    glances_log_path = "./logs/basicmlx_glances.jsonl"
    agent_trace_path = "./logs/basicmlx_trace.json" # Currently only smolagent's trace is accepted
    output_dir = "./analysis_logs"

    analyzer = Analyzer(glances_log_path, agent_trace_path, output_dir)
    analyzer.analyze()