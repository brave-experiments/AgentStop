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

def wrap_text_to_axis_width(ax, text, x_start, x_end, fontsize=None):
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

class Analyzer:
    def __init__(
        self, glances_log_path, agent_trace_path,
        output_dir="./analysis_logs", output_ext="png", full_execution=False,
    ):
        with Path(glances_log_path).open("r", encoding="utf-8") as f:
            glances_log = [json.loads(l) for l in f if l.strip()]
        with Path(agent_trace_path).open("r", encoding="utf-8") as f:
            agent_trace = json.load(f)
            
        self.glances_log = glances_log
        self.agent_trace = agent_trace

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
            {
                "name": t["name"],
                "level": id_to_level[t["span_id"]],
                "start_time": (t["start_time"] - start_time) / SEC_TO_NANOSEC,
                "end_time": (t["end_time"] - start_time) / SEC_TO_NANOSEC,
                "attributes": t["attributes"],
            } for t in self.agent_trace if (t["end_time"] - t["start_time"]) / SEC_TO_NANOSEC >= 0.1
        ]

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
        other_label = "Oth"
        ax.set_yticklabels([other_label, "LLM", "Tool"])
        ax.set_ylabel("Execution Type")
        type_to_level = {
            other_label: 0,
            "LLM": 1,
            "TOOL": 2,
        }

        # Plot stages
        custom_init_trace = {
            "name": "Init",
            "start_time": 0,
            "end_time": agent_trace[0]["start_time"],
            "attributes": {"openinference.span.kind": other_label}
        }
        custom_end_trace = {
            "name": "End",
            "start_time": agent_trace[-1]["end_time"],
            "end_time": self.glances_df["timestamp_plot"].iloc[-1],
            "attributes": {"openinference.span.kind": other_label}
        }
        grouped_agent_trace = [
            sorted([t for t in agent_trace if t["attributes"]["openinference.span.kind"] == kind], key=lambda t: t["start_time"])
            for kind in ["LLM", "TOOL"]
        ]
        grouped_agent_trace.append([custom_init_trace, custom_end_trace])
        for level_group in grouped_agent_trace:
            for i, t in enumerate(level_group):
                stage = t["name"]
                t_start = t["start_time"]
                t_end = t["end_time"]
                attributes = t["attributes"]
                kind = attributes["openinference.span.kind"]
                duration = max(t_end - t_start, 0.2)
                y_pos = level_positions[type_to_level[kind]]
                color = stage_to_color[stage]
                
                # Draw the bar
                ax.barh(y_pos, duration, left=t_start, height=bar_height, color=color, edgecolor=None)

                # Draw border
                if i > 0 and (t_start - level_group[i-1]["end_time"]) < 0.1:
                    ax.vlines(t_start, y_pos - bar_height / 2 + 0.001, y_pos + bar_height / 2, color='black', linewidth=0.5)

                # Draw stage name inside bar if the bar is long enough
                bar_center = (t_start + t_end) / 2
                bar_pixel_width = ax.transData.transform((t_end, 0))[0] - ax.transData.transform((t_start, 0))[0]
                text_obj = ax.text(0, 0, stage, fontsize=fontsize)
                text_pixel_width = text_obj.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
                text_obj.remove()
                if text_pixel_width <= bar_pixel_width * 1.25:
                    ax.text(bar_center, y_pos, stage, ha='center', va='center', fontsize=fontsize)

                # Draw extra annotation text outside of bar:
                if kind == "LLM":
                    input_tokens = attributes.get("llm.token_count.prompt", None)
                    output_tokens = attributes.get("llm.token_count.completion", None)
                    if input_tokens is not None and output_tokens is not None:
                        text = f"Tkn: {input_tokens} in, {output_tokens} out"
                        wrapped_text = wrap_text_to_axis_width(ax, text, t_start, t_start + duration, fontsize=fontsize)
                        ax.text(bar_center, y_pos - bar_height, wrapped_text, ha="center", va="top", fontsize=fontsize)

        # Legend
        legend_handles = [mpatches.Patch(color=color, label=stage) for stage, color in stage_to_color.items()]
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=7, frameon=False)

        # Ax grid vertical lines
        if other_axes is not None:
            for axis in other_axes + [ax]:
                for t in [custom_init_trace] + agent_trace + [custom_end_trace]:
                    axis.axvspan(t["start_time"], t["end_time"], color=stage_to_color[t["name"]], alpha=0.2, zorder=999)

        ax.set_xlim(-custom_end_trace["end_time"] * 0.01, custom_end_trace["end_time"] * 1.01)

    def plot_metrics(
        self, save_name, title, y_axis_label, metrics,
        second_y_axis_label=None, second_y_axis_color="black", second_subplot=False, second_metrics=None,
    ):
        if second_subplot:
            fig, (ax, ax_second, ax_stage) = plt.subplots(
                3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1.5, 1]}
            )
        else:
            fig, (ax, ax_stage) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
            )
        linewidth = 0.75

        # Plot first y
        for m in metrics:
            if len(m) == 2:
                metric, label = m
                plot_kwargs = {}
            elif len(m) == 3:
                metric, label, plot_kwargs = m
            else:
                raise Exception("Invalid metrics input")
            ax.plot("timestamp_plot", metric, data=self.glances_df, label=label, linewidth=linewidth, **plot_kwargs)
        
        ax.set_ylabel(y_axis_label)
        ax.tick_params(axis="y")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        if len(metrics) > 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(metrics), frameon=False)
        

        # Plot second y
        if second_y_axis_label is not None:
            if not second_subplot:
                ax_second = ax.twinx()
            for m in second_metrics:
                if len(m) == 2:
                    metric, label = m
                    plot_kwargs = {}
                elif len(m) == 3:
                    metric, label, plot_kwargs = m
                ax_second.plot("timestamp_plot", metric, data=self.glances_df, label=label, linewidth=linewidth, **plot_kwargs)

            ax_second.set_ylabel(second_y_axis_label, color=second_y_axis_color)
            ax_second.tick_params(axis="y", labelcolor=second_y_axis_color)
            
            if second_subplot:
                ax_second.grid(True, alpha=0.3, axis="y")
            if len(second_metrics) > 1:
                ax_second.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(second_metrics), frameon=False)

        # Plot stage timeline
        if self.full_execution:
            self.plot_trace_execution_timeline_full(ax_stage)
        else:
            other_axes=[ax] if (second_y_axis_label is None or not second_subplot) else [ax, ax_second]
            self.plot_trace_execution_timeline_summarized(ax_stage, other_axes)
        ax_stage.set_xlabel("Time Elapsed (sec)")

        plt.tight_layout()
        if self.output_ext == "png":
            plt.savefig(f"{self.output_dir}/{save_name}.{self.output_ext}", dpi=300)
        else:
            plt.savefig(f"{self.output_dir}/{save_name}.{self.output_ext}")
        plt.show()

    def plot_gpu_metrics(self):
        self.plot_metrics(
            "gpu",
            title="GPU Memory and Utilization Over Time",
            y_axis_label="GPU Memory (GB)",
            metrics=[("gpu_mem_plot", "GPU Memory (GB)")],
            second_y_axis_label="GPU Utilization (%)",
            second_y_axis_color="green",
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


    def analyze(self):
        self.process_glances_log()
        self.process_agent_trace()

        self.plot_gpu_metrics()
        self.plot_cpu_metrics()
        self.plot_cpu_and_gpu_metrics()
        self.plot_mem_metrics()
        self.plot_diskio_metrics()
        self.plot_concurrency_metrics()
    

if __name__ == "__main__":
    example_text = """
Examples:

python analyze.py \\
--glances_log_path ./logs/smolagent_glances.jsonl \\
--agent_trace_path ./logs/smolagent_trace.json \\
--output_dir ./analysis_logs \\
--output_ext png
"""
    parser = argparse.ArgumentParser(
        description="Analyze an execution's trace and along with its system resource usage",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--glances_log_path", type=str, required=True, help="Path to the glances log.")
    parser.add_argument("--agent_trace_path", type=str, required=True, help="Path to the agent's trace.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to write analyses, figures, etc.")
    parser.add_argument("--output_ext", type=str, choices=["png", "pdf", "svg"], default="png", help="File type for saving (e.g., png, pdf, svg).")
    parser.add_argument("--full_execution", action=argparse.BooleanOptionalAction, help="Whether to print all agent's execution steps.")

    args = parser.parse_args()

    analyzer = Analyzer(args.glances_log_path, args.agent_trace_path, args.output_dir, args.output_ext, args.full_execution)
    analyzer.analyze()