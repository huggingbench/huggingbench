import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from bench.config import get_os_friendly_path

LOG = logging.getLogger(__name__)


class ChartGen:
    # Define colors based on format
    colors = {"onnx": "blue", "trt": "green", "openvino": "red"}
    # Labels for each bar
    labels = ["format", "device", "half", "batch_size", "client_workers", "instance_count", "async_client", "success_rate"]

    def __init__(self, workspace_dir: str):
        self.output_dir = workspace_dir

    def plot_chart(self, hf_id:str, labels: pd.DataFrame , chart_data: pd.DataFrame, chart_name: str):
        # Set the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        x_ticks = range(len(chart_data))
        ax.bar(x_ticks, chart_data, color=[self.colors.get(format_val, "gray") for format_val in labels["format"]])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([])
        ax.set_xlabel("Configuration")
        ax.set_ylabel(chart_name)
        ax.set_title(f"Comparison of {chart_name}", loc="left")

        for i, label in enumerate(labels.values):
            success_rate = "{:.2f}".format(label[7])
            ax.text(
                i,
                chart_data[i],
                f"format={label[0]}\ndevice={label[1]}\nhalf={label[2]}\nbatch_size={label[3]}\nclient_workers={label[4]}\ninstance_count={label[5]}\nasync={label[6]}\nsucc-rate={success_rate}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Adjust spacing between subplots and ensure labels fit
        plt.tight_layout()
        hf_id_str = get_os_friendly_path(hf_id)

        chart_abs_path = os.path.abspath(f"{self.output_dir}/{hf_id_str}-{chart_name}.png")
        fig.savefig(chart_abs_path)
        LOG.info(f"Saved chart to '{chart_abs_path}'")
        plt.close(fig)

    def plot_charts(self, hf_id: str, df: pd.DataFrame):
        # Extract required columns
        labels = df[ChartGen.labels]
        median_latencies = df["median"]
        percentile90_latencies = df["90_percentile"]
        percentile99_latencies = df["99_percentile"]
        throughputs = df["success_rate"]
        avg_latencies = df["avg"]

        charts = {
            "median_latencies": median_latencies,
            "90_percentile_latencies": percentile90_latencies,
            "99_percentile_latencies": percentile99_latencies,
            "throughputs": throughputs,
            "avg_latencies": avg_latencies,
        }

        for chart_name, chart_data in charts.items():
            self.plot_chart(hf_id, labels, chart_data, chart_name)


if __name__ == "__main__":
    ChartGen("temp/").plot_charts("prajjwal1/bert-tiny")
