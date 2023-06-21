import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from bench.config import get_os_friendly_path

LOG = logging.getLogger(__name__)


class ChartGen:
    # Define colors based on format
    colors = {"onnx": "blue", "trt": "green", "openvino": "red"}

    def __init__(self, workspace_dir: str):
        self.output_dir = workspace_dir

    def plot_chart(self, hf_id, labels, chart_data, chart_name):
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
            success_rate = "{:.2f}".format(label[6])
            ax.text(
                i,
                chart_data[i],
                f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch={label[3]}\ncli={label[5]}\nasync={label[6]}\nsucc-rate={success_rate}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Adjust spacing between subplots and ensure labels fit
        plt.tight_layout()
        hf_id_str = get_os_friendly_path(hf_id)

        chart_abs_path = os.path.abspath(f"{self.output_dir}/{hf_id_str}-{chart_name}.png")
        fig.savefig(chart_abs_path)
        LOG.info(f"Saved chart '{chart_abs_path}'")
        plt.close(fig)

    def plot_charts(self, hf_id: str):
        os_hf_id = get_os_friendly_path(hf_id)
        input_csv = os.path.join(self.output_dir, f"{os_hf_id}.csv")
        df = pd.read_csv(input_csv)

        # Extract required columns
        labels = df[["format", "device", "half", "batch_size", "client_workers", "async_client", "success_rate"]]
        median_latencies = df["median"]
        percentile90_latencies = df["90_percentile"]
        percentile99_latencies = df["99_percentile"]
        throughputs = df["success_rate"]

        charts = {
            "median_latencies": median_latencies,
            "90_percentile_latencies": percentile90_latencies,
            "99_percentile_latencies": percentile99_latencies,
            "throughputs": throughputs,
        }

        for chart_name, chart_data in charts.items():
            self.plot_chart(hf_id, labels, chart_data, chart_name)


if __name__ == "__main__":
    ChartGen("temp/").plot_charts("prajjwal1/bert-tiny")
