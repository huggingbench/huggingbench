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
    labels = [
        "format",
        "device",
        "precision",
        "batch_size",
        "clients",
        "instances",
        "sequence_length",
        "throughput",
        "dataset",
        "id",
    ]

    def __init__(self):
        self.data = None

    def add_data(self, df: pd.DataFrame):
        if self.data is None:
            self.data = df
        else:
            self.data = pd.concat([self.data, df], ignore_index=True)

    def plot_chart(
        self, labels: pd.DataFrame, chart_data: pd.DataFrame, chart_name: str, output_dir: str, model_id: str
    ):
        # Set the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        x_ticks = range(len(chart_data))
        ax.bar(x_ticks, chart_data, color=[self.colors.get(format_val, "gray") for format_val in labels["format"]])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([])
        ax.set_xlabel("Configuration")
        ax.set_ylabel(chart_name)
        ax.set_title(f"Comparison of {chart_name}", loc="left")

        for i, label_val in enumerate(labels.values):
            success_rate = "{:.2f}".format(label_val[7])
            ax.text(
                i,
                chart_data[i],
                f"format={label_val[0]}\ndevice={label_val[1]}\nprecision={label_val[2]}\nbatch_size={label_val[3]}\nclients={label_val[4]}\ninstances={label_val[5]}\nsequence_length={label_val[6]} \nthroughput={success_rate}\ndataset={label_val[8]}\nid={label_val[9]}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Adjust spacing between subplots and ensure labels fit
        plt.tight_layout()
        file_name = f"{get_os_friendly_path(model_id)}_{chart_name}.png"
        chart_abs_path = os.path.abspath(f"{output_dir}/{file_name}")
        fig.savefig(chart_abs_path)
        LOG.info(f"Saved chart to '{chart_abs_path}'")
        plt.close(fig)

    def plot_charts(self, output_dir: str, model_id: str, df: pd.DataFrame = None):
        if df is None and self.data is None:
            return ValueError("No data to plot")
        # Extract required columns
        df = self.data if df is None else df
        labels = df[ChartGen.labels]
        median_latencies = df["median"]
        percentile90_latencies = df["90_percentile"]
        throughputs = df["throughput"]
        avg_latencies = df["avg"]

        charts = {
            "median_latencies": median_latencies,
            "90_percentile_latencies": percentile90_latencies,
            "throughputs": throughputs,
            "avg_latencies": avg_latencies,
        }

        for chart_name, chart_data in charts.items():
            self.plot_chart(labels, chart_data, chart_name, output_dir, model_id)


if __name__ == "__main__":
    ChartGen().plot_charts(hf_id="prajjwal1/bert-tiny", df=None, output_dir="./temp")
