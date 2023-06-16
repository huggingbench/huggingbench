import pandas as pd
import matplotlib.pyplot as plt


def create_charts(stats_path="temp/prajjwal1-bert-tiny.csv"):
    df = pd.read_csv(stats_path)

    # Extract required columns
    labels = df[
        ["format", "device", "half", "batch_size", "sequence_length", "client_workers", "async_client", "success_rate"]
    ]
    median_latencies = df["median"]
    percentile90_latencies = df["90_percentile"]
    percentile99_latencies = df["99_percentile"]
    throughputs = df["success_rate"]

    # Set the figure size
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    # Define colors based on format
    colors = {"onnx": "blue", "trt": "green", "openvino": "red"}

    # Create the chart for median latencies
    x_ticks = range(len(median_latencies))
    ax1.bar(x_ticks, median_latencies, color=[colors.get(format_val, "gray") for format_val in labels["format"]])
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([])  # Remove x-axis labels

    # Add x-axis labels inside the bars
    for i, label in enumerate(labels.values):
        success_rate = "{:.2f}".format(label[7])
        ax1.text(
            i,
            median_latencies[i],
            f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch={label[3]}\nseq={label[4]}\ncli={label[5]}\nasync={label[6]}\nsucc-rate={success_rate}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Median Latency")
    ax1.set_title("Comparison of Median Latencies", loc="left")

    # Create the chart for 90th percentiles
    ax2.bar(x_ticks, percentile90_latencies, color=[colors.get(format_val, "gray") for format_val in labels["format"]])
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([])  # Remove x-axis labels

    # Add x-axis labels inside the bars
    for i, label in enumerate(labels.values):
        success_rate = "{:.2f}".format(label[7])
        ax2.text(
            i,
            percentile90_latencies[i],
            f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch={label[3]}\nseq={label[4]}\ncli={label[5]}\nasync={label[6]}\nsucc-rate={success_rate}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("90th Percentile Latency")
    ax2.set_title("Comparison of 90th Percentile Latencies", loc="left")

    # Create the chart for 99th percentiles
    ax3.bar(x_ticks, percentile99_latencies, color=[colors.get(format_val, "gray") for format_val in labels["format"]])
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([])  # Remove x-axis labels

    # Add x-axis labels inside the bars
    for i, label in enumerate(labels.values):
        success_rate = "{:.2f}".format(label[7])
        ax3.text(
            i,
            percentile99_latencies[i],
            f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch={label[3]}\nseq={label[4]}\ncli={label[5]}\nasync={label[6]}\nsucc-rate={success_rate}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax3.set_xlabel("Configuration")
    ax3.set_ylabel("99th Percentile Latency")
    ax3.set_title("Comparison of 99th Percentile Latencies", loc="left")

    # Create the chart for throughput
    ax4.bar(x_ticks, throughputs, color=[colors.get(format_val, "gray") for format_val in labels["format"]])
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([])  # Remove x-axis labels

    # Add x-axis labels inside the bars
    for i, label in enumerate(labels.values):
        p99 = "{:.4f}".format(percentile99_latencies[i])
        ax4.text(
            i,
            throughputs[i],
            f"{label[0]}\n{label[1]}\np99={p99}\nhalf={label[2]}\nbatch={label[3]}\nseq={label[4]}\ncli={label[5]}\nasync={label[6]}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax4.set_xlabel("Configuration")
    ax4.set_ylabel("throughput")
    ax4.set_title("Throughput", loc="left")

    # Adjust spacing between subplots and ensure labels fit
    plt.tight_layout()

    # Save the charts as PNG files
    fig1.savefig("median_latencies.png", dpi=300)
    fig2.savefig("percentile90_latencies.png", dpi=300)
    fig3.savefig("percentile99_latencies.png", dpi=300)
    fig4.savefig("throughputs.png", dpi=300)

    # Close the plots
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


if __name__ == "__main__":
    create_charts()
