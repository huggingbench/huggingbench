import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("temp/prajjwal1-bert-tiny.csv")

# Extract required columns
labels = df[["format", "device", "half", "batch_size", "sequence_length", "client_workers", "async_clients"]]
median_latencies = df["median"]
percentile90_latencies = df["90_percentile"]
percentile99_latencies = df["99_percentile"]

# Set the figure size
fig1, ax1 = plt.subplots(figsize=(12, 6))
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig3, ax3 = plt.subplots(figsize=(12, 6))

# Create the chart for median latencies
x_ticks = range(len(median_latencies))
ax1.bar(x_ticks, median_latencies)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([])  # Remove x-axis labels

# Add x-axis labels inside the bars
for i, label in enumerate(labels.values):
    ax1.text(
        i,
        median_latencies[i],
        f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch_size={label[3]}\nsequence_length={label[4]}\nclient_workers={label[5]}\nasync_clients={label[6]}",
        ha="center",
        va="bottom",
    )

ax1.set_xlabel("Configuration")
ax1.set_ylabel("Median Latency")
ax1.set_title("Comparison of Median Latencies")

# Create the chart for 90th percentiles
ax2.bar(x_ticks, percentile90_latencies)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([])  # Remove x-axis labels

# Add x-axis labels inside the bars
for i, label in enumerate(labels.values):
    ax2.text(
        i,
        percentile90_latencies[i],
        f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch_size={label[3]}\nsequence_length={label[4]}\nclient_workers={label[5]}\nasync_clients={label[6]}",
        ha="center",
        va="bottom",
    )

ax2.set_xlabel("Configuration")
ax2.set_ylabel("90th Percentile Latency")
ax2.set_title("Comparison of 90th Percentile Latencies")

# Create the chart for 99th percentiles
ax3.bar(x_ticks, percentile99_latencies)
ax3.set_xticks(x_ticks)
ax3.set_xticklabels([])  # Remove x-axis labels

# Add x-axis labels inside the bars
for i, label in enumerate(labels.values):
    ax3.text(
        i,
        percentile99_latencies[i],
        f"{label[0]}\n{label[1]}\nhalf={label[2]}\nbatch_size={label[3]}\nsequence_length={label[4]}\nclient_workers={label[5]}\nasync_clients={label[6]}",
        ha="center",
        va="bottom",
    )

ax3.set_xlabel("Configuration")
ax3.set_ylabel("99th Percentile Latency")
ax3.set_title("Comparison of 99th Percentile Latencies")

# Adjust spacing between subplots and ensure labels fit
plt.tight_layout()

# Save the charts as PNG files
fig1.savefig("median_latencies.png", dpi=300)
fig2.savefig("percentile90_latencies.png", dpi=300)
fig3.savefig("percentile99_latencies.png", dpi=300)

# Close the plots
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
