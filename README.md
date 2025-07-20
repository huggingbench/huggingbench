# HuggingBench

[![python](https://img.shields.io/badge/python-3.9%20|%203.10-blue)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://github.com/legobench/huggingbench/actions/workflows/main.yml/badge.svg?branch=main&event=push)](https://github.com/legobench/huggingbench/actions/workflows/main.yml)
[![Issues](https://img.shields.io/github/issues/legobench/huggingbench)](https://github.com/legobench/huggingbench/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/legobench/huggingbench)](https://github.com/legobench/huggingbench/pulls)
[![License](https://img.shields.io/github/license/legobench/huggingbench)](LICENSE)

<div align="center">
    <img src="./docs/huggingbench-logo.jpg" height="200" alt="HuggingBench">
</div>

<h3 align="center">Benchmark and experiment with ML model serving</h3>

## 🚀 Project Overview

**HuggingBench** is an extensible, open-source MLOps tool for benchmarking and experimenting with ML model serving. It automates:

- Exporting models to various formats (ONNX, TensorRT, OpenVINO)
- Deploying them on inference servers (e.g., NVIDIA Triton)
- Running load tests with configurable concurrency, batch size, and more
- Collecting and visualizing performance metrics (latency, throughput, etc.) using Prometheus and Grafana

**Key Features:**
- Plugin architecture for easy extension
- Automated Docker usage for model conversion and serving
- Experiment runner for matrix-based benchmarking
- Metrics and visualization with Prometheus & Grafana
- Support for HuggingFace models and local PyTorch models

---

# 🤖 Introduction

HuggingBench helps you quickly evaluate ML model serving configurations, automate deployment, and analyze performance. Run a single command to:

- Generate model serving configs
- Deploy models
- Perform load testing
- View metrics and charts

**Design Principles:**
- Extensibility
- Reproducibility
- Production-fidelity

# 📋 Requirements

- Python 3.9 or 3.10
- Docker (for running inference servers)
  - [Install Docker Engine (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/)
  - [Install Docker Desktop (Mac)](https://docs.docker.com/desktop/install/mac-install/)

**GPU support (optional):**
- For NVIDIA GPU acceleration, see [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit#getting-started)

**Mac M1/M2:**
- CPU inference works; OpenVINO may not.

# 🏃 Getting Started

> HuggingBench runs locally on your system.

**1. Clone the repo:**
```bash
git clone https://github.com/legobench/huggingbench.git
```

**2. Build Docker images:**
```bash
./docker/build-all.sh
```

**3. Create and activate Python environment:**
```bash
python -m venv env
source env/bin/activate
```

**4. Install dependencies:**
```bash
pip install -e .
```

**5. Pull Triton server Docker image:**
```bash
docker pull nvcr.io/nvidia/tritonserver:23.04-py3
```

**6. Run a benchmark:**
```bash
hbench triton --id prajjwal1/bert-tiny --client_workers 1 2 4
```

This runs experiments with 1, 2, and 4 concurrent clients, spinning up Triton server containers, deploying the model, and generating load. Results and charts are saved locally.

# 💡 How it works

See the [architecture diagram](./docs/huggingbench-arch.png) for an overview.

HuggingBench uses Docker containers for model conversion and serving. Models and configs are stored on disk and shared with containers. Plugins allow easy extension to new inference servers.

# 🧩 Experiments

Experiments are generated as the cartesian product of CLI arguments. For example, `--format onnx openvino` and `--client_workers 1 2 4` produces 6 experiments.

# 📈 Prometheus metrics

Metrics are collected with Prometheus and visualized in Grafana. See [observability setup](./docker/observability/README.md).

# 🔌 Plugins

Add new inference servers by implementing a plugin. See [plugin.py](./src/bench/plugin.py) and [triton plugin example](./src/plugins/triton/).

# 📖 More Examples

See the [docs](./docs/development.md) for more advanced usage and examples.

# 💁 Contributing

We welcome contributions! Please open issues for bugs, feature requests, or suggestions. Fork and submit a pull request for code changes. Discuss improvements on the [GitHub issues page](https://github.com/legobench/huggingbench/issues).

# 📜 License

Distributed under the Apache License Version 2.0. See [LICENSE](LICENSE).

---

# 🧩 MCP Server Integration

HuggingBench can be extended to support MCP (Model Context Protocol) servers as inference backends. To use a GitHub MCP server:

1. **Set up the MCP server:**
   - Clone the MCP server repo (e.g., `microsoft/model-context-protocol`) and follow its setup instructions.
2. **Integrate with HuggingBench:**
   - Implement a plugin in `src/bench/plugin.py` or `src/plugins/` to communicate with the MCP server endpoint.
   - Configure HuggingBench to use the MCP server via CLI or config file.
3. **Run experiments:**
   - Use HuggingBench CLI, specifying the MCP server as the backend.

See [plugin documentation](./src/bench/plugin.py) for details on writing new plugins.

---

# 💬 Community & Support

- [GitHub Issues](https://github.com/legobench/huggingbench/issues) — Report bugs, request features, ask questions
- [Pull Requests](https://github.com/legobench/huggingbench/pulls) — Contribute code
- [Development Docs](./docs/development.md)
