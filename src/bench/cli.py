from gevent import monkey

monkey.patch_all()  # this is needed to make gevent work with Threads

from bench.plugin_manager import PluginManager, PLUGINS
from client.dataset import MODEL_DATASET


import argparse
import logging
import sys

from bench.config import TEMP_DIR, ExperimentSpec
from bench.exp_runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("CLI")

plugin_manager = PluginManager()


def add_common_args(parser: argparse.ArgumentParser):
    # Define the command-line arguments with their default values

    parser.add_argument("--id", help="HuggingFace model ID to benchmark or unique model identfier", required=True)
    parser.add_argument("--format", default=["onnx"], nargs="*", choices=["onnx", "trt", "openvino"])
    parser.add_argument("--device", default=["cpu"], nargs="*", choices=["cpu", "gpu"])
    parser.add_argument(
        "--precision",
        default=["fp32"],
        nargs="*",
        choices=["fp32", "fp16"],  # TODO:  add int8 support in the future
        help="What precision to use when converting the model.",
    )
    parser.add_argument(
        "--client_workers",
        default=[1],
        nargs="*",
        type=int,
        help="Number of client workers sending concurrent requests to the server.",
    )
    parser.add_argument("--batch_size", default=[1], nargs="*", type=int, help="Batch size(s) to use for inference..")
    parser.add_argument("--sequence_length", default=[100], nargs="*", type=int, help="Sequence length(s) to use.")
    parser.add_argument("--instance_count", default=[1], nargs="*", type=int, help="ML model instance count.")
    parser.add_argument(
        "--workspace", default=TEMP_DIR, help="Directory holding model configuration and experiment results"
    )
    parser.add_argument(
        "--dataset_id",
        default="random",
        choices=MODEL_DATASET.keys(),
        help="HuggingFace Dataset ID to use for benchmarking. By default we generate random dataset.",
    )
    parser.add_argument(
        "--model_local_path",
        default=None,
        help="Path of model to benchmark. Model has to be in PyTorch format and folder must contain `config.json`. You have to provide `--task` (can't be autodetected) and `--id`(since we use it as a unique identifier for the model).",
    )
    parser.add_argument(
        "--task", default="autodetect", help="Model tasks to benchmark. Only used with --model_local_path"
    )


def hbench():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="HuggingBench CLI")
    subparsers = parser.add_subparsers(
        dest="plugin",
        required=True,
        help="Choose one of following plugins to run: " + str(PLUGINS),
    )

    plugin_parsers = {name: subparsers.add_parser(name) for name in PLUGINS}
    plugin_manager.arg_parsers(plugin_parsers)

    # Add the arguments shared for all plugins
    for plugin_parser in plugin_parsers.values():
        add_common_args(plugin_parser)

    if len(sys.argv) == 1:
        args = parser.parse_args(["--help"])
    elif len(sys.argv) == 2:
        args = parser.parse_args(sys.argv[1:] + ["--help"])
    else:
        args = parser.parse_args()

    run(args)
    # Run the plugin


def run(args):
    format_types = args.format
    devices = args.device
    precisions = args.precision
    client_workers = args.client_workers
    hf_id = args.id
    model_local_path = args.model_local_path
    task = args.task
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    instance_count = args.instance_count
    workspace_dir = args.workspace
    dataset_id = args.dataset_id
    async_req = args.async_req

    triton_plugin = plugin_manager.get_plugin("triton")

    experiments = []
    for f in format_types:
        for d in devices:
            for p in precisions:
                for w in client_workers:
                    for b in batch_size:
                        for s in sequence_length:
                            for i in instance_count:
                                experiment = ExperimentSpec(
                                    id=hf_id,
                                    task=task,
                                    model_local_path=model_local_path,
                                    format=f,
                                    device=d,
                                    precision=p,
                                    clients=w,
                                    batch_size=b,
                                    sequence_length=s,
                                    instances=i,
                                    workspace_dir=workspace_dir,
                                    dataset=dataset_id,
                                    async_req=async_req,
                                )
                                if experiment.is_valid():
                                    LOG.info(f"Adding valid experiment: {experiment}")
                                    experiments.append(experiment)
                                else:
                                    LOG.warning(f"Skipping invalid experiment: {experiment}")

    ExperimentRunner(
        triton_plugin,
    ).run(experiments)


if __name__ == "__main__":
    hbench()
