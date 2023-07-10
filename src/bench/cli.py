import sys
from gevent import monkey

from bench.plugin_manager import PluginManager, PLUGINS

monkey.patch_all()  # this is needed to make gevent work with Threads
import argparse
import logging

from bench.chart import ChartGen
from bench.config import ExperimentSpec
from bench.exp_runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("CLI")


def add_common_args(parser: argparse.ArgumentParser):
    # Define the command-line arguments with their default values

    parser.add_argument("--format", default=["onnx"], nargs="*", choices=["onnx", "trt", "openvino"])
    parser.add_argument("--device", default=["cpu"], nargs="*", choices=["cpu", "gpu"])
    parser.add_argument(
        "--precision",
        default=["fp32"],
        nargs="*",
        choices=["fp32, fp16"],  # TODO:  add int8 support in the future
        help="What precision to use when converting the model.",
    )
    parser.add_argument(
        "--client_workers",
        default=[1],
        nargs="*",
        type=int,
        help="Number of client workers sending concurrent requests to the server.",
    )
    parser.add_argument("--hf_id", help="HuggingFace model ID(s) to benchmark", required=True)
    parser.add_argument(
        "--model_local_path",
        default=None,
        help="If not specified, will download from HuggingFace. When given a task name must also be specified.",
    )
    parser.add_argument("--task", default=None, help="Model tasks to benchmark. Only used with --model_local_path")
    parser.add_argument("--batch_size", default=[1], nargs="*", help="Batch size(s) to use for inference..")
    parser.add_argument("--instance_count", default=[1], nargs="*", help="ML model instance count.")
    parser.add_argument(
        "--workspace", default="temp/", help="Directory holding model configuration and experiment results"
    )
    parser.add_argument(
        "--dataset_id",
        default="random",
        help="HuggingFace dataset ID to use for benchmarking. By default we generate random dataset.",
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

    # Add the arguments shared for all plugins
    for plugin_parser in plugin_parsers.values():
        add_common_args(plugin_parser)

    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    # Run the plugin
    run(args)


def run(args):
    format_types = args.format
    devices = args.device
    precisions = args.precision
    client_workers = args.client_workers
    hf_id = args.hf_id
    model_local_path = args.model_local_path
    task = args.task
    batch_size = args.batch_size
    instance_count = args.instance_count
    workspace_dir = args.workspace
    dataset_id = args.dataset_id

    plugin_manager = PluginManager()
    triton_plugin = plugin_manager.get_plugin("triton")

    experiments = []
    for f in format_types:
        for d in devices:
            for p in precisions:
                for w in client_workers:
                    for b in batch_size:
                        for i in instance_count:
                            experiment = ExperimentSpec(
                                hf_id=hf_id,
                                task=task,
                                model_local_path=model_local_path,
                                format=f,
                                device=d,
                                precision=p,
                                client_workers=w,
                                batch_size=b,
                                instance_count=i,
                                workspace_dir=workspace_dir,
                                dataset_id=dataset_id,
                            )
                            if experiment.is_valid():
                                LOG.info(f"Adding valid experiment: {experiment}")
                                experiments.append(experiment)
                            else:
                                LOG.info(f"Skipping invalid experiment: {experiment}")

        ExperimentRunner(
            triton_plugin,
            experiments,
        ).run()


## run mlperf by default
if __name__ == "__main__":
    hbench()
