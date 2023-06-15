from gevent import monkey
monkey.patch_all()
from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec
from hugging_bench.hugging_bench_runner import ExperimentRunner
import argparse
import logging

LOG = logging.getLogger("CLI")


def mlperf():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="runbench options")

    # Define the command-line arguments with their default values
    parser.add_argument("--format", default=["onnx"], nargs="*", choices=["onnx", "trt", "openvino"])
    parser.add_argument("--device", default=["cpu"], nargs="*", choices=["cpu", "cuda"])
    parser.add_argument("--half", default=[False, True], nargs="*", type=bool, help="Whether to use half precision")
    parser.add_argument(
        "--client_worker",
        default=[1],
        nargs="*",
        type=int,
        help="Number of client workers sending concurrent requests to Triton",
    )
    parser.add_argument(
        "--hf_ids", default=["prajjwal1/bert-tiny"], nargs="*", help="HuggingFace model ID(s) to benchmark"
    )
    parser.add_argument(
        "--model_local_path",
        default=None,
        nargs="*",
        help="If not specified, will download from HuggingFace. When given a task name must also be specified.",
    )
    parser.add_argument(
        "--task", default=None, nargs="*", help="Model task(s) to benchmark. Used with --model_local_path"
    )
    parser.add_argument(
        "--batch_size", default=[1], nargs="*", help="Batch size(s) to use for inference.."
    )
    parser.add_argument(
        "--instance_count", default=1, type=int, help="Triton server instance count."
    )

    #  potential hf_ids: ["bert-base-uncased", "distilbert-base-uncased", "microsoft/resnet-5"]

    # Parse the command-line arguments
    args = parser.parse_args()

    # Store the arguments in variables of the same name
    format_types = args.format
    devices = args.device
    half = args.half
    client_workers = args.client_worker
    hf_ids = args.hf_ids
    model_local_path = args.model_local_path
    task = args.task
    batch_size = args.batch_size

    experiments = []
    for f in format_types:
        for d in devices:
            for h in half:
                for w in client_workers:
                    for b in batch_size:
                        experiment = ExperimentSpec(format=f, device=d, half=h, client_workers=w, batch_size=b)
                        if experiment.is_valid():
                            LOG.info(f"Adding valid experiment: {experiment}")
                            experiments.append(ExperimentSpec(format=f, device=d, half=h, client_workers=w))
                        else:
                            LOG.info(f"Skipping invalid experiment: {experiment}")

    for idx, hf_id in enumerate(hf_ids):
        ExperimentRunner(
            hf_id,
            experiments,
            TritonServerSpec(instance_count=args.instance_count),
            dataset=None,
            model_local_path=model_local_path[idx] if model_local_path else None,
            task=task[idx] if task else None,
        ).run()


## run mlperf by default
if __name__ == "__main__":
    mlperf()
