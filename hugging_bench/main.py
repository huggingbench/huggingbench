from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec
from hugging_bench.hugging_bench_runner import ExperimentRunner
import argparse

def mlperf():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="runbench options")

    # Define the command-line arguments with their default values
    parser.add_argument("--format", default="onnx", choices=["onnx", "trt", "openvino"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--half", default=False, type=bool)
    parser.add_argument("--client_worker", default=1, type=int)
    parser.add_argument("--hf_ids", default=["prajjwal1/bert-tiny"], nargs="*")

    #  potential hf_ids: ["bert-base-uncased", "distilbert-base-uncased", "microsoft/resnet-5"]

    # Parse the command-line arguments
    args = parser.parse_args()

    # Store the arguments in variables of the same name
    format_type = args.format
    device = args.device
    half = args.half
    client_worker = args.client_worker
    hf_ids = args.hf_ids

    for hf_id in hf_ids:
        ExperimentRunner(hf_id, [ExperimentSpec(format=format_type, device=device, half=half, client_workers=client_worker)], TritonServerSpec(), dataset=None).run()
