import csv
from dataclasses import replace
from typing import NamedTuple, Dict
from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata
import onnx
from hugging_bench.hugging_bench_config import Input, Output
import os
import os
import subprocess
import logging
from threading import Thread


ONNX_BACKEND = "onnxruntime_onnx"
TORCH_BACKEND = "pytorch_libtorch"
OPENVINO_BACKEND = "openvino"
TRT_BACKEND = "tensorrt_plan"
PRINT_HEADER = "\n\n============================%s=====================================\n"
ENV_TRITON_SERVER_DOCKER = "triton_server_docker_image"

LOG = logging.getLogger(__name__)


def dtype_np_type(dtype: str):
    from tritonclient.utils import triton_to_np_dtype
    from hugging_bench_triton import TritonConfig
    return triton_to_np_dtype(TritonConfig.DTYPE_MAP.get(dtype, None))


def run_docker(image_name, workspace=None, docker_args=[]):
    import shlex

    # Construct Docker command
    if (not workspace):
        workspace = os.getcwd()
    command = f'docker run --gpus=all -v {workspace}:{workspace} -w {workspace}  {image_name} {" ".join(docker_args)}'
    try:
        # Run command
        LOG.info(command)

        process = subprocess.Popen(shlex.split(command))
        # Get output and errors
        error = process.communicate()

        if process.returncode != 0:
            # If there are errors, raise an exception
            raise Exception(f'Error executing Docker container: {error}')
    except Exception as e:
        raise e


def print_container_logs(container, callback=None):
    """
    Prints logs of a Docker container until a specific message appears or a timeout is reached.
    """
    for line in container.logs(stream=True):
        log_line = line.strip().decode('utf-8')
        LOG.info(log_line)
        callback(log_line) if callback else None


def run_docker_sdk(image_name, workspace=None, docker_args=[], gpu=False, env={}, model_input=None):
    import docker
    client = docker.from_env()

    if (not workspace):
        workspace = os.getcwd()

    volumes = {
        workspace: {'bind': workspace, 'mode': 'rw'},
        model_input: {'bind': "/model_input", 'mode': 'rw'}
    }

    LOG.info(
        f"Running Docker container {image_name} gpu: {gpu} with command: {docker_args} and volumes: {volumes}")
    container = client.containers.run(
        image_name,
        command=docker_args,
        volumes=volumes,
        device_requests=[
            docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])] if gpu else [],
        working_dir=workspace,
        detach=True,
        environment=env,
        auto_remove=True
    )

    t = Thread(target=print_container_logs, args=[container])
    t.start()
    exit_code = container.wait()
    LOG.info(f"Docker container exit code {exit_code}")
    return exit_code


SHAPE_MAP = {
    "batch_size": 1,
    "sequence_length": -1,
    "width": 224,
    "height": 224,
    "channels": 3,
    "num_channels": 3,
    "audio_sequence_length": 16000,
    "nb_max_frames": 3000,
    "feature_size": 80,
}


def get_dim_value(dim):
    if isinstance(dim, int) or dim.isnumeric():
        return int(dim)
    value = SHAPE_MAP.get(dim, -1)
    if value is None:
        raise ValueError(f"Dimension {dim} not found in SHAPE_MAP")
    return value


def half_fp32(input):
    new_input = replace(input, dtype="FP16" if input.dtype ==
                        "FP32" else input.dtype)
    return new_input


def hf_model_input(onnx_model_path: str, half=False):
    onnx_model = onnx.load(onnx_model_path)
    input_metadata_dict = get_input_metadata(onnx_model.graph)
    inputs = []
    for input_name, input_metadata in input_metadata_dict.items():
        dims = [get_dim_value(dim)
                for dim in input_metadata.shape if dim != "batch_size"]
        dtype = str(input_metadata.dtype).upper()
        dtype = format_dtype(dtype)
        inputs.append(Input(name=input_name, dtype=dtype, dims=dims))

    return list(map(half_fp32, inputs)) if half else inputs


def hf_model_output(onnx_model_path: str, half=False):
    onnx_model = onnx.load(onnx_model_path)
    output_metadata_dict = get_output_metadata(onnx_model.graph)
    outputs = []
    for output_name, output_metadata in output_metadata_dict.items():
        dims = [get_dim_value(dim) for dim in list(
            output_metadata.shape) if dim != "batch_size"]
        dtype = str(output_metadata.dtype).upper()
        dtype = format_dtype(dtype)
        outputs.append(Output(name=output_name, dtype=dtype, dims=dims))

    return list(map(half_fp32, outputs)) if half else outputs


def format_dtype(dtype):
    if (dtype == 'FLOAT32'):
        return "FP32"
    elif (dtype == 'FLOAT16'):
        return "FP16"
    else:
        return dtype


class ExperimentSpec(NamedTuple):
    format: str
    device: str
    half: bool


def append_to_csv(spec_dict: Dict, info: Dict, csv_file: str):
    """
    Appends the given Spec instance and info dictionary to a CSV file.

    Parameters
    ----------
    spec : Spec
        Instance of Spec class.
    info : dict
        Additional information to be written to the CSV file.
    csv_file : str
        The CSV file to append to.
    """
    # Merge Spec fields and info into a single dict
    data = {**spec_dict, **info}

    # Define fieldnames with Spec fields first
    fieldnames = list(spec_dict.keys()) + list(info.keys())

    # Check if the file exists to only write the header once
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            LOG.info(f"Writing header to CSV file {fieldnames}")
            writer.writeheader()  # Write header only once

        LOG.info(f"Writing data to CSV file '{csv_file}': {data}")
        writer.writerow(data)
