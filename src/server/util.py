import csv
import logging
import os
from dataclasses import replace
from threading import Thread
from typing import Dict

import onnx
from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata

from bench.config import Input, Output

PRINT_HEADER = "\n\n============================%s=====================================\n"
ENV_TRITON_SERVER_DOCKER = "triton_server_docker_image"

LOG = logging.getLogger(__name__)


def print_container_logs(container, callback=None):
    """
    Prints logs of a Docker container until a specific message appears or a timeout is reached.
    """
    for line in container.logs(stream=True, stdout=True, stderr=True):
        log_line = line.strip().decode("utf-8")
        LOG.info(log_line)
        callback(log_line) if callback else None


def run_docker_sdk(image_name, workspace=None, docker_args=[], gpu=False, env={}, model_input=None):
    import docker

    client = docker.from_env()

    if not workspace:
        workspace = os.getcwd()

    volumes = {workspace: {"bind": workspace, "mode": "rw"}, model_input: {"bind": "/model_input", "mode": "rw"}}

    LOG.info(
        f"Running docker image: {image_name} gpu: {gpu} volumes: {volumes} env: {env}: \ncommand: {' '.join(docker_args)}\n"
    )

    container = client.containers.run(
        image_name,
        command=docker_args,
        volumes=volumes,
        device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[["gpu"]])] if gpu else [],
        working_dir=workspace,
        detach=True,
        environment=env,
        auto_remove=True,
        shm_size="1G",  # TODO: make this configurable
    )

    t = Thread(target=print_container_logs, args=[container])
    t.start()
    exit_code = container.wait()
    t.join()
    if exit_code["StatusCode"] != 0:
        raise Exception(f"Container failed {exit_code}. Check log output for failure reason(s)")
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


def get_dim_value(dim, custom_shape_map):
    if isinstance(dim, int) or dim.isnumeric():
        return int(dim)
    shapes = {**SHAPE_MAP, **custom_shape_map}
    value = shapes.get(dim, -1)
    if value is None:
        raise ValueError(f"Dimension {dim} not found in SHAPE_MAP")
    return value


def half_fp32(input):
    new_input = replace(input, dtype="FP16" if input.dtype == "FP32" else input.dtype)
    return new_input


def half_int64(input):
    new_input = replace(input, dtype="INT32" if input.dtype == "INT64" else input.dtype)
    return new_input


def hf_model_input(onnx_model_path: str, half=False, int64to32=False, custom_shape_map={}):
    onnx_model = onnx.load(onnx_model_path)
    input_metadata_dict = get_input_metadata(onnx_model.graph)
    inputs = []
    for input_name, input_metadata in input_metadata_dict.items():
        dims = [get_dim_value(dim, custom_shape_map) for dim in input_metadata.shape if dim != "batch_size"]
        dtype = str(input_metadata.dtype).upper()
        dtype = format_dtype(dtype)
        inputs.append(Input(name=input_name, dtype=dtype, dims=dims))

    inputs = list(map(half_fp32, inputs)) if half else inputs
    inputs = list(map(half_int64, inputs)) if int64to32 else inputs
    return inputs


def hf_model_output(onnx_model_path: str, half=False, custom_shape_map={}):
    onnx_model = onnx.load(onnx_model_path)
    output_metadata_dict = get_output_metadata(onnx_model.graph)
    outputs = []
    for output_name, output_metadata in output_metadata_dict.items():
        dims = [get_dim_value(dim, custom_shape_map) for dim in list(output_metadata.shape) if dim != "batch_size"]
        dtype = str(output_metadata.dtype).upper()
        dtype = format_dtype(dtype)
        outputs.append(Output(name=output_name, dtype=dtype, dims=dims))

    return list(map(half_fp32, outputs)) if half else outputs


def format_dtype(dtype):
    if dtype == "FLOAT32":
        return "FP32"
    elif dtype == "FLOAT16":
        return "FP16"
    else:
        return dtype


HF_TOKEN_ENV = "HUGGING_FACE_HUB_TOKEN"
HF_TOKEN_FILE = os.getenv("HOME") + "/.cache/huggingface/token"


def hf_token() -> str:
    token = None
    token_exists = os.path.isfile(HF_TOKEN_FILE)
    if token_exists:
        with open(HF_TOKEN_FILE) as f:
            token = f.read()
    else:
        token = os.getenv(HF_TOKEN_ENV)

    return token
