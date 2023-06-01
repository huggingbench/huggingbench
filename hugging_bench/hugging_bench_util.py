import os
import os
import subprocess
import logging
from threading import Thread
from typing import Any

import numpy as np

from hugging_bench_config import Format, ModelInfo
from hugging_bench_config import ExperimentSpec
# from model_config_constants import *

ONNX_BACKEND = "onnxruntime_onnx"
TORCH_BACKEND = "pytorch_libtorch"
OPENVINO_BACKEND = "openvino"
TRT_BACKEND = "tensorrt_plan"
PRINT_HEADER = "\n\n============================%s=====================================\n"

LOG = logging.getLogger(__name__)
class ModelExporter:
    # export model to onnx, openvino, trt...
    
    def __init__(self, hf_id, spec: ExperimentSpec, task=None, base_dir=None) -> None:
        self.hf_id = hf_id
        self.spec = spec
        self.task = task
        self.base_dir = base_dir if(base_dir) else os.getcwd()
        
    def export(self, model_input_path: str = None) -> ModelInfo:
        #  onnx format is a starting point    
        model_info = self._export_hf2onnx("0.001", self.spec.device, self.spec.half, model_input_path)
        model_info = model_info.with_shapes(
                    input_shape=hf_model_input(model_info.model_file_path(), half=model_info.half()), 
                    output_shape=hf_model_output(model_info.model_file_path(), half=model_info.half()))  
        
        
        if(self.spec.format == "onnx"):   
            None
        elif(self.spec.format == "openvino"):
            model_info = self._export_onnx2openvino(model_info)
        elif(self.spec.format == "trt"):
            model_info = self._export_onnx2trt(model_info)
        else:
            raise Exception(f"Unknown format {self.spec.format}")
        
        LOG.info(f"Model info {model_info}")
        return model_info

    def _export_hf2onnx(self, atol=0.001, device=None, half=False, model_input:str = None) -> ModelInfo:
        print(PRINT_HEADER % " ONNX EXPORT ")
        model_info = ModelInfo(self.hf_id, self.task, Format("onnx", {"atol": atol, "device": device, "half": half}), base_dir=self.base_dir)
        
        if(all(os.path.exists(file) for file in model_info.model_file_path())): 
            LOG.info(f"Model already exists at {model_info.model_file_path()}")
            return model_info
        
        model_dir = model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        model_arg = f"--model={self.hf_id}" if model_input is None else f"--model=/model_input"

        cmd = [
            "optimum-cli", "export", "onnx",
            model_arg, 
            "--framework=pt",
            "--monolit", 
            f"--atol={atol}"]
        
        if(not half and device):
            cmd.append(f"--device={device}")
        
        if(half):
            cmd.append("--fp16")
            cmd.append("--device=cuda")
        
        if(self.task):
            cmd.append(f"--task={self.task}")
        

        
        cmd.append(model_info.model_dir())
        
        run_docker_sdk("optimum", model_dir, cmd, model_info.gpu_enabled(), model_input=model_input)
        
        return model_info


    def _export_onnx2openvino(self, onnx_model_info: ModelInfo):
        LOG.info(PRINT_HEADER % " ONNX 2 OPENVINO CONVERSION ")   
        ov_model_info = ModelInfo(
            onnx_model_info.hf_id, 
            onnx_model_info.task, 
            format=Format("openvino", origin=onnx_model_info.format), 
            base_dir=self.base_dir, 
            input_shape=onnx_model_info.input_shape,
            output_shape=onnx_model_info.output_shape)
        model_dir = ov_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)
        
        cmd = [
            "mo",
            f"--input_model={onnx_model_info.model_file_path()[0]}",
            f"--output_dir={model_dir}"
        ]
        run_docker_sdk(image_name="openvino", docker_args=cmd)
        return ov_model_info
    

    def _export_onnx2trt(self, onnx_model_info):
        LOG.info(PRINT_HEADER % " ONNX 2 TRT CONVERSION ") 
        trt_model_info = ModelInfo(
            onnx_model_info.hf_id, 
            onnx_model_info.task, 
            Format("trt", origin=onnx_model_info.format), 
            self.base_dir,
            input_shape=onnx_model_info.input_shape,
            output_shape=onnx_model_info.output_shape)
        
        model_dir = trt_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        input_str = ' '.join([f"{input.name}:{input.dims}" for input in trt_model_info.input_shape])

        cmd = [
            "polygraphy",
            "convert",
            "--model-type=onnx",
            "--convert-to=trt",
            f"--input-shapes={input_str}",
            f"--output={trt_model_info.model_file_path()[0]}",
            onnx_model_info.model_file_path()[0]
        ]
        run_docker_sdk(image_name="nvcr.io/nvidia/tensorrt:23.04-py3", docker_args=cmd, gpu=True)
        return trt_model_info


    def _inspect_onnx(self, model_info: ModelInfo):
        LOG.info(PRINT_HEADER % " ONNX MODEL INSPECTION ")
        run_docker_sdk(image_name="nvcr.io/nvidia/tensorrt:23.04-py3", docker_args=["polygraphy", "inspect", "model", f"{model_info.model_file_path()[0]}", "--mode=onnx"], env={"POLYGRAPHY_AUTOINSTALL_DEPS": 1})
        

def dtype_np_type(dtype: str):
    from tritonclient.utils import triton_to_np_dtype
    from hugging_bench_triton import TritonConfig
    return triton_to_np_dtype(TritonConfig.DTYPE_MAP.get(dtype, None))
    

def run_docker(image_name, workspace=None, docker_args=[]):
    import shlex

    # Construct Docker command
    if(not workspace):
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

    if(not workspace):
        workspace = os.getcwd()

    volumes = {
        workspace: {'bind': workspace, 'mode': 'rw'},
        model_input: {'bind': "/model_input", 'mode': 'rw'}
    }

    LOG.info(f"Running Docker container {image_name} gpu: {gpu} with command: {docker_args}")
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
    
    t = Thread(target = print_container_logs, args=[container])
    t.start()
    exit_code = container.wait()
    LOG.info(f"Docker container exit code {exit_code}")
    return exit_code


from hugging_bench_config import Input, Output
import onnx
from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata

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
    new_input = input._replace(dtype = "FP16" if input.dtype=="FP32" else input.dtype)
    return new_input

def hf_model_input(onnx_model_path: str, half=False):
    onnx_model = onnx.load(onnx_model_path)
    input_metadata_dict = get_input_metadata(onnx_model.graph)
    inputs = []
    for input_name, input_metadata in input_metadata_dict.items():
        dims = [get_dim_value(dim) for dim in input_metadata.shape if dim != "batch_size"]
        dtype = str(input_metadata.dtype).upper()
        dtype=format_dtype(dtype)
        inputs.append(Input(name=input_name, dtype=dtype, dims=dims))

    return list(map(half_fp32, inputs)) if half else inputs


def hf_model_output(onnx_model_path: str, half=False):
    onnx_model = onnx.load(onnx_model_path)
    output_metadata_dict = get_output_metadata(onnx_model.graph)
    outputs = []
    for output_name, output_metadata in output_metadata_dict.items():
        dims = [get_dim_value(dim) for dim in list(output_metadata.shape) if dim != "batch_size"]
        dtype = str(output_metadata.dtype).upper()
        dtype=format_dtype(dtype)
        outputs.append(Output(name=output_name, dtype=dtype, dims=dims))
    
    return list(map(half_fp32, outputs)) if half else outputs


def format_dtype(dtype):
    if(dtype=='FLOAT32'):
        return "FP32"
    elif(dtype=='FLOAT16'):
        return "FP16"
    else:
        return dtype

import csv
from typing import NamedTuple, Dict

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

        LOG.info(f"Writing data to CSV file: {data}")
        writer.writerow(data)
