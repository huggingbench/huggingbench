import os
import os
import subprocess
from typing import Any

from hugging_bench_config import Format, ModelInfo
from hugging_bench_config import ExperimentSpec
# from model_config_constants import *

ONNX_BACKEND = "onnxruntime_onnx"
TORCH_BACKEND = "pytorch_libtorch"
OPENVINO_BACKEND = "openvino"
PRINT_HEADER = "\n\n============================%s=====================================\n"

class ModelExporter:
    # from util import just_export_hf_onnx_optimum_docker, convert_onnx2openvino_docker
    
    def __init__(self, hf_id, spec: ExperimentSpec, task=None, base_dir=None) -> None:
        self.hf_id = hf_id
        self.spec = spec
        self.task = task
        self.base_dir = base_dir if(base_dir) else os.getcwd()
        
    def export(self):
        model_info = None
        if(self.spec.format == "onnx"):
            model_info = self._export_hf2onnx("0.001", self.spec.device, self.spec.half)
        elif(self.spec.format == "openvino"):
            model_info = self._export_hf2onnx("0.001", self.spec.device, self.spec.half)
            model_info = self._export_onnx2openvino(model_info)
        else:
            raise Exception(f"Unknown format {self.spec.format}")

        model_info = model_info.with_shapes(
            input_shape=hf_model_input(self.hf_id, model_info.half()), 
            output_shape=hf_model_output(self.hf_id, model_info.half()))  
        
        return model_info

    def _export_hf2onnx(self, atol=0.001, device=None, half=False):
        print(PRINT_HEADER % " ONNX EXPORT ")
        model_info = ModelInfo(self.hf_id, self.task, Format("onnx", {"atol": atol, "device": device, "half": half}), base_dir=self.base_dir)
        
        if(all(os.path.exists(file) for file in model_info.model_file_path())): 
            print(f"Model already exists at {model_info.model_file_path()}")
            return model_info
        
        model_dir = model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        cmd = [
            "optimum-cli", "export", "onnx",
            f"--model={self.hf_id}", 
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
        
        run_docker_sdk("optimum", model_dir, cmd, model_info.gpu_enabled())
        
        return model_info


    def _export_onnx2openvino(self, onnx_model_info: ModelInfo):
        print(PRINT_HEADER % " ONNX 2 OPENVINO CONVERSION ")   
        ov_model_info = ModelInfo(onnx_model_info.hf_id, onnx_model_info.task, Format("openvino", origin=onnx_model_info), self.base_dir)
        model_dir = ov_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)
        
        cmd = [
            "mo",
            f"--input_model={onnx_model_info.model_file_path()[0]}",
            f"--output_dir={model_dir}"
        ]
        run_docker_sdk(image_name="openvino", docker_args=cmd)
        #  for mocking purposes
        # def create_empty_file(path):
        #     with open(path, 'w'):
        #         pass
        # create_empty_file(os.path.join(model_dir, "model.xml"))
        # create_empty_file(os.path.join(model_dir, "model.bin"))
        return ov_model_info
    

    def _inspect_onnx(self, model_info: ModelInfo):
        print(PRINT_HEADER % " ONNX MODEL INSPECTION ")
        run_docker_sdk(image_name="polygraphy", docker_args=["polygraphy", "inspect", "model", f"{model_info.model_file_path()[0]}", "--mode=onnx"])
        

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
        print(command)

        process = subprocess.Popen(shlex.split(command))
        # Get output and errors
        error = process.communicate()

        if process.returncode != 0:
            # If there are errors, raise an exception
            raise Exception(f'Error executing Docker container: {error}')
    except Exception as e:
        raise e



def run_docker_sdk(image_name, workspace=None, docker_args=[], gpu=False):
    import docker
    client = docker.from_env()

    if(not workspace):
        workspace = os.getcwd()

    volumes = {
        workspace: {'bind': workspace, 'mode': 'rw'}
    }

    container = client.containers.run(
        image_name,
        command=docker_args,
        volumes=volumes,
        device_requests=[
                    docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])] if gpu else [],
        working_dir=workspace,
        detach=True,
        auto_remove=True
        )
    
    def _print_triton_bootup_logs(container, timeout, stop_message="Started Metrics Service"):
        """
        Prints logs of a Docker container until a specific message appears or a timeout is reached.
        """
        import time
        stop_time = time.time() + timeout
        for line in container.logs(stream=True):
            log_line = line.strip().decode('utf-8')
            print(log_line)
            print(container.status)
            if stop_message in log_line or time.time() > stop_time or container.status in ["running", "created"]:
                break
    _print_triton_bootup_logs(container, timeout=100)

    return container.wait()

import timeit
import numpy as np

def measure_execution_time(func, num_executions) -> dict:
    """
    Executes a function a specified number of times and measures the execution time.

    Parameters
    ----------
    func : callable
        The function to execute.
    num_executions : int
        The number of times to execute the function.

    Returns
    -------
    dict
        A dictionary with keys 'median', '90_percentile' and '99_percentile' indicating 
        the execution time for median, 90th percentile and 99th percentile, respectively.
    """
    # Create a list to store execution times
    execution_times = []

    # Execute the function and measure execution time
    for _ in range(num_executions):
        start_time = timeit.default_timer()
        func()
        end_time = timeit.default_timer()
        execution_time = end_time - start_time

        # Store the execution time
        execution_times.append(execution_time)

    # Convert execution times to a numpy array
    execution_times = np.array(execution_times)

    # Calculate percentiles
    median = np.median(execution_times)
    percentile_90 = np.percentile(execution_times, 90)
    percentile_99 = np.percentile(execution_times, 99)

    return {'median': median, '90_percentile': percentile_90, '99_percentile': percentile_99}

from hugging_bench_config import Input, Output

def half_fp32(input):
    new_input = input._replace(dtype = "FP16" if input.dtype=="FP32" else input.dtype)
    return new_input

def hf_model_input(hf_id, task=None, sequence_length=-1, half=False):
    INPUTS = {
        "microsoft/resnet-50": [Input(name="pixel_values", dtype="FP32", dims=[3, 224, 224])],
        "bert-base-uncased": [
            Input(name="input_ids", dtype="INT64", dims=[sequence_length]),
            Input(name="attention_mask", dtype="INT64", dims=[sequence_length]),
            Input(name="token_type_ids", dtype="INT64", dims=[sequence_length]),
        ],
        "distilbert-base-uncased": [
            Input(name="input_ids", dtype="INT64", dims=[sequence_length]),
            Input(name="attention_mask", dtype="INT64", dims=[sequence_length]),
        ]
    }

    return list(map(half_fp32, INPUTS[hf_id])) if half else INPUTS[hf_id]


def hf_model_output(hf_id, task=None, sequence_length=-1, half=False):
    OUTPUTS = {
        "microsoft/resnet-50": [Output(name="logits", dtype="FP32", dims=[1000])],
        "bert-base-uncased": [Output(name="logits", dtype="FP32", dims=[sequence_length, 30522])],
        "distilbert-base-uncased": [Output(name="logits", dtype="FP32", dims=[sequence_length, 30522])]
    }
    
    return list(map(half_fp32, OUTPUTS[hf_id])) if half else OUTPUTS[hf_id]


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
            print(f"Writing header to CSV file {fieldnames}")
            writer.writeheader()  # Write header only once

        print(f"Writing data to CSV file: {data}")
        writer.writerow(data)

# Usage
# spec = Spec('png', 'cpu', False)
# info = {'additional_field': 'additional_value'}
# append_to_csv(spec, info, 'output.csv')

# # Usage
# def my_function():
#     for _ in range(1000000):
#         pass

# results = measure_execution_time(my_function, 100)
# print(results)

print(hf_model_input("microsoft/resnet-50", half=True))
print(hf_model_output("microsoft/resnet-50", half=True))