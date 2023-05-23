# Models to bench from HF:
# bert-base-uncased
# distilbert-base-uncased
# resnet50 from HF (microsoft/resnet-50)
import os
import subprocess
from typing import Any
import shutil
from hugging_bench_config import ModelInfo

from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
from types import MappingProxyType
from google.protobuf import text_format
from hugging_bench_util import PRINT_HEADER, ONNX_BACKEND, OPENVINO_BACKEND
import tritonclient.http as httpclient
import numpy as np
from tritonclient.utils import triton_to_np_dtype
from hugging_bench_util import dtype_np_type

class TritonConfig:
    # numpy types
    DTYPE_MAP = MappingProxyType({
            "INT64": DataType.TYPE_INT64,
            "FP32": DataType.TYPE_FP32,
            # add more dtype mappings if needed
        })
    
    BACKEND_MAP = MappingProxyType({
            "onnx": ONNX_BACKEND,
            "openvino": OPENVINO_BACKEND,
            # add more backend mappings if needed
        })
    
    def __init__(self, model_repo_dir: str, model_info: ModelInfo) -> None:
        self.model_info = model_info
        self.model_repo = os.path.abspath(model_repo_dir)


    def create_model_repo(self, max_batch_size=1):
        print(PRINT_HEADER % 'CTREAT TRITON CONFIG')
        if(not self.model_repo):
            raise Exception("No model repo is set")
        
        conf = self._config(max_batch_size)
        conf_pbtxt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
        conf_dir = os.path.join(self.model_repo, self.model_info.unique_name())
        
        if os.path.exists(conf_dir):
            print(f"Removing existing model directory: {conf_dir}")
            shutil.rmtree(conf_dir)
        
        model_dir = os.path.join(conf_dir, "1")
        os.makedirs(model_dir, exist_ok=True)
    
        shutil.move(self.model_info.model_file_path(), model_dir)
        
        config_path = os.path.join(conf_dir, "config.pbtxt")
        try:
            with open(config_path, 'w') as file:
                file.write(conf_pbtxt)
                print(f"Config written to {config_path} \n\n{conf_pbtxt}\n")
        except Exception as e:
            print(f"Error occurred while writing to file: {e}")
            raise e
        return self    


    def _config(self, max_batch_size):
        return ModelConfig(
            name=self.model_info.unique_name(),
            max_batch_size=max_batch_size,
            input=self._model_input(),
            output=self._model_output(),
            platform=self.BACKEND_MAP.get(self.model_info.format.format_type),
        )

    
    def _model_input(self) -> ModelInput:
        return [ 
            ModelInput(
            name=input.name,
            data_type=self.DTYPE_MAP.get(input.dtype, DataType.TYPE_FP32),  # Default to DataType.TYPE_FP32 if dtype not found in the mapping
            dims=input.dims) for input in self.model_info.input_shape ]


    def _model_output(self):
        return [ 
            ModelOutput(
            name=output.name,
            data_type=self.DTYPE_MAP.get(output.dtype, DataType.TYPE_FP32),  # Default to DataType.TYPE_FP32 if dtype not found in the mapping
            dims=output.dims) for output in self.model_info.output_shape ]
    


class TritonServer:
    def __init__(self, triton_config: TritonConfig) -> None:
        self.triton_config = triton_config
        self._server_url =  "localhost:8000"
    

    def start(self, gpu=False, no_processor=1, tritonserver_docker='nvcr.io/nvidia/tritonserver:23.04-py3'):
        if(not self.triton_config.model_repo):
            raise Exception("No model repo is set")
        
        command = [
            'docker', 'run', '-d', '--rm',
            '-p', '8000:8000',
            '-p', '8001:8001',
            '-p', '8002:8002',
            '--gpus' if gpu else '--cpus', f'{no_processor}',
            '-v', f'{self.triton_config.model_repo}:/models',
            tritonserver_docker,
            'tritonserver',
            '--model-repository=/models'
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")


    #  TODO not working atm
    def infer_with_rnd_sample(self):
        sample_input = self._create_sample_input()
        output_names = [os.name for os in self.triton_config.model_info.output_shape]
        model_name = self.triton_config.model_info.unique_name()
        server_target = self._server_url

        

        try:
            with httpclient.InferenceServerClient(url=server_target) as client:
                result = client.infer(model_name, sample_input, output_names=output_names)
                print("Inference response: {}".format(result))
        except Exception as e:
            print("Inference failed: {}".format(e))
        
        
    def _create_sample_input(self):
        sample_input = []

        for model_input in self.triton_config.model_info.input_shape:
            shape = model_input.dims
            dtype = dtype_np_type(model_input.dtype)
            sample_data = np.random.randn(*shape).astype(dtype)
            input_data = httpclient.InferInput(model_input.name, shape, dtype)
            input_data.set_data_from_numpy(sample_data)
            sample_input.append(input_data)

        return sample_input
        
