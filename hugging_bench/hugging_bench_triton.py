# Models to bench from HF:
# bert-base-uncased
# distilbert-base-uncased
# resnet50 from HF (microsoft/resnet-50)

# from typing import Any

from hugging_bench_config import ModelInfo
from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
from types import MappingProxyType
from hugging_bench_util import PRINT_HEADER, ONNX_BACKEND, OPENVINO_BACKEND
import os, logging
import multiprocessing
from hugging_bench_config import TritonServerSpec

multiprocessing.set_start_method('spawn')

LOG = logging.getLogger(__name__)

class TritonConfig:
    
    # numpy types
    DTYPE_MAP = MappingProxyType({
            "INT64": DataType.TYPE_INT64,
            "FP32": DataType.TYPE_FP32,
            "FP16": DataType.TYPE_FP16,
            # add more dtype mappings if needed
        })
    
    BACKEND_MAP = MappingProxyType({
            "onnx": ONNX_BACKEND,
            "openvino": OPENVINO_BACKEND,
            # add more backend mappings if needed
        })
    
    def __init__(self, server_spec: TritonServerSpec, model_info: ModelInfo) -> None:
        self.model_info = model_info
        self.model_repo = os.path.abspath(server_spec.model_repository_dir)
        self.grpc_port = server_spec.grpc_port
        self.http_port = server_spec.http_port


    def create_model_repo(self, max_batch_size=1):
        from google.protobuf import text_format
        import shutil

        LOG.info(PRINT_HEADER % 'CTREAT TRITON CONFIG')
        if(not self.model_repo):
            raise Exception("No model repo is set")
        
        conf = self._config(max_batch_size)
        conf_pbtxt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
        conf_dir = os.path.join(self.model_repo, self.model_info.unique_name())
        
        if os.path.isdir(conf_dir):
            LOG.info(f"Removing existing model directory: {conf_dir}")
            shutil.rmtree(conf_dir)
        
        model_dir = os.path.join(conf_dir, "1")
        os.makedirs(model_dir, exist_ok=True)
    
        [shutil.copy(file, model_dir) for file in self.model_info.model_file_path()]
        
        config_path = os.path.join(conf_dir, "config.pbtxt")
        try:
            with open(config_path, 'w') as file:
                file.write(conf_pbtxt)
                LOG.info(f"Config written to {config_path} \n\n{conf_pbtxt}\n")
        except Exception as e:
            LOG.error(f"Error occurred while writing to file: {e}")
            raise e
        return self    


    def _config(self, max_batch_size):
        if(self.model_info.format.format_type == "onnx"):
            return ModelConfig(
                name=self.model_info.unique_name(),
                max_batch_size=max_batch_size,
                input=self._model_input(),
                output=self._model_output(),
                platform = self.BACKEND_MAP.get(self.model_info.format.format_type)
            )
        elif(self.model_info.format.format_type == "openvino"):
            return ModelConfig(
                name=self.model_info.unique_name(),
                max_batch_size=max_batch_size,
                input=self._model_input(),
                output=self._model_output(),
                backend = self.BACKEND_MAP.get(self.model_info.format.format_type)
            )
        else:
            raise Exception(f"Unsupported model format {self.model_info.format.format_type}")

    
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
    
    
    
import docker
import time

class TritonServer:  # This is just a placeholder. Replace it with your actual class.
    def __init__(self, triton_config, no_processor=1):
        self.model_repo = triton_config.model_repo
        self.model_name = triton_config.model_info.unique_name()
        self.no_processor = no_processor
        self.gpu = triton_config.model_info.gpu_enabled()
        self.container = None

    def _print_triton_bootup_logs(self, container, timeout, stop_message="Started Metrics Service"):
        """
        Prints logs of a Docker container until a specific message appears or a timeout is reached.
        """
        LOG.info(PRINT_HEADER % " TRITON SERVER LOGS ")
        stop_time = time.time() + timeout
        for line in container.logs(stream=True):
            log_line = line.strip().decode('utf-8')
            LOG.info(log_line)
            if stop_message in log_line or time.time() > stop_time:
                break

    def start(self, tritonserver_docker='nvcr.io/nvidia/tritonserver:23.04-py3'):
        LOG.info(PRINT_HEADER % " STARTING TRITON SERVER ")
        self.client = docker.from_env()

        volumes = {
            self.model_repo: {'bind': '/models', 'mode': 'rw'}
        }

        ports = {
            '8000/tcp': 8000,
            '8001/tcp': 8001,
            '8002/tcp': 8002,
        }

        self.container = self.client.containers.run(
            tritonserver_docker,
            command=["tritonserver", "--model-repository=/models"],
            volumes=volumes,
            cpu_count= 0 if (self.gpu) else self.no_processor,
            device_requests=[
                        docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])] if self.gpu else [],
            ports=ports,
            detach=True,
            auto_remove=True
            )
        
                        
        LOG.info(f"Starting container {self.container.name}")
        self._print_triton_bootup_logs(self.container, 10) 
        
        return self
    
    def stop(self):
        try:
            if(not self.container):
                LOG.info("No container found")
            elif(self.container.status in ["running", "created"]):
                self.container.stop()
                LOG.info("Container stopped")
            else:
                LOG.info(f"Skipped container.stop(). container status: {self.container.status}")
            return self
        except Exception as e:
            LOG.error(e)
            return self
    