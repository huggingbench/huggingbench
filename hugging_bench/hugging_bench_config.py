
from dataclasses import dataclass, field
from typing import NamedTuple
# from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
import os

TEMP_DIR = "./temp"
TEMP_MODEL_REPO_DIR = f"{TEMP_DIR}/model_repository"

@dataclass
class TritonServerSpec:
    # TODO ports not used yet
    grpc_port: int = 8001
    http_port: int = 8000
    model_repository_dir: str = TEMP_MODEL_REPO_DIR

@dataclass
class ExperimentSpec:
    format: str
    device: str
    half: bool
    batch_size: int = 1
    sequence_length: int = 128

@dataclass
class Format:
    format_type: str # onnx, openvino, torchscript, tensorflow
    parameters: dict = field(default_factory=dict)
    origin: 'Format' = None

    def gpu_enabled(self):
        if self.parameters.get('device') == 'cuda' or (self.origin and self.origin.parameters.get('device') == 'cuda'):
            return True
        return False
    
    def half(self):
        if self.parameters.get('half', False) or (self.origin and self.origin.parameters.get('half', False)):
            return True
        return False

@dataclass
class Input:
    name: str
    dtype: str
    dims: list[int]
    
@dataclass
class Output:
    name: str
    dtype: str
    dims: list[int]
 
@dataclass
class ModelInfo:
    hf_id: str
    task: str
    format: Format
    base_dir: str
    input_shape: list[Input] =  field(default_factory=list)  
    output_shape: list[Output] = field(default_factory=list)

    def unique_name(self):
        params_str = f"-{self.param_str()}" if(self.param_str()) else ""
        return f"{self.hf_id}-{self.task}-{self.format.format_type}{params_str}".replace("/", "-")
    
    def model_dir(self):
        return os.path.join(os.path.abspath(self.base_dir), self.unique_name())
    
    def model_file_path(self):
        if(self.format.format_type == "onnx"):
            return os.path.join(self.model_dir(), "model.onnx")
        elif(self.format.format_type == "openvino"):
            return os.path.join(self.model_dir(), "model.xml") , os.path.join(self.model_dir(), "model.bin")
        elif(self.format.format_type == "trt"):
            return os.path.join(self.model_dir(), "model.plan")
        else: 
            raise Exception("Model format is not onnx")
        
    
    def param_str(self):
        format_params = '-'.join(sorted(map(str, self.format.parameters.values())))
        if self.format.origin:
            origin_params = '-'.join(sorted(map(str, self.format.origin.parameters.values())))
            format_params += origin_params
        return format_params
    
    def gpu_enabled(self):
        return self.format.gpu_enabled()
    
    def half(self):
        return self.format.half()
    
    def with_shapes(self, input_shape, output_shape):
        return self.__class__(self.hf_id, self.task, self.format, self.base_dir, input_shape, output_shape)
    
    def tags(self):
        return {
            'hf_id': str(self.hf_id),
            'task': str(self.task),
            'format': self.format.format_type,
            'gpu': str(self.gpu_enabled()),
            'half': str(self.half()),
        }


