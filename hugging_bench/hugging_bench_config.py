
from dataclasses import dataclass, field
from typing import NamedTuple
# from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
import os


@dataclass
class TritonServerSpec:
    # TODO ports not used yet
    grpc_port: int = 8001
    http_port: int = 8000
    model_repository_dir: str = "./model_repository"


class LoadGenerator:
    def init(self, target, model_name):
        pass
    
    def load(self):
        pass

    def close(self):
        pass
    
    def summary(self):
        return {}
    
    def __str__(self):
        return self.__class__.__name__


@dataclass
class ExperimentSpec:
    format: str
    device: str
    half: bool
    load_generator: LoadGenerator


class Format(NamedTuple):
    format_type: str # onnx, openvino, torchscript, tensorflow
    parameters: dict = {}
    origin: 'Format' = None

    def gpu_enabled(self):
        if self.parameters.get('device') == 'cuda' or (self.origin and self.origin.parameters.get('device') == 'cuda'):
            return True
        return False
    
    def half(self):
        if self.parameters.get('half', False) or (self.origin and self.origin.parameters.get('half', False)):
            return True
        return False


class Input(NamedTuple):
    name: str
    dtype: str
    dims: list[int]
    

class Output(NamedTuple):
    name: str
    dtype: str
    dims: list[int]
 

class ModelInfo(NamedTuple):
    hf_id: str
    task: str
    format: Format
    base_dir: str
    input_shape: list[Input] = field(init=False)
    output_shape: list[Output] = field(init=False)

    def unique_name(self):
        params_str = f"-{self.param_str()}" if(self.param_str()) else ""
        return f"{self.hf_id}-{self.task}-{self.format.format_type}{params_str}".replace("/", "-")
    
    def model_dir(self):
        return os.path.join(os.path.abspath(self.base_dir), self.unique_name())
    
    def model_file_path(self):
        if(self.format.format_type == "onnx"):
            return [os.path.join(self.model_dir(), "model.onnx")]
        elif(self.format.format_type == "openvino"):
            return [os.path.join(self.model_dir(), "model.xml"), os.path.join(self.model_dir(), "model.bin")]
        else: 
            raise Exception("Model format is not onnx")
        
    
    def param_str(self):
        format_params = '-'.join(sorted(map(str, self.format.parameters.values())))
        # if self.format.origin:
            # origin_params = '-'.join(sorted(map(str, self.format.origin.parameters.values())))
            # format_params += '-' + origin_params
        return format_params
    
    def gpu_enabled(self):
        return self.format.gpu_enabled()
    
    def half(self):
        return self.format.half()
    

    def with_shapes(self, input_shape, output_shape):
        return self.__class__(self.hf_id, self.task, self.format, self.base_dir, input_shape, output_shape)



def test_gpu():
    format_with_gpu = Format(format_type='openvino', origin=Format(format_type='onnx', parameters={'device': 'cpu'}))
    # Create a ModelInfo instance with the above format
    model_info_with_gpu = ModelInfo(
        hf_id='some_id', 
        task='some_task',
        format=format_with_gpu,
        base_dir='some_dir'
    )
    print(model_info_with_gpu.gpu_enabled()) # True