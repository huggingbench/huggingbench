from enum import Enum
import os
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType


class Format(NamedTuple):
    format_type: str # onnx, openvino, torchscript, tensorflow
    parameters: dict = {}
    origin: 'Format' = None



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
    

    def with_shapes(self, input_shape, output_shape):
        return self.__class__(self.hf_id, self.task, self.format, self.base_dir, input_shape, output_shape)

