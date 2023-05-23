from enum import Enum
import os
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType


class Format(NamedTuple):
    format_type: str # onnx, openvino, torchscript, tensorflow
    parameters: dict = {}
    origin: Optional['Format'] = None


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
        return f"{self.hf_id}-{self.task}-{self.format.format_type}-{self.param_str()}".replace("/", "-")
    
    def model_dir(self):
        return os.path.join(self.base_dir, self.unique_name())
    
    def model_file_path(self):
        if(self.format.format_type == "onnx"):
            return os.path.join(self.model_dir(), "model.onnx")
        elif(self.format.format_type == "openvino"):
            return os.path.join(self.model_dir(), "model.xml")
        else: 
            raise Exception("Model format is not onnx")
        
    
    def param_str(self):
        format_params = '-'.join(sorted(map(str, self.format.parameters.values())))
        if self.format.origin:
            origin_params = '-'.join(sorted(map(str, self.format.origin.parameters.values())))
            format_params += '-' + origin_params
        return format_params
    

    def with_shapes(self, input_shape, output_shape):
        return self.__class__(self.hf_id, self.task, self.format, self.base_dir, input_shape, output_shape)


    



# # test
# original_format = Format("onnx", {"hardware": "gpu", "half": True, "atol": 0.001})
# print(original_format)  # None

# new_format = Format("openvino", origin=original_format)
# print(new_format)  # None

# print(new_format.parameters.values())

# model = ModelInfo("distilbert-base-uncased", "fill-mask", new_format, "./triton-server/model_repository")
# print(model.unique_name())  # distilbert-base-uncased-fill-mask-onnx-{'hardware': 'gpu', 'half': True, 'atol': 0.001}

# model = ModelInfo("distilbert-base-uncased", "fill-mask", original_format, "./triton-server/model_repository")
# print(model.unique_name()) 