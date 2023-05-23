import os
import os
import subprocess
from typing import Any
import shutil
from hugging_bench_config import Format, ModelInfo
# from model_config_constants import *

ONNX_BACKEND = "onnxruntime_onnx"
TORCH_BACKEND = "pytorch_libtorch"
OPENVINO_BACKEND = "openvino"
PRINT_HEADER = "\n\n============================%s=====================================\n"

class ModelExporter:
    # from util import just_export_hf_onnx_optimum_docker, convert_onnx2openvino_docker
    
    def __init__(self, hf_id, task=None, base_dir=None) -> None:
        self.hf_id = hf_id
        self.task = task
        self.base_dir = base_dir if(base_dir) else os.getcwd()
        

    def export_hf2onnx(self, atol=0.001, device=None, half=False):
        print(PRINT_HEADER % " ONNX EXPORT ")
        model_info = ModelInfo(self.hf_id, self.task, Format("onnx", {"atol": atol, "device": device, "half": half}), base_dir=self.base_dir)
        
        if(os.path.exists(model_info.model_file_path())): 
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

        if(half):
            cmd.append("--f16")
            cmd.append("--device=cuda")
        
        if(self.task):
            cmd.append(f"--task={self.task}")
        
        if(not half and device):
            cmd.append(f"--device={device}")
        
        cmd.append(model_info.model_dir())
        
        run_docker("optimum", model_dir, cmd)
        
        return model_info


    def export_onnx2openvino(self, onnx_model_info: ModelInfo):
        print(PRINT_HEADER % " ONNX 2 OPENVINO CONVERSION ")   
        ov_model_info = ModelInfo(onnx_model_info.hf_id, onnx_model_info.task, Format("openvino", origin=onnx_model_info), self.base_dir)
        model_dir = ov_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)
        
        cmd = [
            "mo",
            f"--input_model={onnx_model_info.model_file_path()}",
            f"--output_dir={model_dir}"
        ]
        run_docker(image_name="openvino", docker_args=cmd)
        return ov_model_info
    

    def inspect_onnx(self, model_info: ModelInfo):
        print(PRINT_HEADER % " ONNX MODEL INSPECTION ")
        run_docker(image_name="polygraphy", docker_args=["polygraphy", "inspect", "model", f"{model_info.model_file_path()}", "--mode=onnx"])
        

def dtype_np_type(dtype: str):
    from tritonclient.utils import triton_to_np_dtype
    from hugging_bench_triton import TritonConfig
    return triton_to_np_dtype(TritonConfig.DTYPE_MAP.get(dtype, None))
    

def run_docker(image_name, workspace=None, docker_args=[]):
    import shlex
    # Construct Docker command
    if(not workspace):
        workspace = os.getcwd()
    command = f'docker run -v {workspace}:{workspace} -w {workspace}  {image_name} {" ".join(docker_args)}'
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
    
    
# # ----------------------------------------examples-----------------------------------------------

# def resnetExample():
#     from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
#     from google.protobuf import text_format
#     model_dir, _ = export_hf_onnx_optimum_docker(model_id="microsoft/resnet-50", model_repo="./triton-server/model_repository", atol=0.001)
#     # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
#     inspect_onnx(model_dir)
#     conf = ModelConfig(
#         name = get_model_name("microsoft/resnet-50", "onnx"),
#         max_batch_size = 1,
#         platform = ONNX_BACKEND,
#         input = [ModelInput(name="pixel_values", data_type=DataType.TYPE_FP32, dims=[-1, -1, -1])],
#         output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[2048, 2, 2])],
#         )
#     txt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
#     write_config_file(txt,  
#                     model_dir)

# def bertBaseUncasedExample():
#     from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
#     from google.protobuf import text_format

#     model_dir, _ = export_hf_onnx_optimum_docker(model_id="bert-base-uncased", model_repo="./triton-server/model_repository", atol=0.001)
#     # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
#     inspect_onnx(model_dir)
#     conf = ModelConfig(
#         name = get_model_name("bert-base-uncased", "onnx"),
#         max_batch_size = 1,
#         platform = ONNX_BACKEND,
#         input = [
#             ModelInput(name="input_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ModelInput(name="attention_mask", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ModelInput(name="token_type_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ],
#         output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[-1, 768])],
#         )
#     txt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
#     write_config_file(txt,  
#                     model_dir)
    
# def distilBertBaseUncasedExample():
#     from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
#     from google.protobuf import text_format

#     model_dir, _ = export_hf_onnx_optimum_docker(model_id="distilbert-base-uncased",  model_repo="./triton-server/model_repository", atol=0.001)
#     # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
#     inspect_onnx(model_dir)
#     conf = ModelConfig(
#         name = get_model_name("distilbert-base-uncased", "onnx"),
#         max_batch_size = 1,
#         platform = ONNX_BACKEND,
#         input = [
#             ModelInput(name="input_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ModelInput(name="attention_mask", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ],
#         output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[-1, 768])],
#         )
#     txt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
#     write_config_file(txt,  
#                     model_dir)


# def distilBertBaseUncasedOnnxOpenVinoExample():
#     from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
#     from google.protobuf import text_format

#     onnx_model_dir, onnx_model_path = export_hf_onnx_optimum_docker(model_id="distilbert-base-uncased",  model_repo="./triton-server/model_repository", atol=0.001)
#     ov_model_dir = convert_onnx2openvino_docker(input_model=onnx_model_path, model_repo="./triton-server/model_repository", model_id="distilbert-base-uncased")
#     # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
#     inspect_onnx(onnx_model_dir)
#     onnx_conf = ModelConfig(
#         name = get_model_name("distilbert-base-uncased", "onnx"),
#         max_batch_size = 1,
#         platform = ONNX_BACKEND,
#         input = [
#             ModelInput(name="input_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ModelInput(name="attention_mask", data_type=DataType.TYPE_INT64, dims=[-1]),
#             ],
#         output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[-1, 768])],
#         )
#     txt = text_format.MessageToString(onnx_conf, use_short_repeated_primitives=True)
#     write_config_file(txt,  
#                     onnx_model_dir)  

#     onnx_conf.platform = OPENVINO_BACKEND
#     ov_txt = text_format.MessageToString(onnx_conf, use_short_repeated_primitives=True)
#     write_config_file(ov_txt,  
#                     ov_model_dir) 

# # distilBertBaseUncasedOnnxOpenVinoExample()

# # just_export_hf_onnx_optimum_docker("_temp")
