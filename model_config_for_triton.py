# Models to bench from HF:
# bert-base-uncased
# distilbert-base-uncased
# resnet50 from HF (microsoft/resnet-50)
import os
import os
import subprocess
# from model_config_constants import *

ONNX_BACKEND = "onnxruntime_onnx"
TORCH_BACKEND = "pytorch_libtorch"
OPENVINO_BACKEND = "openvino"

TYPE_FP32 = "TYPE_FP32"
TYPE_INT64 = "TYPE_INT64"

PRINT_HEADER = "\n\n============================%s=====================================\n"

def docker_base_cmd(docker_image, workspace):
    return ["docker", "run", "--rm", "-it", "-v", f"{workspace}:/{workspace}", docker_image]


def get_model_name(model_id, postfix):
    return model_id.replace("/", "-") + f"-{postfix}"


def get_model_dir(model_repo, model_id, postfix):
    model_repo_abspath = os.path.abspath(model_repo)
    return os.path.join(model_repo_abspath, get_model_name(model_id, postfix))

def convert_onnx2openvino_docker(input_model, model_repo, model_id):
    print(PRINT_HEADER % " ONNX 2 OPENVINO CONVERSION ")
    cmd = docker_base_cmd("openvino", os.path.abspath(model_repo))
    
    model_dir = get_model_dir(model_repo, model_id, "ov")
    model_version_dir = os.path.join(model_dir, "1")
    os.makedirs(model_version_dir, exist_ok=True)
    
    cmd.extend([
        "mo",
        f"--input_model={input_model}",
        f"--output_dir={model_version_dir}"
    ])
    print(f"Command {cmd}")
    print(f"OpenVino model being exported to {model_version_dir} ...")
    subprocess.run(cmd, stderr=subprocess.STDOUT)
    return model_dir
    

def export_hf_onnx_optimum_docker(model_id, model_repo, task=None, atol=0.001, device=None, half=False):
    print(PRINT_HEADER % " ONNX EXPORT ")
    try:
        model_dir = get_model_dir(model_repo, model_id, "onnx")
        model_version_dir = os.path.join(model_dir, "1")
        cmd = docker_base_cmd("optimum", model_version_dir)

        cmd.extend([
            "optimum-cli", "export", "onnx",
            f"--model={model_id}", 
            "--framework=pt",
            "--monolit", 
            f"--atol={atol}"])
        
        if(half):
            cmd.append("--f16")
            cmd.append("--device=cuda")

        if(task):
            cmd.append(f"--task={task}")
        
        if(not half and device):
            cmd.append(f"--device={device}")

        cmd.append(model_version_dir)

        print(f"Onnx model being exported to {model_version_dir} ...")
        subprocess.run(cmd, stderr=subprocess.STDOUT)
        
        return (model_dir, os.path.join(model_version_dir, "model.onnx"))
    except subprocess.CalledProcessError as e:
        # Error occurred while executing the command
        print(f"Command execution failed with error: {e.output}")
        raise e
   

def export_hf_onnx_optimum(model_id, model_repo, task=None, atol=0.001, device=None, half=False):
    print(PRINT_HEADER % " ONNX EXPORT ")
    try:
        model_dir = get_model_dir(model_repo, model_id, "onnx")
        model_version_dir = os.path.join(model_dir, "1")
        cmd = [
            "optimum-cli", "export", "onnx",
            f"--model={model_id}", 
            "--framework=pt",
            "--monolit", 
            f"--atol={atol}"]
        
        if(half):
            cmd.append("--f16")
            cmd.append("--device=cuda")

        if(task):
            cmd.append(f"--task={task}")
        
        if(not half and device):
            cmd.append(f"--device={device}")

        cmd.append(model_version_dir)

        print(f"Onnx model being exported to {model_version_dir} ...")
        subprocess.run(cmd, stderr=subprocess.STDOUT)

        return model_dir
    except subprocess.CalledProcessError as e:
        # Error occurred while executing the command
        print(f"Command execution failed with error: {e.output}")
        raise e
    
def export_hf_onnx(model_id, atol, model_repo):
    print(PRINT_HEADER % " ONNX EXPORT ")
    try:
        model_dir = get_model_dir(model_repo, model_id, "onnx")
        model_version_dir = os.path.join(model_dir, "1")
        print(f"Onnx model being exported to {model_version_dir} ..")
        subprocess.run(["python", "-m", "transformers.onnx", f"--model={model_id}", f"--atol={atol}", model_version_dir], stderr=subprocess.STDOUT)
        
        return model_dir
    except subprocess.CalledProcessError as e:
        # Error occurred while executing the command
        print(f"Command execution failed with error: {e.output}")
        raise e


def inspect_onnx(model_dir):
    print(PRINT_HEADER % " MODEL INSPECTION ")
    model_path = os.path.join(model_dir, "1", "model.onnx")
    output = subprocess.run(["polygraphy", "inspect", "model", f"{model_path}", "--mode=onnx"])
    print(f"Onnx model inspection: {output}")


def write_config_file(config_body, model_dir):
    print(PRINT_HEADER % " MODEL CONFIG CREATION ")
    config_path = os.path.join(model_dir, "config.pbtxt")
    try:
        with open(config_path, 'w') as file:
            file.write(config_body)
        print(f"Config written to {config_path} \n\n{config_body}\n")
    except Exception as e:
        print(f"Error occurred while writing to file: {e}")
        raise e


def resnetExample():
    from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
    from google.protobuf import text_format
    model_dir, _ = export_hf_onnx(model_id="microsoft/resnet-50", model_repo="./triton-server/model_repository", atol=0.001)
    # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
    inspect_onnx(model_dir)
    conf = ModelConfig(
        name = get_model_name("microsoft/resnet-50", "onnx"),
        max_batch_size = 1,
        platform = ONNX_BACKEND,
        input = [ModelInput(name="pixel_values", data_type=DataType.TYPE_FP32, dims=[-1, -1, -1])],
        output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[2048, 2, 2])],
        )
    txt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
    write_config_file(txt,  
                    model_dir)

def bertBaseUncasedExample():
    from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
    from google.protobuf import text_format

    model_dir, _ = export_hf_onnx(model_id="bert-base-uncased", model_repo="./triton-server/model_repository", atol=0.001)
    # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
    inspect_onnx(model_dir)
    conf = ModelConfig(
        name = get_model_name("bert-base-uncased", "onnx"),
        max_batch_size = 1,
        platform = ONNX_BACKEND,
        input = [
            ModelInput(name="input_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
            ModelInput(name="attention_mask", data_type=DataType.TYPE_INT64, dims=[-1]),
            ModelInput(name="token_type_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
            ],
        output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[-1, 768])],
        )
    txt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
    write_config_file(txt,  
                    model_dir)
    
def distilBertBaseUncasedExample():
    from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
    from google.protobuf import text_format

    model_dir, _ = export_hf_onnx(model_id="distilbert-base-uncased",  model_repo="./triton-server/model_repository", atol=0.001)
    # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
    inspect_onnx(model_dir)
    conf = ModelConfig(
        name = get_model_name("distilbert-base-uncased", "onnx"),
        max_batch_size = 1,
        platform = ONNX_BACKEND,
        input = [
            ModelInput(name="input_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
            ModelInput(name="attention_mask", data_type=DataType.TYPE_INT64, dims=[-1]),
            ],
        output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[-1, 768])],
        )
    txt = text_format.MessageToString(conf, use_short_repeated_primitives=True)
    write_config_file(txt,  
                    model_dir)


def distilBertBaseUncasedOnnxOpenVinoExample():
    from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
    from google.protobuf import text_format

    onnx_model_dir, onnx_model_path = export_hf_onnx_optimum_docker(model_id="distilbert-base-uncased",  model_repo="./triton-server/model_repository", atol=0.001)
    ov_model_dir = convert_onnx2openvino_docker(input_model=onnx_model_path, model_repo="./triton-server/model_repository", model_id="distilbert-base-uncased")
    # model_dir = get_model_dir("./triton-server/model_repository", "microsoft/resnet-50", "onnx")
    inspect_onnx(onnx_model_dir)
    onnx_conf = ModelConfig(
        name = get_model_name("distilbert-base-uncased", "onnx"),
        max_batch_size = 1,
        platform = ONNX_BACKEND,
        input = [
            ModelInput(name="input_ids", data_type=DataType.TYPE_INT64, dims=[-1]),
            ModelInput(name="attention_mask", data_type=DataType.TYPE_INT64, dims=[-1]),
            ],
        output = [ModelOutput(name="last_hidden_state", data_type=DataType.TYPE_FP32, dims=[-1, 768])],
        )
    txt = text_format.MessageToString(onnx_conf, use_short_repeated_primitives=True)
    write_config_file(txt,  
                    onnx_model_dir)  

    onnx_conf.platform = OPENVINO_BACKEND
    ov_txt = text_format.MessageToString(onnx_conf, use_short_repeated_primitives=True)
    write_config_file(ov_txt,  
                    ov_model_dir) 

distilBertBaseUncasedOnnxOpenVinoExample()
# model_dir, model_path = export_hf_onnx_optimum_docker(
#     model_id="distilbert-base-uncased",
#     model_repo="triton-server/model_repository",
#     atol=0.001)    

# model_dir = convert_onnx2openvino_docker(
#     input_model=model_path,
#     model_repo="triton-server/model_repository",
#     model_id="distilbert-base-uncased")

# resnetExample()
# bertBaseUncasedExample()
# distilBertBaseUncasedExample()

# export_hf_onnx_optimum("distilbert-base-uncased", 0.001, "_temp")
# activate_openvinto_env()
# deactivate_openvinto_env()