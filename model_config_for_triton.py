# Models to bench from HF:
# bert-base-uncased
# distilbert-base-uncased
# resnet50 from HF (microsoft/resnet-50)
import os
import os
import subprocess
from model_config_constants import *
from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
from google.protobuf import text_format
import fire

PRINT_HEADER = "\n\n============================%s=====================================\n"

def get_model_name(model_id, postfix):
    return model_id.replace("/", "-") + f"-{postfix}"


def get_model_dir(model_repo, model_id, postfix):
    return os.path.join(model_repo, get_model_name(model_id, postfix))


def export_hf_onnx(model_id, atol, model_repo):
    print(PRINT_HEADER % " ONNX EXPORT ")
    try:
        model_dir = get_model_dir(model_repo, model_id, "onnx")
        model_version_dir = os.path.join(model_dir, "1")
        subprocess.run(["python", "-m", "transformers.onnx", f"--model={model_id}", f"--atol={atol}", model_version_dir], stderr=subprocess.STDOUT)
        print(f"Onnx model successfully exported to {model_version_dir}")
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
    model_dir = export_hf_onnx("microsoft/resnet-50", 0.001, "./triton-server/model_repository")
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

def bertbaseuncasedExample():
    model_dir = export_hf_onnx("bert-base-uncased", 0.001, "./triton-server/model_repository")
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
    
resnetExample()
# bertbaseuncasedExample()