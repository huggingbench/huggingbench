# from tritonclient.grpc.model_config_pb2 import ModelConfig, ModelInput, ModelOutput, DataType
import os
from dataclasses import dataclass, field

TEMP_DIR = "./temp"
TEMP_MODEL_REPO_DIR = f"{TEMP_DIR}/model_repository"


@dataclass
class ExperimentSpec:
    hf_id: str
    format: str
    device: str
    task: str = None
    batch_size: int = 1
    sequence_length: int = 100
    client_workers: int = 1
    instance_count: int = 1
    model_local_path: str = None
    dataset_id: str = "random"  # if not set we generate random data
    precision: str = "fp32"  # allowed values are fp32, fp16, int8
    extra_params: dict = field(default_factory=dict)  # extra parameters to pass to the model
    workspace_dir: str = TEMP_DIR

    def is_valid(self):
        if self.format == "onnx" and self.device == "cpu" and self.precision == "fp16":
            return False

        if self.format == "openvino" and self.device == "gpu":
            return False

        if self.format == "trt" and self.device == "cpu":
            return False

        return True

    def metric_tags(self):
        return {
            "hf_id": self.hf_id,
            "task": self.task if self.task else "",
            "format": self.format,
            "device": self.device,
            "precision": self.precision,
            "batch_size": str(self.batch_size),
            "sequence_length": str(self.sequence_length),
            "client_workers": str(self.client_workers),
            "dataset_id": self.dataset_id,
        }

    def get_csv_output_path(self, base_dir):
        return os.path.join(base_dir, get_os_friendly_path(self.hf_id) + ".csv")


@dataclass
class Format:
    format_type: str  # onnx, openvino, torchscript, tensorflow
    parameters: dict = field(default_factory=dict)
    origin: "Format" = None

    def gpu_enabled(self):
        if self.parameters.get("device") == "gpu" or (self.origin and self.origin.parameters.get("device") == "gpu"):
            return True
        return False

    def half(self):
        return self.parameters.get("precision", "fp32") == "fp16" or (
            self.origin and self.origin.parameters.get("precision", "fp32") == "fp16"
        )


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
    input_shape: list[Input] = field(default_factory=list)
    output_shape: list[Output] = field(default_factory=list)

    def unique_name(self):
        params_str = f"-{self.param_str()}" if (self.param_str()) else ""
        return f"{self.hf_id}-{self.task}-{self.format.format_type}{params_str}".replace("/", "-")

    def model_dir(self):
        return os.path.join(os.path.abspath(self.base_dir), self.unique_name())

    def model_file_path(self):
        if self.format.format_type == "onnx":
            return os.path.join(self.model_dir(), "model.onnx")
        elif self.format.format_type == "openvino":
            return os.path.join(self.model_dir(), "model.xml"), os.path.join(self.model_dir(), "model.bin")
        elif self.format.format_type == "trt":
            return os.path.join(self.model_dir(), "model.plan")
        else:
            raise Exception("Model format is not onnx")

    def param_str(self):
        format_params = "-".join(sorted(map(str, self.format.parameters.values())))
        if self.format.origin:
            origin_params = "-".join(sorted(map(str, self.format.origin.parameters.values())))
            format_params += origin_params
        return format_params

    def gpu_enabled(self):
        return self.format.gpu_enabled()

    def half(self):
        return self.format.half()

    def with_shapes(self, input_shape, output_shape):
        return self.__class__(self.hf_id, self.task, self.format, self.base_dir, input_shape, output_shape)


def get_os_friendly_path(hf_id: str):
    return hf_id.replace("/", "-")
