import csv
from dataclasses import replace
from polygraphy.backend.onnx.util import get_input_metadata, get_output_metadata
import os
import os
import logging

from hugging_bench.hugging_bench_config import Format, ModelInfo, ExperimentSpec
from hugging_bench.hugging_bench_util import  hf_model_input, hf_model_output, run_docker_sdk, PRINT_HEADER


LOG = logging.getLogger(__name__)


class ModelExporter:
    # export model to onnx, openvino, trt...

    def __init__(self, hf_id, spec: ExperimentSpec, task=None, base_dir=None) -> None:
        self.hf_id = hf_id
        self.spec = spec
        self.task = task
        self.base_dir = base_dir if (base_dir) else os.getcwd()

    def export(self, model_input_path: str = None) -> ModelInfo:
        #  onnx format is a starting point
        model_info = self._export_hf2onnx(
            "0.001", self.spec.device, self.spec.half, model_input_path, self.spec.batch_size)
        model_info = model_info.with_shapes(
            input_shape=hf_model_input(
                model_info.model_file_path(), half=model_info.half()),
            output_shape=hf_model_output(model_info.model_file_path(), half=model_info.half()))

        if (self.spec.format == "onnx"):
            None
        elif (self.spec.format == "openvino"):
            model_info = self._export_onnx2openvino(model_info)
        elif (self.spec.format == "trt"):
            model_info = self._export_onnx2trt(model_info)
        else:
            raise Exception(f"Unknown format {self.spec.format}")

        LOG.info(f"Model info {model_info}")
        return model_info

    def _export_hf2onnx(self, atol=0.001, device=None, half=False, model_input: str = None, batch_size: int = 1) -> ModelInfo:
        print(PRINT_HEADER % " ONNX EXPORT ")
        model_info = ModelInfo(self.hf_id, self.task, Format(
            "onnx", {"atol": atol, "device": device, "half": half, "batch_size": batch_size}), base_dir=self.base_dir)

        if (all(os.path.exists(file) for file in model_info.model_file_path())):
            LOG.info(f"Model already exists at {model_info.model_file_path()}")
            return model_info

        model_dir = model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        model_arg = f"--model={self.hf_id}" if model_input is None else f"--model=/model_input"

        cmd = [
            "optimum-cli", "export", "onnx",
            model_arg,
            "--framework=pt",
            "--monolit",
            f"--atol={atol}"]

        if (not half and device):
            cmd.append(f"--device={device}")

        if (half):
            cmd.append("--fp16")
            cmd.append("--device=cuda")

        if (self.task):
            cmd.append(f"--task={self.task}")

        cmd.append(model_info.model_dir())

        run_docker_sdk("optimum", model_dir, cmd,
                       model_info.gpu_enabled(), model_input=model_input)

        return model_info

    def _export_onnx2openvino(self, onnx_model_info: ModelInfo):
        LOG.info(PRINT_HEADER % " ONNX 2 OPENVINO CONVERSION ")
        ov_model_info = ModelInfo(
            onnx_model_info.hf_id,
            onnx_model_info.task,
            format=Format("openvino", origin=onnx_model_info.format),
            base_dir=self.base_dir,
            input_shape=onnx_model_info.input_shape,
            output_shape=onnx_model_info.output_shape)
        model_dir = ov_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        cmd = [
            "mo",
            f"--input_model={onnx_model_info.model_file_path()[0]}",
            f"--output_dir={model_dir}"
        ]
        run_docker_sdk(image_name="openvino", docker_args=cmd)
        return ov_model_info

    def _export_onnx2trt(self, onnx_model_info):
        LOG.info(PRINT_HEADER % " ONNX 2 TRT CONVERSION ")
        trt_model_info = ModelInfo(
            onnx_model_info.hf_id,
            onnx_model_info.task,
            Format("trt", origin=onnx_model_info.format),
            self.base_dir,
            input_shape=onnx_model_info.input_shape,
            output_shape=onnx_model_info.output_shape)

        model_dir = trt_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        input_str = ' '.join(
            [f"{input.name}:{input.dims}" for input in trt_model_info.input_shape])

        cmd = [
            "polygraphy",
            "convert",
            "--model-type=onnx",
            "--convert-to=trt",
            f"--input-shapes={input_str}",
            f"--output={trt_model_info.model_file_path()[0]}",
            onnx_model_info.model_file_path()[0]
        ]
        run_docker_sdk(
            image_name="nvcr.io/nvidia/tensorrt:23.04-py3", docker_args=cmd, gpu=True)
        return trt_model_info

    def _inspect_onnx(self, model_info: ModelInfo):
        LOG.info(PRINT_HEADER % " ONNX MODEL INSPECTION ")
        run_docker_sdk(image_name="nvcr.io/nvidia/tensorrt:23.04-py3", docker_args=[
                       "polygraphy", "inspect", "model", f"{model_info.model_file_path()[0]}", "--mode=onnx"], env={"POLYGRAPHY_AUTOINSTALL_DEPS": 1})

