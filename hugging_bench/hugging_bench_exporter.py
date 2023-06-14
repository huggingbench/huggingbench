import logging
import os

from hugging_bench.hugging_bench_config import ExperimentSpec, Format, ModelInfo
from hugging_bench.hugging_bench_util import PRINT_HEADER, hf_model_input, hf_model_output, run_docker_sdk

LOG = logging.getLogger(__name__)


class ModelExporter:
    # export model to onnx, openvino, trt...

    def __init__(self, hf_id, spec: ExperimentSpec, task=None, base_dir=None) -> None:
        self.hf_id = hf_id
        self.spec = spec
        self.task = task
        self.base_dir = os.path.abspath(base_dir) if (base_dir) else os.getcwd()
        self.cache_dir = os.path.join(self.base_dir, "_optimum_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def export(self, model_input_path: str = None) -> ModelInfo:
        #  onnx format is a starting point
        onnx_model_info = self._export_hf2onnx("0.001", model_input_path)
        inputs = hf_model_input(
            onnx_model_info.model_file_path(),
            half=onnx_model_info.half(),
            custom_shape_map={"batch_size": self.spec.batch_size},
        )
        outputs = hf_model_output(
            onnx_model_info.model_file_path(),
            half=onnx_model_info.half(),
            custom_shape_map={"batch_size": self.spec.batch_size},
        )
        onnx_model_info = onnx_model_info.with_shapes(input_shape=inputs, output_shape=outputs)

        if self.spec.format == "onnx":
            return onnx_model_info
        elif self.spec.format == "openvino":
            return self._export_onnx2openvino(onnx_model_info)
        elif self.spec.format == "trt":
            return self._export_onnx2trt(onnx_model_info)
        else:
            raise Exception(f"Unknown format {self.spec.format}")

    def _export_hf2onnx(self, atol=0.001, model_input: str = None) -> ModelInfo:
        LOG.info(PRINT_HEADER % " ONNX EXPORT ")

        onnx_model_info = ModelInfo(
            self.hf_id,
            self.task,
            Format(
                "onnx",
                {"atol": atol, "device": self.spec.device, "half": self.spec.half, "batch_size": self.spec.batch_size},
            ),
            base_dir=self.base_dir,
        )

        if os.path.exists(onnx_model_info.model_dir()):
            LOG.info(f"Model already exists at {onnx_model_info.model_file_path()}")
            return onnx_model_info

        model_dir = onnx_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        model_arg = f"--model={self.hf_id}" if model_input is None else f"--model=/model_input"

        cmd = [
            "optimum-cli",
            "export",
            "onnx",
            model_arg,
            "--framework=pt",
            f"--cache_dir={self.cache_dir}",
            f"--batch_size={self.spec.batch_size}",
            f"--sequence_length={self.spec.sequence_length}",
            "--monolit",
            f"--atol={atol}",
        ]

        if not half and device:
            cmd.append(f"--device={device}")

        if half:
            cmd.append("--fp16")
            cmd.append("--device=cuda")

        if self.task:
            cmd.append(f"--task={self.task}")

        cmd.append(onnx_model_info.model_dir())

        run_docker_sdk("optimum", model_dir, cmd, onnx_model_info.gpu_enabled(), model_input=model_input)

        return onnx_model_info

    def _export_onnx2openvino(self, onnx_model_info: ModelInfo):
        LOG.info(PRINT_HEADER % " ONNX 2 OPENVINO CONVERSION ")
        ov_model_info = ModelInfo(
            onnx_model_info.hf_id,
            onnx_model_info.task,
            format=Format("openvino", origin=onnx_model_info.format),
            base_dir=self.base_dir,
            input_shape=onnx_model_info.input_shape,
            output_shape=onnx_model_info.output_shape,
        )

        # this is kind of hack as current version of openvino in triton server does not suppot dynamic shape and shape can not take -1.
        custom_shape_map = {"sequence_length": self.spec.sequence_length}
        input_shape = hf_model_input(
            onnx_model_path=onnx_model_info.model_file_path(),
            half=onnx_model_info.half(),
            custom_shape_map=custom_shape_map,
        )
        output_shape = hf_model_output(
            onnx_model_path=onnx_model_info.model_file_path(),
            half=onnx_model_info.half(),
            custom_shape_map=custom_shape_map,
        )
        ov_model_info = ov_model_info.with_shapes(input_shape, output_shape)

        model_dir = ov_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)
        input_str = ",".join(
            [f"{input.name}{[self.spec.sequence_length] + input.dims}" for input in ov_model_info.input_shape]
        )
        cmd = [
            "mo",
            f"--input_model={onnx_model_info.model_file_path()}",
            f"--output_dir={model_dir}",
            f"--input={input_str}",
        ]
        run_docker_sdk(image_name="openvino", docker_args=cmd)
        return ov_model_info

    def _export_onnx2trt(self, onnx_model_info):
        LOG.info(PRINT_HEADER % " ONNX 2 TRT CONVERSION ")

        trt_onnx_model_info = ModelInfo(
            onnx_model_info.hf_id,
            onnx_model_info.task,
            Format("trt", origin=onnx_model_info.format),
            self.base_dir,
            input_shape=onnx_model_info.input_shape,
            output_shape=onnx_model_info.output_shape,
        )

        custom_shape_map = {"sequence_length": self.spec.sequence_length}
        input_shape = hf_model_input(
            onnx_model_path=onnx_model_info.model_file_path(),
            half=onnx_model_info.half(),
            int64to32=True,
            custom_shape_map=custom_shape_map,
        )
        output_shape = hf_model_output(
            onnx_model_path=onnx_model_info.model_file_path(),
            half=onnx_model_info.half(),
            custom_shape_map=custom_shape_map,
        )
        trt_onnx_model_info = trt_onnx_model_info.with_shapes(input_shape, output_shape)

        model_dir = trt_onnx_model_info.model_dir()
        os.makedirs(model_dir, exist_ok=True)

        def x(arr):
            shape = [self.spec.batch_size] + arr
            shape = [self.spec.sequence_length if x == -1 else x for x in shape]
            list_string = "x".join(str(x) for x in shape)
            return list_string

        max_shapes = ",".join([f"{input.name}:{x(input.dims)}" for input in trt_onnx_model_info.input_shape])

        cmd = [
            "trtexec",
            f"--shapes={max_shapes}",
            f"--onnx={onnx_model_info.model_file_path()}",
            f"--saveEngine={trt_onnx_model_info.model_file_path()}",
            "--skipInference",
            "--best",
        ]
        run_docker_sdk(image_name="nvcr.io/nvidia/tensorrt:23.04-py3", docker_args=cmd, gpu=True)
        return trt_onnx_model_info

    def _inspect_onnx(self, onnx_model_info: ModelInfo):
        LOG.info(PRINT_HEADER % " ONNX MODEL INSPECTION ")

        run_docker_sdk(
            image_name="polygraphy",
            docker_args=["polygraphy", "inspect", "model", f"{onnx_model_info.model_file_path()[0]}", "--mode=onnx"],
            env={"POLYGRAPHY_AUTOINSTALL_DEPS": 1},
        )
