import pytest
from server.config import ModelInfo, Format, Input, Output


@pytest.fixture
def onnx_model_info():
    format_params = {"device": "cuda", "half": True}
    format = Format(format_type="onnx", parameters=format_params)
    input_shape = [Input(name="input", dtype="FP32", dims=[1, 3, 224, 224])]
    output_shape = [Output(name="output", dtype="FP32", dims=[1, 1000])]
    return ModelInfo(
        hf_id="hf_model_id",
        task="classification",
        format=format,
        base_dir="/path/to/models",
        input_shape=input_shape,
        output_shape=output_shape,
    )


def test_unique_name(onnx_model_info):
    assert onnx_model_info.unique_name() == "hf_model_id-classification-onnx-True-cuda"


def test_model_dir(onnx_model_info):
    assert onnx_model_info.model_dir() == "/path/to/models/hf_model_id-classification-onnx-True-cuda"


def test_model_file_path(onnx_model_info):
    assert onnx_model_info.model_file_path() == "/path/to/models/hf_model_id-classification-onnx-True-cuda/model.onnx"


def test_gpu_enabled(onnx_model_info):
    assert onnx_model_info.gpu_enabled() == True


def test_half(onnx_model_info):
    assert onnx_model_info.half() == True


def test_with_shapes(onnx_model_info):
    input_shape = [Input(name="input", dtype="FP32", dims=[1, 3, 224, 224])]
    output_shape = [Output(name="output", dtype="FP32", dims=[1, 1000])]
    new_model_info = onnx_model_info.with_shapes(input_shape, output_shape)
    assert new_model_info.input_shape == input_shape
    assert new_model_info.output_shape == output_shape


@pytest.fixture
def trt_model_info():
    format_params = {"device": "cuda", "half": False}
    format_origin = Format(format_type="onnx", parameters=format_params)
    format_trt = Format(format_type="trt", origin=format_origin)
    input_shape = [Input(name="input", dtype="float32", dims=[1, 3, 224, 224])]
    output_shape = [Output(name="output", dtype="float32", dims=[1, 1000])]
    return ModelInfo(
        hf_id="model_id",
        task="classification",
        format=format_trt,
        base_dir="/path/to/model",
        input_shape=input_shape,
        output_shape=output_shape,
    )


def test_trt_unique_name(trt_model_info):
    assert trt_model_info.unique_name() == "model_id-classification-trt-False-cuda"


def test_trt_model_file_path(trt_model_info):
    assert trt_model_info.model_file_path() == "/path/to/model/model_id-classification-trt-False-cuda/model.plan"


def test_trt_gpu_enabled(trt_model_info):
    assert trt_model_info.gpu_enabled() == True


def test_trt_half(trt_model_info):
    assert trt_model_info.half() == False


def test_trt_with_shapes(trt_model_info):
    input_shape = [Input(name="input", dtype="float32", dims=[1, 3, 224, 224])]
    output_shape = [Output(name="output", dtype="float32", dims=[1, 1000])]
    new_model_info = trt_model_info.with_shapes(input_shape, output_shape)
    assert new_model_info.input_shape == input_shape
    assert new_model_info.output_shape == output_shape
