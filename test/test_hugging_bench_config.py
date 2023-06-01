import pytest
from hugging_bench.hugging_bench_config import ModelInfo, Format, Input, Output

@pytest.fixture
def sample_model_info():
    format_params = {'device': 'cuda', 'half': True}
    format = Format(format_type='onnx', parameters=format_params)
    input_shape = [Input(name='input', dtype='FP32', dims=[1, 3, 224, 224])]
    output_shape = [Output(name='output', dtype='FP32', dims=[1, 1000])]
    return ModelInfo(hf_id='hf_model_id', task='classification', format=format,
                     base_dir='/path/to/models', input_shape=input_shape, output_shape=output_shape)

def test_unique_name(sample_model_info):
    assert sample_model_info.unique_name() == 'hf_model_id-classification-onnx-True-cuda'

def test_model_dir(sample_model_info):
    assert sample_model_info.model_dir() == '/path/to/models/hf_model_id-classification-onnx-True-cuda'

def test_model_file_path(sample_model_info):
    assert sample_model_info.model_file_path() == '/path/to/models/hf_model_id-classification-onnx-True-cuda/model.onnx'

def test_gpu_enabled(sample_model_info):
    assert sample_model_info.gpu_enabled() == True

def test_half(sample_model_info):
    assert sample_model_info.half() == True

def test_with_shapes(sample_model_info):
    input_shape = [Input(name='input', dtype='FP32', dims=[1, 3, 224, 224])]
    output_shape = [Output(name='output', dtype='FP32', dims=[1, 1000])]
    new_model_info = sample_model_info.with_shapes(input_shape, output_shape)
    assert new_model_info.input_shape == input_shape
    assert new_model_info.output_shape == output_shape

def test_tags(sample_model_info):
    tags = sample_model_info.tags()
    assert tags['hf_id'] == 'hf_model_id'
    assert tags['task'] == 'classification'
    assert tags['format'] == 'onnx'
    assert tags['gpu'] == 'True'
    assert tags['half'] == 'True'
