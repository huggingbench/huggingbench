from hugging_bench_util import ModelExporter, measure_execution_time
from hugging_bench_config import Input, Output
from hugging_bench_triton import TritonConfig, TritonServer

MODEL_REPO="/home/kiarash/kiarash_server/model_repository"
RESNET = "microsoft/resnet-50"
BERT="bert-base-uncased"



exporter = ModelExporter("microsoft/resnet-50")
onnx = exporter.export_hf2onnx()
exporter.inspect_onnx(onnx)
#  This step could potentially be automated at some point
onnx_with_shapes = onnx.with_shapes(
    input_shape=[Input(name="pixel_values", dtype="FP32", dims=[3, 224, 224])], 
    output_shape=[Output(name="logits", dtype="FP32", dims=[1000])])

config = TritonConfig("./kiarash_server/model_repository", onnx_with_shapes).create_model_repo()
server = TritonServer(config)
server.start()
client = server.test_client()

try:
    stats = measure_execution_time(client.infer_sample, 100)
    print(stats)
    server.stop()
except Exception as e:
    print(e)
    server.stop()
