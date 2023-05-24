from hugging_bench_util import ModelExporter, measure_execution_time, hf_model_input, hf_model_output, append_to_csv
from hugging_bench_config import Input, Output
from hugging_bench_triton import TritonConfig, TritonServer
from typing import NamedTuple, Optional
import os
from functools import partial


class Spec(NamedTuple):
    format: str
    device: str
    half: bool
    

class ExperimentRunner:
    def __init__(self, hf_id, triton_model_repo, experiments: list[Spec]) -> None:
        self.hf_id = hf_id
        self.triton_model_repo = os.path.abspath(triton_model_repo)
        self.experiments = experiments
        self.output = self.hf_id.replace("/", "-") + ".csv"
    
    def run(self):
        for spec in self.experiments:
            exporter = ModelExporter(self.hf_id)
            model_info = exporter.export_hf2onnx(device=spec.device, half=spec.half)
            # exporter.inspect_onnx(model_info)

            #  openvino takes onnx as input so onnx export must run anyhow
            if(spec.format == "openvino"):
                model_info = exporter.export_onnx2openvino(model_info)

            model_info = model_info.with_shapes(
                input_shape=hf_model_input(self.hf_id), 
                output_shape=hf_model_output(self.hf_id))  
            
            config = TritonConfig(self.triton_model_repo, model_info).create_model_repo()
            server = TritonServer(config)
            try:
                server.start()
                client = server.test_client()      
                stats = measure_execution_time(client.infer_sample, 100)
                append_to_csv(spec, stats, self.output)
                print(stats)
                server.stop()
            except Exception as e:
                print(e)
                server.stop()


experiments=[ 
    # Spec(format="onnx", device="cpu", half=False),
    # Spec(format="onnx", device="cuda", half=False), # this needs to be run on a GPU machine
    # Spec(format="onnx", device="cuda", half=True), # this needs to be run on a GPU machine
    Spec(format="openvino", device="cpu", half=False), # this needs to be run on a intel cpu
]

ExperimentRunner("microsoft/resnet-50", "./kiarash_server/model_repository", experiments).run()

# ExperimentRunner("bert-base-uncased", "./kiarash_server/model_repository", experiments).run()

# ExperimentRunner("distilbert-base-uncased", "./kiarash_server/model_repository", experiments).run()