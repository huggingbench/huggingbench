from hugging_bench_util import ModelExporter, measure_execution_time, append_to_csv
from hugging_bench_config import ExperimentSpec, TritonServerSpec, LoadGenerator
from hugging_bench_triton import TritonConfig, TritonServer, AnyModelTestClient
import os



class TestLoadGenerator(LoadGenerator):
    def init(self, target, model_name):
        self.client = AnyModelTestClient(target, model_name)

    def load(self):
        self.stats = measure_execution_time(self.client.infer_sample, 100)

    def summary(self):
        return self.stats


class ExperimentRunner:
    def __init__(self, hf_id, experiments: list[ExperimentSpec], server_spec: TritonServerSpec) -> None:
        self.hf_id = hf_id
        self.output = self.hf_id.replace("/", "-") + ".csv"
        self.experiments = experiments
        
        self.server_spec = server_spec
        self.server_spec.model_repository_dir = os.path.abspath(server_spec.model_repository_dir)
        
    
    def run(self):
        for spec in self.experiments:
            exporter = ModelExporter(self.hf_id, spec)
            model_info = exporter.export()
            triton_config = TritonConfig(self.server_spec, model_info).create_model_repo()
            triton_server = TritonServer(triton_config)
            spec.load_generator.init("localhost:8001", model_info.unique_name())
            try:
                triton_server.start()

                spec.load_generator.load()
                summary = spec.load_generator.summary()
                print(summary)
                # client = triton_server.test_client()      
                # stats = measure_execution_time(client.infer_sample, 100)
                append_to_csv(vars(spec), summary, self.output)
                
                spec
                triton_server.stop()
            except Exception as e:
                print(e)
                triton_server.stop()
    

experiments=[ 
    ExperimentSpec(format="onnx", device="cpu", half=False, load_generator=TestLoadGenerator()),
    # Spec(format="onnx", device="cuda", half=False), # this needs to be run on a GPU machine
    # Spec(format="onnx", device="cuda", half=True), # this needs to be run on a GPU machine
    # Spec(format="openvino", device="cpu", half=False), # this needs to be run on a intel cpu
]


server_spec = TritonServerSpec(model_repository_dir="./kiarash_server/model_repository")

ExperimentRunner("microsoft/resnet-50", experiments, server_spec).run()
ExperimentRunner("microsoft/resnet-50", experiments, server_spec).run()
# ExperimentRunner("bert-base-uncased", "./kiarash_server/model_repository", experiments).run()
# ExperimentRunner("distilbert-base-uncased", "./kiarash_server/model_repository", experiments).run()
