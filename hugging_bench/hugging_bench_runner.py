from client.base import DatasetAlias
from hugging_bench_util import ModelExporter, measure_execution_time, append_to_csv
from hugging_bench_config import ExperimentSpec, TritonServerSpec
from hugging_bench_triton import TritonConfig, TritonServer
from client.triton_client import TritonClient
from client.runner import RunnerConfig, Runner, get_dataset
import os

class ExperimentRunner:
    def __init__(self, hf_id: str, dataset: DatasetAlias, experiments: list[ExperimentSpec], server_spec: TritonServerSpec) -> None:
        self.hf_id = hf_id
        self.dataset = dataset
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
            triton_server.start()
            triton_client = TritonClient("localhost:{}".format(server_spec.http_port), model_info.unique_name())
            runner_config = RunnerConfig()
            client_runner = Runner(runner_config, triton_client, self.dataset)
            try:
                client_runner.run()
            except Exception as e:
                print(e)
            finally:
                triton_server.stop()
                triton_client.write_metrics('metrics.csv')
    

experiments=[ 
    ExperimentSpec(format="onnx", device="cpu", half=False),
    # ExperimentSpec(format="onnx", device="cuda", half=False, load_generator=FitAllModelLoadGenerator()),
    # ExperimentSpec(format="onnx", device="cuda", half=True, load_generator=FitAllModelLoadGenerator()),
    # ExperimentSpec(format="openvino", device="cpu", half=False, load_generator=FitAllModelLoadGenerator()),
    # Spec(format="onnx", device="cuda", half=False), # this needs to be run on a GPU machine
    # Spec(format="onnx", device="cuda", half=True), # this needs to be run on a GPU machine
    # Spec(format="openvino", device="cpu", half=False), # this needs to be run on a intel cpu
]


server_spec = TritonServerSpec(model_repository_dir="./kiarash_server/model_repository")

resnet50_gen_dataset = get_dataset("microsoft/resnet-50-gen")
ExperimentRunner("microsoft/resnet-50", resnet50_gen_dataset, experiments, server_spec).run()
# ExperimentRunner("bert-base-uncased", experiments, server_spec).run()
# ExperimentRunner("distilbert-base-uncased", experiments, server_spec).run()
# ExperimentRunner("microsoft/resnet-50", experiments, server_spec).run()
# ExperimentRunner("bert-base-uncased", "./kiarash_server/model_repository", experiments).run()
# ExperimentRunner("distilbert-base-uncased", "./kiarash_server/model_repository", experiments).run()
