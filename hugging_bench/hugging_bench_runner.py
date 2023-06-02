import numpy as np
from client.base import DatasetAlias, DatasetGen
from hugging_bench.hugging_bench_util import ModelExporter, append_to_csv
from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec, Input, TEMP_DIR
from hugging_bench.hugging_bench_triton import TritonConfig, TritonServer
from client.triton_client import TritonClient
from client.runner import RunnerConfig, Runner
from client.datasets import get_dataset
import os

class ExperimentRunner:
    def __init__(self, hf_id: str, experiments: list[ExperimentSpec], server_spec: TritonServerSpec, dataset: DatasetAlias=None, task=None, model_local_path: str = None) -> None:
        self.hf_id = hf_id
        self.task = task
        self.dataset = dataset
        self.output = f"{TEMP_DIR}/" + self.hf_id.replace("/", "-") + ".csv"
        self.experiments = experiments
        self.model_local_path = model_local_path
        
        self.server_spec = server_spec
        self.server_spec.model_repository_dir = os.path.abspath(server_spec.model_repository_dir)
        
    
    def run(self):
        for spec in self.experiments:
            exporter = ModelExporter(self.hf_id, spec, self.task, TEMP_DIR)
            model_info = exporter.export(self.model_local_path)
            triton_config = TritonConfig(self.server_spec, model_info).create_model_repo()
            triton_server = TritonServer(triton_config)
            triton_server.start()
            triton_client = TritonClient("localhost:{}".format(server_spec.http_port), model_info.unique_name())
            runner_config = RunnerConfig()
            client_runner = Runner(runner_config, triton_client, self._dataset_or_default(triton_client.inputs))
            try:
                exec_times = client_runner.run()
            except Exception as e:
                print(e)
            finally:
                triton_server.stop()
            self.process_results(spec, exec_times)

    def process_results(self, spec: ExperimentSpec, exec_times: list[float]):
        # Calculate percentiles and append to csv
        exec_times = np.array(exec_times)
        median = np.median(exec_times)
        percentile_90 = np.percentile(exec_times, 90)
        percentile_99 = np.percentile(exec_times, 99)
        res_dict = {'median': median, '90_percentile': percentile_90, '99_percentile': percentile_99}
        append_to_csv(vars(spec), res_dict, self.output)

    def _dataset_or_default(self, input_metadata):
        if(self.dataset): 
            return self.dataset
        else:
            inputs = [Input(name=i['name'], dtype=i['datatype'], dims=[100 if s==-1 else s for s in i['shape']][1:]) for i in input_metadata.values()]
            return DatasetGen(inputs).dataset

    

experiments=[ 
    # ExperimentSpec(format="trt", device="cuda", half=True),
    # ExperimentSpec(format="trt", device="cuda", half=False),
# 
    # ExperimentSpec(format="onnx", device="cuda", half=True),   
    # ExperimentSpec(format="onnx", device="cuda", half=False),
    ExperimentSpec(format="onnx", device="cpu", half=False),
]

server_spec = TritonServerSpec()

# cover all models with random data

#ExperimentRunner("microsoft/resnet-50", experiments, server_spec, dataset=None, model_local_path="/Users/niksa/projects/models/resnet-50").run()
# ExperimentRunner("bert-base-uncased", experiments, server_spec, dataset=None, model_local_path="/Users/niksa/.cache/huggingface/hub/models--bert-base-uncased/snapshots/0a6aa9128b6194f4f3c4db429b6cb4891cdb421b", task="text-classification").run()

# ExperimentRunner("microsoft/resnet-50", experiments, server_spec, dataset=None).run()
# ExperimentRunner("bert-base-uncased", experiments, server_spec, dataset=None).run()

# ExperimentRunner("distilbert-base-uncased", experiments, server_spec, dataset=None).run()
# ExperimentRunner("facebook/bart-large", experiments, server_spec, dataset=None, task='feature-extraction').run()


# resnet50_gen_dataset = get_dataset("microsoft/resnet-50-gen")
# bert_gen_dataset = get_dataset("bert-base-uncased-gen")
# ExperimentRunner("microsoft/resnet-50", resnet50_gen_dataset, experiments, server_spec).run()
# ExperimentRunner("bert-base-uncased", bert_gen_dataset, experiments, server_spec).run()

# ExperimentRunner("xlm-roberta-base", experiments, server_spec, dataset=None).run()

# ExperimentRunner("bert-base-uncased", experiments, server_spec).run()
# ExperimentRunner("distilbert-base-uncased", experiments, server_spec).run()
# ExperimentRunner("microsoft/resnet-50", experiments, server_spec).run()
# ExperimentRunner("bert-base-uncased", "./kiarash_server/model_repository", experiments).run()
# ExperimentRunner("distilbert-base-uncased", "./kiarash_server/model_repository", experiments).run()
