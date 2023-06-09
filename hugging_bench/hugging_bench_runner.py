import numpy as np
from client.base import DatasetAlias, DatasetGen
from hugging_bench.hugging_bench_util import append_to_csv
from hugging_bench.hugging_bench_exporter import ModelExporter
from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec, Input, TEMP_DIR
from hugging_bench.hugging_bench_triton import TritonConfig, TritonServer
from client.triton_client import TritonClient
from client.runner import RunnerConfig, Runner
import logging
import os

LOG = logging.getLogger(__name__)

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
            triton_config = TritonConfig(self.server_spec, model_info).create_model_repo(spec.batch_size)
            triton_server = TritonServer(triton_config)
            triton_server.start()
            triton_client = TritonClient("localhost:{}".format(self.server_spec.http_port), model_info.unique_name())
            runner_config = RunnerConfig(batch_size=spec.batch_size, workers=spec.client_workers)
            client_runner = Runner(runner_config, triton_client, self._dataset_or_default(triton_client.inputs))
            success = False
            try:
                exec_times = client_runner.run()
                success = True
            except Exception as e:
                LOG.error(f"Client load generation: {e}", exc_info=True)
            finally:
                triton_server.stop()
            if success:
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
            return DatasetGen(inputs, size=500).dataset

    




