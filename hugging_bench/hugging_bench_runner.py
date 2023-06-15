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
from datetime import datetime

LOG = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        hf_id: str,
        experiments: list[ExperimentSpec],
        server_spec: TritonServerSpec,
        dataset: DatasetAlias = None,
        task=None,
        model_local_path: str = None,
        experiment_id: str = None,
    ) -> None:
        self.hf_id = hf_id
        self.task = task
        self.dataset = dataset
        self.output = f"{TEMP_DIR}/" + self.hf_id.replace("/", "-") + ".csv"
        self.experiments = experiments
        self.model_local_path = model_local_path

        self.server_spec = server_spec
        self.server_spec.model_repository_dir = os.path.abspath(server_spec.model_repository_dir)

        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        self.experiment_id = experiment_id if experiment_id else formatted_timestamp

    def run(self):
        for spec in self.experiments:
            exporter = ModelExporter(self.hf_id, spec, self.task, TEMP_DIR)
            model_info = exporter.export(self.model_local_path)
            triton_config = TritonConfig(self.server_spec, model_info).create_model_repo(spec.batch_size)
            triton_server = TritonServer(triton_config)
            triton_server.start()
            triton_client = TritonClient("localhost:{}".format(self.server_spec.http_port), model_info.unique_name(), max_paralell_requests=spec.client_workers)
            runner_config = RunnerConfig(batch_size=spec.batch_size, workers=spec.client_workers)
            client_runner = Runner(runner_config, triton_client, self._dataset_or_default(triton_client.inputs))
            success = False
            try:
                exec_times, success_rate, failure_rate, total, success_count = client_runner.run()
                LOG.info(
                    f"Total requests: {total} Success count: {success_count} Success rate: {success_rate} Failure rate: {failure_rate}"
                )
                success = True
            except Exception as e:
                LOG.error(f"Client load generation: {e}", exc_info=True)
            finally:
                triton_server.stop()
            if success:
                self.process_results(spec, exec_times, success_rate)

    def process_results(self, spec: ExperimentSpec, exec_times: list[float], success_rate: float):
        # Calculate percentiles and append to csv
        exec_times = np.array(exec_times)
        median = np.median(exec_times)
        percentile_90 = np.percentile(exec_times, 90)
        percentile_99 = np.percentile(exec_times, 99)
        res_dict = {
            "success_rate": success_rate,
            "median": median,
            "90_percentile": percentile_90,
            "99_percentile": percentile_99,
            "experiment_id": self.experiment_id,
        }
        append_to_csv(vars(spec), res_dict, self.output)

    def _dataset_or_default(self, input_metadata):
        if self.dataset:
            return self.dataset
        else:
            inputs = [
                Input(name=i["name"], dtype=i["datatype"], dims=[100 if s == -1 else s for s in i["shape"]][1:])
                for i in input_metadata.values()
            ]
            return DatasetGen(inputs, size=1000).dataset
