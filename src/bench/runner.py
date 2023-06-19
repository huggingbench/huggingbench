import logging
import os
from datetime import datetime

import numpy as np

from client.base import DatasetAlias, DatasetGen
from client.runner import Runner, RunnerConfig
from client.triton_client import TritonClient
from bench.config import TEMP_DIR, ExperimentSpec, Input
from server.exporter import ModelExporter
from server.triton import TritonConfig, TritonServer, TritonServerSpec
from server.util import append_to_csv

LOG = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        experiments: list[ExperimentSpec],
        server_spec: TritonServerSpec,
        dataset: DatasetAlias = None,
    ) -> None:
        self.dataset = dataset
        self.experiments = experiments
        self.server_spec = server_spec
        self.server_spec.model_repository_dir = os.path.abspath(server_spec.model_repository_dir)

    def run(self):
        for spec in self.experiments:
            exporter = ModelExporter(spec.hf_id, spec, spec.task, TEMP_DIR)
            model_info = exporter.export(spec.model_local_path)
            triton_config = TritonConfig(self.server_spec, model_info, spec).create_model_repo(spec.batch_size)
            triton_server = TritonServer(triton_config)
            triton_server.start()
            triton_client = TritonClient(
                "localhost:{}".format(self.server_spec.http_port),
                model_info.unique_name(),
                max_paralell_requests=spec.client_workers,
            )
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
        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        experiment_id = formatted_timestamp
        res_dict = {
            "success_rate": success_rate,
            "median": median,
            "90_percentile": percentile_90,
            "99_percentile": percentile_99,
            "experiment_id": experiment_id,
        }
        output_file = spec.get_csv_output_path()
        append_to_csv(vars(spec), res_dict, output_file)

    def _dataset_or_default(self, input_metadata):
        if self.dataset:
            return self.dataset
        else:
            inputs = [
                Input(name=i["name"], dtype=i["datatype"], dims=[100 if s == -1 else s for s in i["shape"]][1:])
                for i in input_metadata.values()
            ]
            return DatasetGen(inputs, size=1000).dataset
