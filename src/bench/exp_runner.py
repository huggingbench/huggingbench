import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from bench.config import ExperimentSpec, Input
from client.base import DatasetAlias, DatasetGen
from client.runner import Runner, RunnerConfig, RunnerStats
from client.triton_client import TritonClient
from server.exporter import ModelExporter
from server.triton import TritonConfig, TritonServer
from bench.chart import ChartGen

LOG = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        experiments: list[ExperimentSpec],
        dataset: DatasetAlias = None,
        workspace_dir: str = "./temp",
    ) -> None:
        self.dataset = dataset
        self.experiments = experiments
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.chart_gen = ChartGen(self.workspace_dir)

    def run(self):
        for spec in self.experiments:
            exporter = ModelExporter(spec.hf_id, spec, spec.task, self.workspace_dir)
            model_info = exporter.export(spec.model_local_path)
            triton_config = TritonConfig(model_info, spec, self.workspace_dir).create_model_repo(spec.batch_size)
            triton_server = TritonServer(triton_config)
            triton_server.start()
            triton_client = TritonClient(
                "localhost:{}".format(triton_config.http_port),
                model_info.unique_name(),
                max_paralell_requests=spec.client_workers,
            )
            runner_config = RunnerConfig(batch_size=spec.batch_size, workers=spec.client_workers)
            client_runner = Runner(runner_config, triton_client, self._dataset_or_default(triton_client.inputs))
            success = False
            try:
                stats = client_runner.run()
                success = True
            except Exception as e:
                LOG.error(f"Client load generation: {e}", exc_info=True)
            finally:
                triton_server.stop()
            if success:
                df = self.process_results(spec, stats)
                self.chart_gen.plot_charts(spec.hf_id, df)

    def process_results(self, spec: ExperimentSpec, stats: RunnerStats) -> pd.DataFrame:
        # Calculate percentiles and append to csv
        exec_times = np.array(stats.execution_times)
        median = np.median(exec_times)
        avg = np.average(exec_times)
        percentile_90 = np.percentile(exec_times, 90)
        percentile_99 = np.percentile(exec_times, 99)
        res_dict = {
            "success_rate": stats.success_rate,
            "avg": avg,
            "median": median,
            "90_percentile": percentile_90,
            "99_percentile": percentile_99,
        }
        info = vars(spec)
        data = {**info, **res_dict}
        df = pd.DataFrame(data, index=[0])
        output_file = spec.get_csv_output_path(self.workspace_dir)
        print(tabulate(df, headers="keys", tablefmt="psql", showindex="never"))
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        LOG.info(f"Results written to {output_file}")
        return df

    def _dataset_or_default(self, input_metadata):
        if self.dataset:
            return self.dataset
        else:
            inputs = [
                Input(name=i["name"], dtype=i["datatype"], dims=[100 if s == -1 else s for s in i["shape"]][1:])
                for i in input_metadata.values()
            ]
            return DatasetGen(inputs, size=1000).dataset
