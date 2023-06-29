import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from bench.config import ExperimentSpec, Input
from bench.plugin import Plugin
from client.base import DatasetAlias, DatasetGen
from client.dataset import get_dataset
from client.runner import Runner, RunnerConfig, RunnerStats
from bench.chart import ChartGen

LOG = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        plugin: Plugin,
        experiments: list[ExperimentSpec],
    ) -> None:
        self.plugin = plugin
        self.experiments = experiments
        self.chart_gen = ChartGen()

    def run(self):
        for spec in self.experiments:
            try:
                model = self.plugin.model(spec)
                server = self.plugin.server(spec, model)
                server.start()
                client = self.plugin.client(spec, model)
                runner_config = RunnerConfig(batch_size=spec.batch_size, workers=spec.client_workers)
                client_runner = Runner(
                    runner_config, client, self._dataset_or_random(spec.dataset_id, model.input_shape)
                )
                success = False
                stats = client_runner.run()
                success = True
            except Exception as e:
                LOG.error(f"Experiment {spec} has failed: {e}", exc_info=True)
            finally:
                server.stop()
            if success:
                df = self.process_results(spec, stats)
                self.chart_gen.add_data(df)
        print(tabulate(self.chart_gen.data, headers="keys", tablefmt="psql", showindex="never"))
        self.chart_gen.plot_charts(workspace_dir=self.experiments[0].workspace_dir)

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
        output_file = spec.get_csv_output_path(spec.workspace_dir)
        df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
        LOG.info(f"Results written to {output_file}")
        return df

    def _dataset_or_random(self, dataset_id: str, inputs: list[Input]) -> DatasetAlias:
        if dataset_id:
            return get_dataset(dataset_id)
        else:
            adjusted_inputs = [
                Input(name=i.name, dtype=i.dtype, dims=[100 if d == -1 else d for d in i.dims]) for i in inputs
            ]
            #     lambda i: Input(name=i.name, dtype=i.dtype, dims=[100 if s == -1 else s for s in i.dims][1:]), inputs
            # )

            #  for i in inputs
            # ]
            return DatasetGen(adjusted_inputs, size=1000).dataset
