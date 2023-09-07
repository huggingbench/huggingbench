from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import logging
import os

import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import List

from bench.config import ExperimentSpec, Input
from bench.plugin import Plugin
from client.base import DatasetAlias, DatasetGen
from client.dataset import get_dataset
from client.runner import Runner, RunnerConfig, RunnerStats
from bench.chart import ChartGen

from multiprocessing import Process, Queue

LOG = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        plugin: Plugin,
    ) -> None:
        self.plugin = plugin
        self.chart_gen = ChartGen()

    def parallel_client_run(self, idx, spec, model, dataset, queue):
        """To utilize available CPU cores we run clients in parallel processes."""
        client = self.plugin.client(spec, model)
        runner_config = RunnerConfig(batch_size=spec.batch_size, workers=spec.clients, async_req=spec.async_req)
        client_runner = Runner(idx, runner_config, client, dataset)
        stats = client_runner.run()
        queue.put(stats)
        return

    def run(self, experiments: List[ExperimentSpec]):
        failed_exp = []
        LOG.info(f"Running {len(experiments)} experiments")
        multi_proc_queue = Queue()
        ctx = RunContext(0, "")
        dataset = None
        for spec in experiments:
            try:
                server = None
                model = self.plugin.model(spec)
                server = self.plugin.server(spec, model)
                server.start()
                if not dataset:
                    """We generate the dataset only once"""
                    dataset = self._dataset_or_random(spec.dataset, model.input_shape)
                processes = []
                for i in range(spec.clients):
                    p = Process(target=self.parallel_client_run, args=(i, spec, model, dataset, multi_proc_queue))
                    p.start()
                    LOG.info(f"Started parallel client {i}")
                    processes.append(p)
                success = False
                all_stats = []
                for i in range(spec.clients):
                    stats = multi_proc_queue.get()
                    all_stats.append(stats)
                for p in processes:
                    p.join()
                    p.close()
                LOG.info("All client parallel processes completed")
                if len(all_stats) == spec.clients:
                    success = True
                stats = RunnerStats.merge(all_stats)
            except Exception as e:
                LOG.error(f"Experiment {spec} has failed: {e}", exc_info=True)
                success = False
                failed_exp.append(spec)
            finally:
                if server:
                    server.stop()
            if success:
                ctx.update_max(stats.success_rate, model.unique_name())
                df = self.process_results(spec, stats, self.plugin.get_name())
                self.chart_gen.add_data(df)
        tabulate_cols = ChartGen.labels + ["avg", "median", "90_percentile"]
        if len(failed_exp) > 0:
            print(f"NOTE! Following experiments have failed:\n {failed_exp}")
        if self.chart_gen.data is not None:
            print(tabulate(self.chart_gen.data[tabulate_cols], headers="keys", tablefmt="psql", showindex="never"))
            out_dir = (
                experiments[0].workspace_dir + "/" + self.plugin.get_name()
            )  # all experiments have the same workspace_dir
            os.makedirs(out_dir, exist_ok=True)
            self.chart_gen.plot_charts(
                output_dir=out_dir, model_id=experiments[0].id
            )  # all experiments have the same id
            best_model_path = os.path.abspath(spec.workspace_dir + "/model_repository/" + ctx.model_unique_name)
            print(f"\nNOTE: The model and configuration with the highest throughput is stored in: '{best_model_path}'")
        else:
            print("Something went wrong! No data to plot!")

    def process_results(self, spec: ExperimentSpec, stats: RunnerStats, plugin: str) -> pd.DataFrame:
        # Calculate percentiles and append to csv
        exec_times = np.array(stats.execution_times)
        median = np.median(exec_times)
        avg = np.average(exec_times)
        percentile_90 = np.percentile(exec_times, 90)
        res_dict = {
            "throughput": stats.success_rate,
            "avg": avg,
            "median": median,
            "90_percentile": percentile_90,
        }
        info = vars(spec)
        data = {**info, **res_dict, "plugin": plugin}
        df = pd.DataFrame(data, index=[0])
        output_dir = spec.workspace_dir + "/" + plugin
        os.makedirs(output_dir, exist_ok=True)
        output_file = spec.get_csv_output_path(output_dir)
        df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
        LOG.info(f"Results written to {output_file}")
        return df

    def _dataset_or_random(self, dataset_id: str, inputs: List[Input]) -> DatasetAlias:
        if dataset_id != "random":
            return get_dataset(dataset_id)
        else:
            adjusted_inputs = [
                Input(name=i.name, dtype=i.dtype, dims=[100 if d == -1 else d for d in i.dims]) for i in inputs
            ]
            return DatasetGen(adjusted_inputs, size=1000).dataset


@dataclass
class RunContext:
    """Here we keep track of max throughput and respective model"""

    max_throughout: float
    model_unique_name: str

    def update_max(self, throughput: float, model_unique_name: str):
        if throughput > self.max_throughout:
            self.max_throughout = throughput
            self.model_unique_name = model_unique_name
