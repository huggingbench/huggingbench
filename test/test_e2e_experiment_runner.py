from gevent import monkey

from bench.plugin_manager import PluginManager, PLUGINS

monkey.patch_all()  # this is needed to make gevent work with pytest
import os
from pathlib import Path

from bench.config import ExperimentSpec
from bench.exp_runner import ExperimentRunner
from server.util import ENV_TRITON_SERVER_DOCKER


def test_experiment_runner():
    experiment = ExperimentSpec(
        hf_id="prajjwal1/bert-tiny", format="onnx", device="cpu", precision="fp32", batch_size=1, workspace_dir="./temp"
    )  # given model only supports batch size 1
    os.environ[
        ENV_TRITON_SERVER_DOCKER
    ] = "ghcr.io/niksajakovljevic/tritonserver:23.04-onnx"  # we use custom docker to reduce image size
    plugin_manager = PluginManager()
    triton_plugin = plugin_manager.get_plugin("triton")
    ExperimentRunner(triton_plugin, [experiment]).run()

    csv_file = Path(experiment.get_csv_output_path("./temp"))
    if not csv_file.exists():
        assert False, "CSV file not generated"
