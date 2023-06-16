from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec
from hugging_bench.hugging_bench_runner import ExperimentRunner
from pathlib import Path
import os

from hugging_bench.hugging_bench_util import ENV_TRITON_SERVER_DOCKER


def test_experiment_runner():
    server_spec = TritonServerSpec()
    experiment = ExperimentSpec(
        hf_id="prajjwal1/bert-tiny", format="onnx", device="cpu", half=False, batch_size=1
    )  # given model only supports batch size 1
    experiment_runner = ExperimentRunner([experiment], server_spec)
    os.environ[
        ENV_TRITON_SERVER_DOCKER
    ] = "ghcr.io/niksajakovljevic/tritonserver:23.04-onnx"  # we use custom docker to reduce image size
    experiment_runner.run()
    csv_file = Path(experiment.get_csv_output_path())
    if not csv_file.exists():
        assert False, "CSV file not generated"
