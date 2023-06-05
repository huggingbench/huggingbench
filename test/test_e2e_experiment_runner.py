from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec
from hugging_bench.hugging_bench_runner import ExperimentRunner
from pathlib import Path

def test_experiment_runner():
    server_spec = TritonServerSpec()
    experiment = ExperimentSpec(format="onnx", device="cpu", half=False, batch_size=1) # given model only supports batch size 1
    experiment_runner = ExperimentRunner("prajjwal1/bert-tiny", [experiment], server_spec)
    experiment_runner.run()
    csv_file = Path(experiment_runner.output)
    if not csv_file.exists():
        assert False, "CSV file not generated"