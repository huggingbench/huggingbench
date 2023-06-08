from gevent import monkey
monkey.patch_all()
from hugging_bench.hugging_bench_config import ExperimentSpec, TritonServerSpec
from hugging_bench.hugging_bench_runner import ExperimentRunner
from client.datasets import get_dataset

def main():
    experiments=[ 
        # ExperimentSpec(format="trt", device="cuda", half=True),
        # ExperimentSpec(format="trt", device="cuda", half=False),
        # ExperimentSpec(format="onnx", device="cuda", half=True),   
        # ExperimentSpec(format="onnx", device="cuda", half=False),
        ExperimentSpec(format="onnx", device="cpu", half=False),
        ExperimentSpec(format="onnx", device="cpu", half=False, batch_size=4),
        # ExperimentSpec(format="onnx", device="cpu", half=True)
    ]
    server_spec = TritonServerSpec()

    # ExperimentRunner("microsoft/resnet-50", experiments, server_spec, dataset=None, model_local_path="/Users/niksa/projects/models/resnet-50").run()
    ExperimentRunner("bert-base-uncased", experiments, server_spec, dataset=None, model_local_path="/Users/niksa/projects/models/bert-base-uncased", task="text-classification").run()

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


if __name__ == "__main__":
    main()
