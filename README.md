# HuggingBench
Welcome to the HuggingBench and Open Database project! This project aims to simplify the process of benchmarking machine learning models, specifically focusing on models available on the Hugging Face Hub. We provide a user-friendly approach to deploying these models on the Triton server, utilizing various backends such as OpenVino, TensorRT, and Onnx, while also exploring the impact of Quantization.

This project started as a hobby prject and it is in it's early days, but we quickly became convinced this can be a meaningful contribution to AI (particularly MLOps) community. By open-sourcing our code, we aim to empower developers and researchers to make informed decisions when deploying their models, ensuring optimal performance without compromising efficiency. Additionally, we are in process of building an open database that houses benchmark results of reproducable benchmarks for all open-source models, allowing users to easily compare and contrast performance metrics.

In this repository, you will find the necessary code, documentation, and examples to get started with ML model benchmarking. We encourage you to explore the different functionalities, experiment with your own models, and contribute to the growth of the open database.

Join us on this exciting journey as we revolutionize ML model benchmarking and create a valuable resource for the AI community. Let's unlock the full potential of our models together!
Happy benchmarking and welcome aboard!

## Project structure

## Current limitations

## Roadmap

### Setup
```
python3 -m venv env
source  env/bin/activate
pip install colored polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com
python3 -m pip install -e . --extra-index-url https://pypi.ngc.nvidia.com
```

```
isort test src
autoflake --in-place --remove-unused-variables --recursive src test
```

#### Test
run
`pytest`

#### Run reproducable benchmarks with CLI
`runbench --format onnx --hf_ids "bert-base-uncased" "distilbert-base-uncased" "microsoft/resnet-5"`

```
runbench --help
usage: runbench [-h] [--format [{onnx,trt,openvino} ...]] [--device [{cpu,cuda} ...]] [--half [HALF ...]] [--client_worker [CLIENT_WORKER ...]]
                [--hf_ids [HF_IDS ...]] [--model_local_path [MODEL_LOCAL_PATH ...]] [--task [TASK ...]] [--batch_size [BATCH_SIZE ...]]
                [--instance_count INSTANCE_COUNT] [--async_client ASYNC_CLIENT]

runbench options

options:
  -h, --help            show this help message and exit
  --format [{onnx,trt,openvino} ...]
  --device [{cpu,cuda} ...]
  --half [HALF ...]     Whether to use half precision
  --client_worker [CLIENT_WORKER ...]
                        Number of client workers sending concurrent requests to Triton
  --hf_ids [HF_IDS ...]
                        HuggingFace model ID(s) to benchmark
  --model_local_path [MODEL_LOCAL_PATH ...]
                        If not specified, will download from HuggingFace. When given a task name must also be specified.
  --task [TASK ...]     Model task(s) to benchmark. Used with --model_local_path
  --batch_size [BATCH_SIZE ...]
                        Batch size(s) to use for inference..
  --instance_count INSTANCE_COUNT
                        Triton server instance count.
  --async_client ASYNC_CLIENT
                        Use async triton client.
```

### inspect input output shape
`polygraphy inspect model model.onnx --mode=onnx`
no pytorch support but there are other ways surely!!!?


### run triton 
```bash
cd triton-server
./run-server.sh
./run-prom.sh
```

### client: run load test 

`cd client`

Run BERT load test with Web UI with charts visualization:
`locust -f bert.py `

Run and record stats in a CSV file (good for comparison across experiments)
`locust -f bert.py,load_test_plan.py --csv=bert-onnx --headless`

Run Resnet50 load test:
`locust -f resnet.py`

To add new ML mode simply extend `TritonUser` class and provide Dataset.

## Open vino learning
Openvino Models with Dynamic shape are not supported in latest version of triton

It throws the following error: `Internal: openvino error in retrieving original shapes from input input_ids : get_shape was called on a descriptor::Tensor with dynamic shape`

In order to resolve this, in conversion from onnx to OV, input shape must be specified and input shape must include the batch size.
In triton server also the input shape (excluding the batch must match the specified input shape of the model) otherwise triton tries to invoke 
the method above in the error and same error is raised. It seems, this was way we are just avoiding the call to `get_shape` method.
None of the shape dimensions can be -1 as this causes the invocation of the method hence an error.

Working conversion of onnx example:
`sudo docker run --rm -v /home/kia/mlperf:/home/kia/mlperf openvino mo --input_model=/home/kia/mlperf/temp/prajjwal1-bert-tiny-None-onnx-0.001-False-cpu/model.onnx --output_dir=/home/kia/mlperf/temp/prajjwal1-bert-tiny-None-openvino-0.001-False-cpu --input=input_ids[1,100],attention_mask[1,100],token_type_ids[1,100]`

Workin triton config:
```json
    "name": "prajjwal1-bert-tiny-None-openvino-0.001-False-cpu",
    "versions": [
        "1"
    ],
    "platform": "openvino",
    "inputs": [
        {
            "name": "input_ids",
            "datatype": "INT64",
            "shape": [
                -1,
                100
            ]
        },
        {
            "name": "attention_mask",
            "datatype": "INT64",
            "shape": [
                -1,
                100
            ]
        },
        {
            "name": "token_type_ids",
            "datatype": "INT64",
            "shape": [
                -1,
                100
            ]
        }
    ],
    "outputs": [
        {
            "name": "last_hidden_state",
            "datatype": "FP32",
            "shape": [
                -1,
                100,
                128
            ]
        }
    ]
}
```
