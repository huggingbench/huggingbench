# CLI

## Run benchmark
`hbench run --format onnx --hf_ids "bert-base-uncased" "distilbert-base-uncased" "microsoft/resnet-5"`

```bash
python3 src/bench/cli.py                                            
usage: cli.py [-h] {run,chart} ...

HuggingBench CLI

positional arguments:
  {run,chart}

options:
  -h, --help   show this help message and exit
```


## Char the results
`hbench chart --input 'temp/mymodel.csv'`

```bash
python3 src/bench/cli.py chart
usage: cli.py chart [-h] --input INPUT
cli.py chart: error: the following arguments are required: --input
```