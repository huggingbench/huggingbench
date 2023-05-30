# mlperf

### setup env
```
python3 -m venv env
source  env/bin/activate
python3 -m pip install .
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
