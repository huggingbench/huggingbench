# mlperf

### setup env
```
python3 -m venv env
source  env/bin/activate
pip3 install -r requirements.txt
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

### run load test

run with web ui with charts visualization
`locust -f pytorch_user.py,load_test_plan.py `

run and record stats in a csv file (good for comparison across experiments)
`locust -f pytorch_user.py,load_test_plan.py --csv=resnet50-pytorch --headless`

`locust -f onnx_user.py,load_test_plan.py --csv=resnet50-onnx --headless`
