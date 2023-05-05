#!/bin/bash

REPO_PARENT=$(pwd)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --cpus 4 \
    -v ${REPO_PARENT}/model_repository:/models nvcr.io/nvidia/tritonserver:23.03-py3 tritonserver \
    --model-repository=/models 
    # --allow-metrics=true \
    # --load-model=resnet50_onnx \
    