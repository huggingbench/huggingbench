# USAGE:
# docker build -t optimum-cli .
# docker exec -v ${PWD}:/workspace/ -w /workspace/ -it optimum-cli bash
# docker run -v ${PWD}:/workspace/ pytorch23  python -m transformers.onnx --model=google/vit-base-patch16-224 --atol=1e-3 onnx/vit

FROM python:3.9

RUN python -m pip install --upgrade pip
RUN python -m pip install transformers==4.29.2
RUN python -m pip install optimum==1.8.5
RUN python -m pip install optimum[onnxruntime]==1.8.5
## Comment out if you don't have GPU
## We might want to create non-gpu version of this dockerfile
# RUN python -m pip pip install optimum[onnxruntime-gpu]==1.8.5

# RUN python -m pip install optimum[openvino,nncf]==1.8.5
# RUN python -m pip install polygraphy==0.47.1
# RUN python -m pip install tritonclient[all]==2.33.0
# RUN python -m pip install protobuf==3.20.2


CMD []
