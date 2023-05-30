# USAGE:
# docker build -t optimum-cli .
# docker exec -v ${PWD}:/workspace/ -w /workspace/ -it optimum-cli bash
# docker run -v ${PWD}:/workspace/ pytorch23  python -m transformers.onnx --model=google/vit-base-patch16-224 --atol=1e-3 onnx/vit

FROM python:3.9

ENV POLYGRAPHY_AUTOINSTALL_DEPS=1
RUN python -m pip install --upgrade pip
RUN python -m pip install nvidia-pyindex==1.0.9
RUN python -m pip install colored polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com

CMD []
