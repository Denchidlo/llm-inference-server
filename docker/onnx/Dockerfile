FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

WORKDIR /workdir

COPY . ./llm-inference-server

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir /workdir/llm-inference-server/backend/.[onnx,perf_test]