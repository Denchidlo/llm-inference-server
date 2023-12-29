FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /workdir

COPY . ./llm-inference-server

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir /workdir/llm-inference-server/backend/.[deepspeed,perf_test]