FROM tensorrt_llm/release

WORKDIR /workdir

COPY . ./llm-inference-server

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir /workdir/llm-inference-server/backend/.[tensorrt-llm,perf_test]