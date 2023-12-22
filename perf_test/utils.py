from enum import Enum
import time

import pandas as pd
import numpy as np


class AvailableBackends(str, Enum):
    HF = "huggingface"
    VLLM = "vllm"
    TENSORRTLLM = "tensorrt-llm"
    DS = "deepspeed"
    ONNX = "onnx-runtime"


WRONG_BACKEND_ERROR_MESSAGE = f"Specified backend type isnt supported. Available backends: {list(b.value for b in AvailableBackends)}"


def create_model(model_dir: str | None, converted_model_dir: str | None, backend: str):
    if backend == AvailableBackends.HF:
        from batched_inference.huggingface import LLM
        llm = LLM(model_dir)
    elif backend == AvailableBackends.VLLM:
        from batched_inference.vllm import LLM
        llm = LLM(model_dir)
    elif backend == AvailableBackends.TENSORRTLLM:
        from batched_inference.tensorrt import LLM
        llm = LLM(model_dir, converted_model_dir)
    elif backend == AvailableBackends.DS:
        from batched_inference.deepspeed import LLM
        llm = LLM(model_dir)
    elif backend == AvailableBackends.ONNX:
        from batched_inference.onnx import LLM
        llm = LLM(model_dir, converted_model_dir)
    else:
        raise ValueError(WRONG_BACKEND_ERROR_MESSAGE)
    return llm


def inference_model(model, backend, batch, max_new_tokens, batch_size, **kwargs):
    if backend in [AvailableBackends.HF, 
                   AvailableBackends.DS, 
                   AvailableBackends.TENSORRTLLM, 
                   AvailableBackends.ONNX]:
        return model.infer_batch(
            batch, 
            batch_size=batch_size, 
            max_new_tokens=max_new_tokens, 
            **kwargs
        )
    if backend == AvailableBackends.VLLM:
        return model.infer_batch(
            batch, 
            batch_size=batch_size, 
            max_tokens=max_new_tokens, 
            **kwargs
        )
    raise ValueError(WRONG_BACKEND_ERROR_MESSAGE)


class InferenceMeasurer:
    def __init__(self, dataloader_fn):
        """
        Args:
            dataloader_fn: function that loads data and returns it in the form of list of strings
        """
        self._dataloader_fn = dataloader_fn

    def load_data(self, filename, **dataloader_kwargs):
        self._data = self._dataloader_fn(filename, **dataloader_kwargs)

    def measure(self, model, backend, max_tokens_generated: int=50, batch_size: int=1):
        begin = time.time()
        _ = inference_model(
            model,
            backend,
            self._data, 
            max_new_tokens=max_tokens_generated, 
            batch_size=batch_size
        )
        end = time.time()
        inference_time = end - begin
        return inference_time


def compute_gpu_stats(filename):
    df = pd.read_csv(filename)
    stats = {}
    total_mem = float(df[" memory.total [MiB]"].values[0][:-4])
    stats[" memory.total [MiB]"] = total_mem
    for col in [" utilization.gpu [%]", " utilization.memory [%]"]:
        arr = np.array([float(val[1:-2]) for val in df[col].values])
        # remove outliars
        alpha = 0.05
        num_rows = df.shape[0]
        lmargin = int(num_rows * alpha)
        rmargin = -lmargin if lmargin > 0 else num_rows
        arr = np.sort(arr)[lmargin:rmargin]

        stats[col] = arr.mean()
    return stats 
