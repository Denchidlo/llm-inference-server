import typing as t
from enum import Enum


class AvailableBackends(str, Enum):
    HF = "huggingface"
    VLLM = "vllm"
    TENSORRTLLM = "tensorrt-llm"
    DS = "deepspeed"
    ONNX = "onnx-runtime"


WRONG_BACKEND_ERROR_MESSAGE = f"Specified backend type isnt supported. Available backends: {list(b.value for b in AvailableBackends)}"


def create_model(backend: str, model_dir: str, converted_model_dir: t.Optional[str]=None):
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


def infer(model, data, max_new_tokens, batch_size, **kwargs):
    backend = model.backend_type
    kwargs['max_new_tokens'] = max_new_tokens
    if backend == AvailableBackends.VLLM:
        kwargs['max_tokens'] = kwargs['max_new_tokens']
        del kwargs['max_new_tokens']

    output = model.infer_batch(data, batch_size=batch_size, **kwargs)
    return output
