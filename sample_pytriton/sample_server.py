import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from vllm import LLM, SamplingParams
from transformers import pipeline
import deepspeed

import argparse

from utils import pack_strings, unpack_strings

MODEL_NAME = "TheBloke/Llama-2-7B-AWQ"


class IInferenceBackend:
    def generate(self, batch: list[str]):
        raise NotImplementedError()
    

class VLLMBackend(IInferenceBackend):
    def __init__(self, hf_model_name):
        self._sampling_params = SamplingParams(
            temperature=0.8,
            top_k=1,
            max_tokens=50
        )
        self._model = LLM(model=hf_model_name, quantization="AWQ")


    def generate(self, batch: list[str]):
        outputs = self._model.generate(batch, self._sampling_params)
        return [output.outputs[0].text for output in outputs]


class DeepSpeedBackend:
    def __init__(self, hf_model_name):
        device = "cuda"
        self._pipe = pipeline("text-generation", model=hf_model_name, device=device)
        self._pipe.model = deepspeed.init_inference(
            self._pipe.model,
            mp_size=1,
            dtype=torch.int8,
            replace_with_kernel_inject=True
        )

    def generate(self, batch):
        outputs = self._pipe(batch, max_length=50)
        if isinstance(batch, str):
            return [outputs[0]["generated_text"]]
        return [out[0]["generated_text"] for out in outputs]
    


class VanillaHFBackend:
    def __init__(self, hf_model_name):
        device = "cuda"
        self._pipe = pipeline("text-generation", model=hf_model_name, device=device)

    def generate(self, batch):
        print("here")
        outputs = self._pipe(batch, max_length=50)
        print(outputs)
        if isinstance(batch, str):
            return [outputs[0]["generated_text"]]
        return [out[0]["generated_text"] for out in outputs]


@batch
def infer_fn(**inputs: np.ndarray):
    (inputs,) = inputs.values()
    inputs = unpack_strings(inputs)

    outputs = model.generate(inputs)

    outputs = pack_strings(outputs)
    return [outputs]


def create_model(backend_type: str):
    if backend_type == "vllm":
        return VLLMBackend(MODEL_NAME)
    elif backend_type == "deepspeed":
        raise ValueError("not working on smaller gpu")
        return DeepSpeedBackend(MODEL_NAME)
    elif backend_type == "hf":
        return VanillaHFBackend(MODEL_NAME)
    else:
        raise ValueError(f"cannot find backend with type {backend_type}")


parser = argparse.ArgumentParser(description="Pytriton server with several inference backends available")
parser.add_argument("-bt", "--backend_type", help="Types available: \"vllm\", \"deepspeed\", \"hf\"", default="vllm")
args = parser.parse_args()

model = create_model(args.backend_type)


# Connecting inference callable with Triton Inference Server
with Triton() as triton:
    # Load model into Triton Inference Server
    triton.bind(
        model_name="LLM",
        infer_func=infer_fn,
        inputs=[
            Tensor(dtype=bytes, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=bytes, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128)
    )
    
    triton.serve()