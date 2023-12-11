import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from batched_inference.base.utils import pack_strings, unpack_strings


class IInferenceBackend:
    def generate(self, batch: list[str]):
        raise NotImplementedError()


class _InferFnWRapper:
    def __init__(self, backend: IInferenceBackend):
        self._backend = backend

    
    @batch
    def __call__(self, **inputs: np.ndarray):
        (inputs,) = inputs.values()
        inputs = unpack_strings(inputs)

        outputs = self._backend.generate(inputs)

        outputs = pack_strings(outputs)
        return [outputs]


def run(backend: IInferenceBackend, triton_model_name: str):
    # Connecting inference callable with Triton Inference Server
    with Triton() as triton:
        # Load model into Triton Inference Server
        triton.bind(
            model_name=triton_model_name,
            infer_func=_InferFnWRapper(backend),
            inputs=[
                Tensor(dtype=bytes, shape=(-1,)),
            ],
            outputs=[
                Tensor(dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128)
        )
        
        triton.serve()