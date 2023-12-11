import numpy as np

from pytriton.decorators import batch, group_by_values
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

import batched_inference.base.utils as data_utils


class IInferenceBackend:
    def generate(self, batch: list[str], **inference_params):
        raise NotImplementedError()


class _InferFnWRapper:
    def __init__(self, backend: IInferenceBackend):
        self._backend = backend


    @batch
    @group_by_values("temperature", "max_tokens")
    def __call__(self, batch, **kwargs):
        kwargs = {k: v[(0,) * v.ndim] for k, v in kwargs.items()}
        inputs = data_utils.unpack_strings(batch)

        outputs = self._backend.generate(inputs, **kwargs)

        outputs = data_utils.pack_strings(outputs)
        return {"outputs": outputs}


def run(backend: IInferenceBackend, triton_model_name: str):
    # Connecting inference callable with Triton Inference Server
    with Triton() as triton:
        # Load model into Triton Inference Server
        triton.bind(
            model_name=triton_model_name,
            infer_func=_InferFnWRapper(backend),
            inputs=[
                Tensor(name="batch", dtype=bytes, shape=(-1,)),
                Tensor(name="temperature", dtype=np.float32, shape=(1,)),
                Tensor(name="max_tokens", dtype=np.float32, shape=(1,)),
            ],
            outputs=[
                Tensor(name="outputs", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128)
        )
        
        triton.serve()