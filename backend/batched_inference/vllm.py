import typing as t

import vllm

from batched_inference.base import BaseLLM
from batched_inference import AvailableBackends

class LLM(BaseLLM):
    def __init__(self, model_name, **kwargs):
        self.backend_type = AvailableBackends.VLLM
        self._model = vllm.LLM(model=model_name, **kwargs)


    def infer_batch(self, data: list[str], batch_size: t.Optional[int]=None, **infer_params) -> list[str]:
        batch_size = batch_size or len(data)

        sampling_params = vllm.SamplingParams(**infer_params)

        outputs = []
        for mini_batch in self._split_to_mini_batches(data, batch_size):
            outputs.extend(self._model.generate(mini_batch, sampling_params))

        outputs = [output.outputs[0].text for output in outputs]
        return outputs
