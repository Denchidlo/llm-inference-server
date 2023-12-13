import vllm

from batched_inference.base import BaseLLM

class LLM(BaseLLM):
    def __init__(self, model_name, **kwargs):
        self._model = vllm.LLM(model=model_name, **kwargs)


    def infer_batch(self, batch: list[str], **infer_params) -> list[str]:
        sampling_params = vllm.SamplingParams(**infer_params)
        outputs = self._model.generate(batch, sampling_params)
        return [output.outputs[0].text for output in outputs]
