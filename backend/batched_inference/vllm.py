import vllm

from batched_inference.base import BaseLLM

class LLM(BaseLLM):
    def __init__(self, model_name, **kwargs):
        self._model = vllm.LLM(model=model_name, **kwargs)


    def infer_batch(self, batch: list[str], **infer_params) -> list[str]:
        batch_size = infer_params.pop("batch_size", len(batch))

        sampling_params = vllm.SamplingParams(**infer_params)

        outputs = []
        for mini_batch in self._split_to_mini_batches(batch, batch_size):
            outputs.extend(self._model.generate(mini_batch, sampling_params))

        outputs = [output.outputs[0].text for output in outputs]
        return outputs
