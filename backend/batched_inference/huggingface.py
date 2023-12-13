from transformers import pipeline
from batched_inference.base import BaseLLM

class LLM(BaseLLM):
    def __init__(self, model_name: str, **kwargs):
        if "device" not in kwargs.keys():
            kwargs["device"] = "cuda"
        self._pipe = pipeline("text-generation", model=model_name, **kwargs)
        if self._pipe.tokenizer.pad_token_id is None:
            self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id


    def infer_batch(self, batch: list[str], **infer_params) -> list[str]:
        # if provided, prefer batch_size from kwargs over the real size of batch
        batch_size = infer_params.pop("batch_size", len(batch))
        outputs = self._pipe(batch, batch_size=batch_size, **infer_params)
        return [out[0]["generated_text"][len(inp):] for inp, out in zip(batch, outputs)]
