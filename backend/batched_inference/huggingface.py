import typing as t

from transformers import pipeline
from batched_inference.base import BaseLLM

class LLM(BaseLLM):
    def __init__(self, model_name: str, **kwargs):
        if "device" not in kwargs.keys():
            kwargs["device"] = "cuda"
        self._pipe = pipeline("text-generation", model=model_name, **kwargs)
        if self._pipe.tokenizer.pad_token_id is None:
            self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id


    def infer_batch(self, data: list[str], batch_size: t.Optional[int]=None, **infer_params) -> list[str]:
        # if provided, prefer batch_size from kwargs over the real size of batch
        batch_size = batch_size or len(data)
        return_full_text = infer_params.pop("return_full_text", False)
        outputs = []
        for mini_batch in self._split_to_mini_batches(data, batch_size):
            outputs.extend(self._pipe(
                mini_batch, 
                batch_size=batch_size, 
                return_full_text=return_full_text, 
                **infer_params
            ))
        
        outputs = [out[0]["generated_text"] for out in outputs]
        return outputs
