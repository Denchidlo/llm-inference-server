from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForCausalLM

from batched_inference import huggingface, AvailableBackends

class LLM(huggingface.LLM):
    def __init__(self, hf_model_name_for_tokenizer, onnx_model_path, **kwargs):
        self.backend_type = AvailableBackends.ONNX

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name_for_tokenizer)
        model = ORTModelForCausalLM.from_pretrained(onnx_model_path, provider="CUDAExecutionProvider")
        self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")
        if self._pipe.tokenizer.pad_token_id is None:
            self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id
