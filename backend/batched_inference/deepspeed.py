import torch
import deepspeed
from batched_inference import huggingface

class LLM(huggingface.LLM):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)

        self._pipe.model = deepspeed.init_inference(
            self._pipe.model,
            dtype=torch.int8,
            # use optimized kernels(doesnt work with TheBloke/Llama-2-7b-AWQ)
            replace_with_kernel_inject=False
        )
