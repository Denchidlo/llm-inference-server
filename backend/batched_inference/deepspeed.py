import torch
import deepspeed
from batched_inference import huggingface, AvailableBackends

class LLM(huggingface.LLM):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        self.backend_type = AvailableBackends.DS

        # model surgery for TheBloke/Llama-2-7b-AWQ
        for module in self._pipe.model.modules():
            if type(module).__name__ == "WQLinear_GEMM":
                # creating attribute weight for quantized layers
                module.weight = module.qweight

        self._pipe.model = deepspeed.init_inference(
            self._pipe.model,
            dtype=torch.int8,
            # use optimized kernels(doesnt work with TheBloke/Llama-2-7b-AWQ)
            replace_with_kernel_inject=False
        )
