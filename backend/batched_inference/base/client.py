import numpy as np
from pytriton.client import ModelClient
import batched_inference.base.utils as data_utils


class Client:
    def __init__(self, url, triton_model_name):
        self._triton_client = ModelClient(url, triton_model_name)
    
    def inference_batch(self, batch, **infer_params):
        inp_bytes = data_utils.pack_strings(batch)
        infer_params = {
            k: np.full(shape=(len(inp_bytes), 1), fill_value=v, dtype=np.float32) 
            for k, v in infer_params.items()
        }
        output = self._triton_client.infer_batch(batch=inp_bytes, **infer_params)
        (output,) = output.values()
        output = data_utils.unpack_strings(output)
        return output