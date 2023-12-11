from pytriton.client import ModelClient
from batched_inference.base.utils import pack_strings, unpack_strings


class Client:
    def __init__(self, url, triton_model_name):
        self._triton_client = ModelClient(url, triton_model_name)
    
    def inference_batch(self, batch):
        inp_bytes = pack_strings(batch)
        output = self._triton_client.infer_batch(inp_bytes)
        (output,) = output.values()
        output = unpack_strings(output)
        return output