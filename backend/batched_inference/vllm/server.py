from vllm import LLM, SamplingParams
import batched_inference.base.server as base_server


class VLLMBackend(base_server.IInferenceBackend):
    def __init__(self, hf_model_name):
        self._sampling_params = SamplingParams(
            temperature=0.8,
            top_k=1,
            max_tokens=50
        )
        self._model = LLM(model=hf_model_name, quantization="AWQ")


    def generate(self, batch: list[str]):
        outputs = self._model.generate(batch, self._sampling_params)
        return [output.outputs[0].text for output in outputs]
    

def run(model_name, triton_model_name):
    back = VLLMBackend(model_name)
    base_server.run(back, triton_model_name=triton_model_name)