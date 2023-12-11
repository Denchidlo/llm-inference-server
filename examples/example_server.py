from batched_inference.vllm.server import run

model_name = "TheBloke/Llama-2-7B-AWQ"

run(model_name, triton_model_name="LLM")