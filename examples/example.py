import time
import json

from batched_inference.vllm import LLM

model_name = "TheBloke/Llama-2-7B-AWQ"
in_file = "in.json"
out_file = "out.json"

def load_data(filename):
    return json.load(open(filename, "r"))


def store_data(data, filename):
    json.dump(data, open(filename, "w"), indent=4)


llm = LLM(model_name)
data  = load_data(in_file)

begin = time.time()
output = llm.infer_batch(data, temperature=0.8, max_tokens=50, top_p=0.75)
end = time.time()
print(f"elapsed: {end - begin} s.")

store_data(output, out_file)

