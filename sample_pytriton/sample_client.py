import numpy as np
from pytriton.client import ModelClient

from utils import pack_strings, unpack_strings

input_data = [
    "Hello",
    "I am Santa Claus",
    "inp3",
]

with ModelClient("localhost:8000", "LLM") as client:
    inp_bytes = pack_strings(input_data)
    output = client.infer_batch(inp_bytes)
    (output,) = output.values()
    output = unpack_strings(output)

for i, o in zip(input_data, output):
    print(f"input: {i}")
    print(f"output: {o}")
    print('-'*10)