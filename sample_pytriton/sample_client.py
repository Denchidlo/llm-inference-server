import numpy as np
from pytriton.client import ModelClient

from utils import pack_strings, unpack_strings

input_data = [
    "inp1",
    "inp2",
    "inp3"
]

with ModelClient("localhost:8000", "Linear") as client:
    input_data = pack_strings(input_data)
    output = client.infer_batch(input_data)
    output = unpack_strings(output)

print(output)