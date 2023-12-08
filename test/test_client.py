import time
import random
import string
import numpy as np
from pytriton.client import ModelClient

from sample_pytriton.utils import pack_strings, unpack_strings


SERVER_SOCK = "localhost:8000"
MODEL_NAME = "LLM"


def run_client(data: list[str], num_it: int) -> float:
    inp_bytes = pack_strings(data)
    with ModelClient(SERVER_SOCK, MODEL_NAME) as client:
        begin = time.time()
        for _ in range(num_it):
            output = client.infer_batch(inp_bytes)
        end = time.time()
        total_avg = (end - begin) / num_it
    return total_avg


def test(filename: str, num_it=1):
    with open(filename, "r") as f:
        lines = f.read().strip(" \n\t").split('\n')
    

    elapsed = run_client(lines, num_it)

    print(f"average elapsed time: {elapsed} s.")


def gen_file(filename, num_str: int, str_len: int) -> None:
    with open(filename, "w") as f:
        for _ in (num_str):
            f.write("".join(random.choise(string.ascii_lowercase + ' ') for _ in range(str_len)))


def main():
    filename = "temp.txt"
    gen_file(filename, num_str=50, str_len=100)
    test(filename, num_it=1)


if __name__ == "__main__":
    main()