import numpy as np


def pack_strings(str_list: list[str]) -> np.array:
    input_data = np.array([[seq] for seq in str_list])
    return np.char.encode(input_data, "utf-8")

def unpack_strings(bytes_arrays: np.array) -> list[str]:
    str_list = np.char.decode(bytes_arrays.astype("bytes"), "utf-8")
    return str_list.squeeze(1).tolist()