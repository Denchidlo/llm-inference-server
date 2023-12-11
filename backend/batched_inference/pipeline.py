import json

from batched_inference.base.client import Client


def infer_batch(input, **kwargs):
    # TODO: client should be initialized outside?
    client = Client("localhost:8000", "LLM")
    output = client.inference_batch(input, **kwargs)
    return output


def load_data(filename):
    return json.load(open(filename, "r"))


def store_data(data, filename):
    json.dump(data, open(filename, "w"), indent=4)
