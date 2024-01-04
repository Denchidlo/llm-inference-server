import time
import json
import argparse

from batched_inference import create_model, infer, AvailableBackends


CONVERTED_MODEL_PATH = "/code/tensorrt_llm/examples/llama/llama_2_hf/7b/trt_engines/weight_only/1-gpu"
MODEL_NAME = "TheBloke/Llama-2-7B-AWQ"
IN_FILE = "in.json"
OUT_FILE = "out.json"


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=list(b.value for b in AvailableBackends))
    return parser.parse_args(args=args)


def load_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


def store_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def main(args):    
    llm = create_model(args.backend, MODEL_NAME, CONVERTED_MODEL_PATH)

    data  = load_data(IN_FILE)

    begin = time.time()
    output = infer(llm, data, max_new_tokens=50, batch_size=4, temperature=0.8, top_k=1)
    end = time.time()
    print(f"elapsed: {end - begin} s.")

    store_data(output, OUT_FILE)



if __name__ == "__main__":
    args = parse_arguments()
    main(args)