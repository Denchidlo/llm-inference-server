import time
import json
import random
import argparse


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, help="One of \{vllm, hf, trt\}")
    return parser.parse_args(args=args)


def load_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


def store_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def main(args):
    trt_engine_path = "/code/tensorrt_llm/examples/llama/llama_2_hf/7b/trt_engines/weight_only/1-gpu"
    tokenizer_path = "/code/tensorrt_llm/examples/llama/llama_2_hf/7b/"
    model_name = "TheBloke/Llama-2-7B-AWQ"
    in_file = "in.json"
    out_file = "out.json"
    
    if args.backend == "vllm":
        from batched_inference.vllm import LLM
        llm = LLM(model_name)
    elif args.backend == "hf":
        from batched_inference.huggingface import LLM
        llm = LLM(model_name)
    elif args.backend == "trt":
        from batched_inference.tensorrt import LLM
        llm = LLM(tokenizer_path, trt_engine_path)

    data  = load_data(in_file)

    begin = time.time()
    if args.backend == "vllm":
        output = llm.infer_batch(
            data, 
            temperature=0.8,
            max_tokens=50, 
            top_p=0.9, 
        )
    elif args.backend == "hf":
        output = llm.infer_batch(
            data, 
            temperature=0.8,
            max_new_tokens=50, 
            do_sample=True
        )
    elif args.backend == "trt":
        output = llm.infer_batch(
            data, 
            temperature=0.8, 
            max_new_tokens=50, 
            top_p=0.9, 
            top_k=5, 
            random_seed=random.randint(0, 1e6)
        )
    end = time.time()
    print(f"elapsed: {end - begin} s.")

    store_data(output, out_file)



if __name__ == "__main__":
    args = parse_arguments()
    main(args)