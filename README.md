# llm-inference-server

## Installation
- vllm, huggingface, deepspeed, onnx-runtime:
    Build docker image with supplied Dockerfiles in `docker/` directory.
- tensorrt-llm:
    Official repo does not provide docker image yet, so you need to build it from source([official installation guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md)). Note: you must build release build, because `Dockerfile` at `docker/tensorrt_llm` relies on release image. Then you can build final image via `docker/tensorrt_llm/Dockerfile`.

Build example(run from root directory):

```bash
docker build -t <image_tag> -f docker/vllm/Dockerfile .
```

After that, you can start a container using following command:

```bash
docker run -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  \
                --gpus=all \
                --volume <path_to_dir_on_host_you_want_to_share_with_container>:<abs_path_to_shared_folder_in_container> \
                --name tensorrt_llm-backend-user \
                <image_tag> 
```

Note: You can install package into your environment via `pip install backend/.[<extras>]`. For available extras please refer to package info or `backend/setup.py`.

## Usage

1. Convertation
    - vllm, huggingface, deepspeed

        Convertation is not required

    - onnx-runtime

        ```bash
        optimum-cli export onnx -m bigscience/bloom-1b1 --optimize O2 --device cuda --fp16 bloom_onnx/
        ```

        Example above performs convertation of bloom-1b1 to onnx format using optimum cli. `bloom_onnx/` here is output dir, where converted model is stored.
        For more examples and quantization refer to [original guide](https://github.com/huggingface/optimum#onnx--onnx-runtime).
        
    - tensorrt-llm

        Script for building llama-like model engine can be found [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama).
        Further actions are done from `examples/llama/` directory in Tensorrt-llm repo.
        Clone model repo:

        ```bash
        mkdir -p ./tmp/llama-2-7b-hf && git clone https://huggingface.co/daryl149/llama-2-7b-hf ./tmp/llama-2-7b-hf
        ```
        Build an engine(for more build examples refer to original repo):

        ```bash
        # Build the LLaMA 7B model using a single GPU and apply INT8 weight-only quantization.
        python build.py --model_dir ./tmp/llama-2-7b-hf/ \
                        --dtype float16 \
                        --remove_input_padding \
                        --use_gpt_attention_plugin float16 \
                        --enable_context_fmha \
                        --use_gemm_plugin float16 \
                        --use_weight_only \
                        --max_batch_size 4 \
                        --max_input_len 1024 \
                        --output_dir ./tmp/llama-2-7b-hf/trt_engines/weight_only/1-gpu/
        ```

2. Creating model instance
    ```python
    from batched_inference import AvailableBackends, create_model

    model_dir = '/path/to/hf/model/directory'
    converted_model_dir = '/path/to/converted/model/directory'

    model = create_model(AvailableBackends.TENSORRTLLM, model_dir, converted_model_dir)
    ```
    
    Note: parameter `converted_model_dir` is generally optional and is only required for backend types, that demand special converted format of original huggingface model. In this case parameter `model_dir` is used only for loading tokenizer.
3. Inference

    ```python
    from batched_inference import infer

    # this package does not provide loading/storing data to the storage, 
    # it is viewed as a client's responsibility
    from my_data_loader import load_data

    data_file = "mydata.json"
    data = load_data(data_file)

    # both data and result have type list[str]
    result = infer(model, data, max_new_tokens=50, batch_size=4)
    ```

4. Performance testing

    ```bash
    python perf_test/run.py --backend vllm \
                            --model_dir TheBloke/Llama-2-7B-AWQ \
                            --data_path perf_test/fg_client/data.xlsx \
                            --num_samples 10 \
                            --batch_size_list 1,2,4 \
                            --max_tokens_generated 100 \
                            --output_file perf_test/out.json
    ```

    This script writes to a file(or stdout, if file is not specified) statistics(walltime of inference, gpu utilization and memory consumption) on model run in json format.