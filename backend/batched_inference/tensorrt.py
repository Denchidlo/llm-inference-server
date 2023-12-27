import typing as t

import torch
from batched_inference.base import BaseLLM
from batched_inference import AvailableBackends

from batched_inference.utils import (load_tokenizer, read_model_name,
                                     parse_input, parse_output)
from tensorrt_llm.runtime import ModelRunner, PYTHON_BINDINGS

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


class LLM(BaseLLM):
    def __init__(self, tokenizer_dir, trt_engine_dir, **kwargs):
        self.backend_type = AvailableBackends.TENSORRTLLM
        model_name = read_model_name(trt_engine_dir)
        self._tokenizer, _, _ = load_tokenizer(
            tokenizer_dir,
            model_name=model_name,
        )
        runner_cls = ModelRunnerCpp if PYTHON_BINDINGS else ModelRunner

        self._runner = runner_cls.from_dir(
            engine_dir=trt_engine_dir,
            **kwargs
        )


    def infer_batch(self, data: list[str], batch_size: t.Optional[int]=None, **infer_params) -> list[str]:
        batch_size = batch_size or self._runner.max_batch_size
        if self._runner.max_batch_size < batch_size:
            print(f"Warning: set batch size {batch_size} exceeds trt engine " \
                    f"max batch size {self._runner.max_batch_size}, " \
                    f"setting batch_size to {self._runner.max_batch_size}")
            batch_size = self._runner.max_batch_size

        outputs = []
        for mini_batch in self._split_to_mini_batches(data, batch_size):
            outputs.extend(self._infer_mini_batch(mini_batch, **infer_params))

        return outputs
    

    def _infer_mini_batch(self, mini_batch: list[str], **infer_params) -> list[str]:
        input_ids = parse_input(
            self._tokenizer, 
            mini_batch,
        )
        input_lengths = [x.size(1) for x in input_ids]

        with torch.no_grad():
            outputs = self._runner.generate(
                input_ids,
                output_sequence_lengths=True,
                return_dict=True,
                pad_id=self._tokenizer.pad_token_id,
                end_id=self._tokenizer.eos_token_id,
                **infer_params
            )

            torch.cuda.synchronize()
        
        output_ids = outputs["output_ids"]
        output_lengths = outputs['sequence_lengths']

        output_strings = parse_output(
            self._tokenizer,
            output_ids,
            output_lengths,
            input_lengths
        )
        return output_strings
