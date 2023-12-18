import torch
from batched_inference.base import BaseLLM

from batched_inference.utils import (load_tokenizer, read_model_name,
                                     parse_input, parse_output)
from tensorrt_llm.runtime import ModelRunner, PYTHON_BINDINGS

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


class LLM(BaseLLM):
    def __init__(self, tokenizer_dir, trt_engine_dir, **kwargs):
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


    def infer_batch(self, batch: list[str], **infer_params) -> list[str]:
        batch_size = infer_params.pop("batch_size", len(batch))

        outputs = []
        for mini_batch in self._split_to_mini_batches(batch, batch_size):
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
