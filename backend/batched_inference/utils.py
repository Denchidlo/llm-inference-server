import json
from pathlib import Path
from typing import Optional

import torch
from transformers import T5Tokenizer, AutoTokenizer
import tensorrt_llm



def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'gpt',
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        if tokenizer_type == "llama":
            use_fast = False
        else:
            use_fast = True
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    else:
        # For gpt-next, directly load from tokenizer.model
        if model_name != 'gpt':
            raise ValueError("model_name must be gpt if vocab_file is provided")
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left')

    if model_name == 'qwen':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == 'raw':
            pad_id = gen_config['pad_token_id']
            end_id = gen_config['eos_token_id']
        elif chat_format == 'chatml':
            pad_id = tokenizer.im_end_id
            end_id = tokenizer.im_end_id
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == 'glm_10b':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def read_model_name(engine_dir: str):
    engine_version = tensorrt_llm.builder.get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        model_name = config['builder_config']['name']
    else:
        model_name = config['pretrained_config']['architecture']

    return model_name


def parse_input(tokenizer,
                input_text=None,
                add_special_tokens=True,
                max_input_length=2000,
                pad_id=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    for curr_text in input_text:
        input_ids = tokenizer.encode(curr_text,
                                     add_special_tokens=add_special_tokens,
                                     truncation=True,
                                     max_length=max_input_length)
        batch_input_ids.append(input_ids)
    
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids
    ]
    return batch_input_ids


def parse_output(tokenizer, 
                 output_ids, 
                 output_lengths, 
                 input_lengths):
    output_strings = []
    batch_size, num_beams, _ = output_ids.size()
    chosen_beam = 0
    for batch_idx in range(batch_size):
        output_begin = input_lengths[batch_idx]
        output_end = output_lengths[batch_idx][chosen_beam]
        output = output_ids[batch_idx][chosen_beam][output_begin:output_end].tolist()
        output_string = tokenizer.decode(output)
        output_strings.append(output_string)
    return output_strings
