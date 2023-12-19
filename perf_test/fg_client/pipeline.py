import json
import time

from batched_inference.huggingface import LLM
from services.parser import Parser
from services.processing import Processing
from services.prompter import Prompter

import config

if __name__ == '__main__':
    raw_data = 'data.xlsx'
    model_name = "TheBloke/Llama-2-7B-AWQ"

    df = Parser.load(raw_data)

    uuid = df[config.uuid].to_list()
    issues = df[config.issue_description].to_list()
    issues = Processing.clean_issues(issues)

    llm = LLM(model_name=model_name)

    prompt = Prompter.get_prompt()
    result_format = Prompter.get_result_format()


    issues = issues[:10]
    inp_batch = [prompt.format(text=issue, result_format=result_format) for issue in issues]

    stats = []
    batch_sizes = [1]
    for batch_size in batch_sizes:
        begin = time.time()
        # out = llm.infer_batch(inp_batch, max_tokens=500, top_p=0.8, batch_size=batch_size) # with vllm params
        llm.infer_batch(inp_batch, max_new_tokens=500, batch_size=batch_size, do_sample=True) # with haf params
        end = time.time()
        stats.append({
            "batch_size": batch_size,
            f"inference_time_of_{len(inp_batch)}_samples": (end - begin)
        })
    json.dump(end, open("temp.json", "w"), indent=4)
    json.dump(stats, open("stats.json", "w"), indent=4)


    # begin = time.time()
    # # answers = llm.infer_batch(inp_batch, max_new_tokens=500, batch_size=5, do_sample=True)
    # answers = llm.infer_batch(inp_batch, max_tokens=500, top_p=0.8)
    # end = time.time()
    # print(f"elapsed: {end - begin} s.")
    # json.dump(inp_batch[:10], open("in.json", "w"), indent=4)
    # json.dump(answers[:10], open("out.json", "w"), indent=4)
