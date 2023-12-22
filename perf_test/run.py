import argparse
import subprocess
import json
import sys

from fg_client.pipeline import load_data as fg_load_data
from utils import AvailableBackends, create_model, InferenceMeasurer, compute_gpu_stats


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", 
                        choices=list(b.value for b in AvailableBackends), 
                        default=f"{AvailableBackends.HF}", type=str, help="backend type")
    parser.add_argument("--model_dir", default="TheBloke/Llama-2-7B-AWQ",
                         type=str, help="model directory, or name on huggingface hub")
    parser.add_argument("--converted_model_dir", type=str)
    parser.add_argument("--data_path", default="./fg_client/data.xlsx", type=str)
    parser.add_argument("--num_samples", default=10, type=int, help="number of samples to run inference on")
    parser.add_argument("--batch_size_list", default="1", type=str, 
                        help="List of batch sizes to run test on. You can "
                             "either pass a single value, or a comma separrated list")
    parser.add_argument("--max_tokens_generated", default=50, type=int, help="max amount of generated tokens")
    parser.add_argument("--output_file", type=str, required=False)
    args = parser.parse_args()
    args.batch_size_list = [int(val) for val in args.batch_size_list.strip().split(',')]
    return args


def main(args):
    gpu_logs_file = "gpu_log.csv"

    model = create_model(args.model_dir, args.converted_model_dir, args.backend)

    measurer = InferenceMeasurer(fg_load_data)
    measurer.load_data(args.data_path, num_samples=args.num_samples)

    result_dict = []
    for batch_size in args.batch_size_list:
        cmd_str = "nvidia-smi --query-gpu=name,timestamp,utilization.gpu,utilization.memory,memory.total -lms 100 --format=csv -f"
        gpu_logging_proc = subprocess.Popen(cmd_str.split(' ') + [gpu_logs_file])

        inference_time = measurer.measure(model, args.backend, 
                                          args.max_tokens_generated, batch_size)

        gpu_logging_proc.terminate()
        gpu_logging_proc.wait()

        gpu_stats = compute_gpu_stats(gpu_logs_file)

        result_dict.append({
            "batch_size": batch_size,
            "inference_time": inference_time,
            "gpu_stats": gpu_stats
        })

    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            json.dump(result_dict, f, indent=4)
    else:
        json.dump(result_dict, sys.stdout, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)