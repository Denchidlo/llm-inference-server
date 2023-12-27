from enum import Enum
import time

import pandas as pd
import numpy as np

from batched_inference import infer


class InferenceMeasurer:
    def __init__(self, dataloader_fn):
        """
        Args:
            dataloader_fn: function that loads data and returns it in the form of list of strings
        """
        self._dataloader_fn = dataloader_fn

    def load_data(self, filename, **dataloader_kwargs):
        self._data = self._dataloader_fn(filename, **dataloader_kwargs)

    def measure(self, model, max_tokens_generated: int=50, batch_size: int=1):
        begin = time.time()
        _ = infer(
            model,
            self._data, 
            max_new_tokens=max_tokens_generated, 
            batch_size=batch_size
        )
        end = time.time()
        inference_time = end - begin
        return inference_time


def compute_gpu_stats(filename):
    df = pd.read_csv(filename)
    stats = {}
    total_mem = float(df[" memory.total [MiB]"].values[0][:-4])
    stats[" memory.total [MiB]"] = total_mem
    for col in [" utilization.gpu [%]", " utilization.memory [%]"]:
        arr = np.array([float(val[1:-2]) for val in df[col].values])
        # remove outliars
        alpha = 0.05
        num_rows = df.shape[0]
        lmargin = int(num_rows * alpha)
        rmargin = -lmargin if lmargin > 0 else num_rows
        arr = np.sort(arr)[lmargin:rmargin]

        stats[col] = arr.mean()
    return stats 
