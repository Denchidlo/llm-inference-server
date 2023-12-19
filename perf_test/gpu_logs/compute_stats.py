import pandas as pd
import numpy as np

filename = "gpu_log.csv"

df = pd.read_csv(filename)
total_mem = float(df[" memory.total [MiB]"].values[0][:-4])
print(total_mem)
for col in [" utilization.gpu [%]", " utilization.memory [%]"]:
    arr = np.array([float(val[1:-2]) for val in df[col].values])
    # remove outliars
    arr = np.sort(arr)[2:-2]
    print(f"col: {col}, avg value: {arr.mean()}")