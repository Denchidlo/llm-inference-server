from batched_inference.pipeline import load_data, store_data, infer_batch

in_file = "in.json"
out_file = "out.json"
data = load_data(in_file)
out = infer_batch(data)
store_data(out, out_file)