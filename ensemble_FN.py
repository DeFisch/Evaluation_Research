import subprocess
import pickle

# modify this to the directory of inference output
INFERENCE_DATA_PATH = "results/"

pkl_paths = subprocess.check_output(["find", INFERENCE_DATA_PATH, "-name", "result.pkl"]) 
pkl_paths = pkl_paths.decode("utf-8").split("\n")

gt_recall_record = []
for path in pkl_paths:
    if len(path) == 0: continue
    with open(path, "rb") as f:
        pred = pickle.load(f)    
    gt_recall_record.append([e['gt_box_coverage'] for e in pred])

