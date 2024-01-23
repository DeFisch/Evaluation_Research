
import numpy as np

path = "data/nuscenes/v1.0-mini/gt_database_10sweeps_withvelo/0_barrier_9.bin"

content = np.fromfile(path)

print("done")