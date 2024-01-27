
import numpy as np
import json
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

DATA_INFO_PATH = "/Users/daniel/Documents/code/python/research/evaluation_research/data/nuscenes/v1.0-mini/v1.0-mini/sample_data.json"
PCD_PATH = "/Users/daniel/Documents/code/python/research/evaluation_research/data/nuscenes/v1.0-mini"

def find_corresponding_pcd(scene_token, data_json, pcd_path):
    with open(data_json, "r") as f:
        data_info = json.load(f)
    for info in data_info:
        if info["sample_token"] == scene_token and info["filename"].endswith(".bin"):
            filename = info["filename"]
            file_path = os.path.join(pcd_path, filename)
            # from https://github.com/nutonomy/nuscenes-devkit/blob/9b165b1018a64623b65c17b64f3c9dd746040f36/python-sdk/nuscenes/nuscenes.py#L1509C9-L1520C40
            pc = LidarPointCloud.from_file(file_path)
            return pc.points
    return None
