import numpy as np
import pickle
import matplotlib.pyplot as plt

DATASET_GT_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/data/nuscenes_dbinfos_10sweeps_withvelo.pkl"
DATA_PKL="/Users/daniel/Documents/code/python/research/evaluation_research/data/nuscenes_infos_10sweeps_val.pkl"
TRANSFUSION_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/transfusion_lidar/default/eval/epoch_2119/val/default/result.pkl"
VOXELNEXT_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_voxel0075_voxelnext/default/eval/epoch_1/val/default/result.pkl"
CENTERPT_VOX01_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_voxel01_res3d_centerpoint/default/eval/epoch_6454/val/default/result.pkl"
CENTERPT_VOX0075_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_voxel0075_res3d_centerpoint/default/eval/epoch_6648/val/default/result.pkl"
SECOND_MULTIHEAD_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_second_multihead/default/eval/epoch_6229/val/default/result.pkl"
PP_MULTIHEAD_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_pp_multihead/default/eval/epoch_5823/val/default/result.pkl"
PP_CENTERPT_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_6070/val/default/result.pkl"

path_list=[PP_MULTIHEAD_PATH,SECOND_MULTIHEAD_PATH,PP_CENTERPT_PATH,CENTERPT_VOX01_PATH,CENTERPT_VOX0075_PATH,VOXELNEXT_PATH,TRANSFUSION_PATH]

def compute_mAP(pred, threshold, print=False):

    CLASS_NAMES = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    class_count = dict((n,0) for n in CLASS_NAMES)
    class_pred_num = dict((n,0) for n in CLASS_NAMES)

    for i in range(len(pred)):
        for key, val in class_count.items():
            filtered_pred = [pred[i]['IoU_gt_record'][j] for j,e in enumerate(pred[i]['name']) if e==key]
            class_pred_num[key] += len(filtered_pred)
            class_count[key] += len([e for e in filtered_pred if e > threshold])
    if print:
        print(f"mAP per class:")
        for k in class_count:
            print(f"{k}: {class_count[k]}/{class_pred_num[k]}={round(class_count[k]/class_pred_num[k],2)}")
        print(f"mAP = {round(100*sum(class_count.values())/sum(class_pred_num.values()),2)}")

def compute_FNs(pred, threshold, print=False):

    FN_count = 0
    total_gt = 0

    for i in range(len(pred)):
        FN_list = [e for _,e in pred[i]['gt_box_coverage'] if e <= threshold]
        FN_count += len(FN_list)
        total_gt += len(pred[i]['gt_box_coverage'])
    if print:
        print(f"{FN_count} out of {total_gt} objects missed")

gt_recall_record = []
for path in path_list:

    print(f"Reading results from: {path}\n")
    with open(path, 'rb') as f:
        pred = pickle.load(f)
    
    gt_recall_record.append([e['gt_box_coverage'] for e in pred])

    compute_mAP(pred, 0.5)
    compute_FNs(pred, 0.5)

    print ("\n")

FN_pred_cnt = []
for scene in gt_recall_record[0]:
    is_FN = [1 if score==0 else 0 for (_,score) in scene]
    FN_pred_cnt.append(is_FN)

for i in range(1,len(gt_recall_record)):
    for j,scene in enumerate(gt_recall_record[i]):
        is_FN = [1 if score==0 else 0 for (_,score) in scene]
        previous_FN_cnt = FN_pred_cnt[j]
        FN_pred_cnt[j] = [sum(x) for x in zip(is_FN, previous_FN_cnt)]

FN_pred_cnt = [e for el in FN_pred_cnt for e in el if e != 0] # flatten list

# counts, bins, bars = plt.hist(FN_pred_cnt, bins=np.arange(0.5, len(path_list)+1.5, 1), rwidth=0.7)
# plt.show()

# print(f"\n\nshared_FN:\n{counts} from {bins}")
# print('done') 




