
import cv2
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter

# modify accoding to your results
DATASET_GT_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/data/nuscenes_dbinfos_10sweeps_withvelo.pkl"
DATA_PKL="/Users/daniel/Documents/code/python/research/evaluation_research/data/nuscenes_infos_10sweeps_val.pkl"
TRANSFUSION_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/transfusion_lidar/default/eval/epoch_2119/val/default/result.pkl"
VOXELNEXT_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_voxel0075_voxelnext/default/eval/epoch_1/val/default/result.pkl"
CENTERPT_VOX01_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_voxel01_res3d_centerpoint/default/eval/epoch_6454/val/default/result.pkl"
CENTERPT_VOX0075_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_voxel0075_res3d_centerpoint/default/eval/epoch_6648/val/default/result.pkl"
SECOND_MULTIHEAD_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_second_multihead/default/eval/epoch_6229/val/default/result.pkl"
PP_MULTIHEAD_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_pp_multihead/default/eval/epoch_5823/val/default/result.pkl"
PP_CENTERPT_PATH="/Users/daniel/Documents/code/python/research/evaluation_research/results/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_6070/val/default/result.pkl"

PATH_LIST=[PP_MULTIHEAD_PATH,SECOND_MULTIHEAD_PATH,PP_CENTERPT_PATH,CENTERPT_VOX01_PATH,CENTERPT_VOX0075_PATH,VOXELNEXT_PATH,TRANSFUSION_PATH]

# load result pickle file one by one
def load_predictions_from_files(paths, filter_class="None", filter_IoU=0):
    model_predictions = []
    for path in paths:

        with open(path, 'rb') as f:
            pred = pickle.load(f)
       
        model_pred = []
        scene_tokens = []
        for scene in pred:
            scene_tokens.append(scene['metadata']['token'])
            IoU = scene['IoU_gt_record']
            classes = list(scene['name'])
            bboxes = list(scene['boxes_lidar'])
            scores = list(scene['score'])
            pop_idx_list = []
            for i, pred_class in enumerate(classes):
                if (filter_class is not None and pred_class != filter_class) or IoU[i] > filter_IoU:
                    pop_idx_list.append(i)
            pop_idx_list = reversed(pop_idx_list)
            for idx in pop_idx_list:
                IoU.pop(idx)
                bboxes.pop(idx)
                scores.pop(idx)
            model_pred.append((IoU,bboxes,scores))
        model_predictions.append(model_pred)

    return model_predictions, scene_tokens

def plot_pred_centerpoints_per_scene(scene_preds, gt=None):
    # plot FP data per model using different color
    for scene_pred in scene_preds:
        iou, bboxes = scene_pred
        centerpoints_x = [e[0] for e in bboxes]
        centerpoints_y = [e[1] for e in bboxes]
        plt.plot(centerpoints_x,centerpoints_y,'o',alpha=0.7)
    
    # plot gt data and opacity based on recall rate
    if gt is not None:
        for data in gt:
            bbox, confidence = data
            plt.plot(bbox[0],bbox[1],"rs",markersize=5, alpha=0.5+0.5*confidence)
    
    plt.plot(0,0,"+",markersize=5)
    plt.show()

def plot_FP_gt_scene(preds, scene=0, draw_gt=True):
    scene_preds = [e[scene] for e in preds]
    if draw_gt:
        with open(PATH_LIST[0], 'rb') as f:
            pred = pickle.load(f)
        gt_boxes = pred[scene]['gt_box_coverage']
    plot_pred_centerpoints_per_scene(scene_preds,gt_boxes) \
        if draw_gt else plot_pred_centerpoints_per_scene(scene_preds)


def cluster_scene_FP(preds, scene=0, eps=2, print_result=False, visualize=False):
    scene_preds = [e[scene][1] for e in preds] # 7 x npreds
    pred_model = []
    flattened_preds = []
    for i in range(len(scene_preds)):
        for pred in scene_preds[i]:
            flattened_preds.append(pred) # get bev coord of bbox
            pred_model.append(i)
    flattened_preds = np.array(flattened_preds)
    db = DBSCAN(eps, min_samples=5).fit(flattened_preds[:,0:2])
    labels = db.labels_
    if print_result:
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"cluster results with a distance limit of {eps}")
        for i in range(-1, n_clusters_):
            num_points_in_cluster = sum([1 for e in labels if e == i])
            corresponding_model = [pred_model[j] for j in range(len(labels)) if labels[j]==i]
            num_unique_models = len(Counter(corresponding_model).keys())
            print(f"cluster {i}: {num_points_in_cluster} FPs, predicted by {num_unique_models} models.")
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    points_in_cluster = []
    for i in range(n_clusters_):
        points_in_cluster.append([flattened_preds[j] for j in range(len(labels)) if labels[j]==i])
    if visualize:
        for cluster in points_in_cluster:
            plt.plot(np.array(cluster)[:,0],np.array(cluster)[:,1], "o", alpha=1)
        plt.show()
    else:
        plt.savefig(f"results/cluster_scene{scene}.png")
    return points_in_cluster

def FP_box_bev(FP_preds, img_dim:int, scene=0, display=False):
    scene_preds = [e[scene][1] for e in FP_preds] # 7 x npreds
    disp = np.zeros((img_dim,img_dim,3))
    scale = int(img_dim/150) # nuscenes pc range 150m max
    
    for model_preds in scene_preds:
        for pred in model_preds:
            layer = np.zeros((img_dim,img_dim,3))
            heading_angle = pred[6]*180/math.pi + 180 # quaternions radians to angle in degrees
            location = (pred[0]*scale+int(img_dim/2), pred[1]*scale+int(img_dim/2))
            rect = (location,(pred[3]*scale, pred[4]*scale), heading_angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(layer,[box],0,(0,0,1),thickness=cv2.FILLED)
            disp += layer
    plt.pcolormesh(disp[:,:,2])
    if display:
        plt.show()
    else:
        plt.savefig(f"results/FP_scene_{scene}.png")

    return disp

def gt_box_bev(gt, img_dim:int, scene=0, display=False):
    gt = [e[0] for e in gt]
    disp = np.zeros((img_dim,img_dim,3))
    scale = int(img_dim/150) # nuscenes pc range 150m max
    
    for bbox in gt:
        if bbox[9] != 1.0: # filter out all other classes
            continue
        layer = np.zeros((img_dim,img_dim,3))
        heading_angle = bbox[6]*180/math.pi + 180 # quaternions radians to angle in degrees
        location = (bbox[0]*scale+int(img_dim/2), bbox[1]*scale+int(img_dim/2))
        rect = (location,(bbox[3]*scale, bbox[4]*scale), heading_angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(layer,[box],0,(0,0,1),thickness=cv2.FILLED)
        disp += layer
    plt.pcolormesh(disp[:,:,2])
    if display:
        plt.show()
    else:
        plt.savefig(f"results/gt_scene_{scene}.png")
    
    return disp


    

FPs,_ = load_predictions_from_files(PATH_LIST, filter_class="car", filter_IoU=1)
scene = 10

def plot_combined_FP_gt(scene=scene):
    with open(PATH_LIST[0], 'rb') as f:
        pred = pickle.load(f)
    gt_boxes = pred[scene]['gt_box_coverage']
    img_dim = 1200
    FP_map = FP_box_bev(FPs,img_dim,scene=scene)
    gt_map = gt_box_bev(gt_boxes,img_dim,scene=scene)
    combined = FP_map-gt_map
    plt.pcolormesh(combined[:,:,2])
    plt.savefig(f"results/combined_scene{scene}.png")
    plt.clf()

cluster_scene_FP(FPs, scene=scene, visualize=False)
