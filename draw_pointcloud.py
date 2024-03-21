import math
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from .ensemble_FP import load_predictions_from_files, cluster_scene_FP, load_TP_predictions_from_files
from .ensemble_FP import PATH_LIST
import pickle
 
def hsv_to_rgb(h, s, v):
    '''
    Convert HSV to RGB
    h: 0-1
    s: 0-1
    v: 0-1
    Return: (r, g, b) 0-255
    '''
    if s == 0.0: v*=255; return (v, v, v)
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f))))
    v*=255; i%=6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)

def draw_pointcloud(pcd):
    xyz = (pcd.T)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def draw_pointcloud_with_active_FP_areas(pcd, FP_clusters, plot_gt=False, gt=None, plot_FP=False, FP=None):

    xyz = (pcd.T)[:, :3]
    xyz = o3d.utility.Vector3dVector(xyz)

    # compute max bound for each cluster
    obj = []
    cluster_pts_idx = []
    for cluster in FP_clusters:
        max_x,min_x,max_y,min_y,max_z,min_z = -1000,1000,-1000,1000,-1000,1000
        for instance in cluster:
            box_corners = get_box_corners(instance)
            max_x = max(max_x, np.max(box_corners[:,0]))
            max_y = max(max_y, np.max(box_corners[:,1]))
            min_x = min(min_x, np.min(box_corners[:,0]))
            min_y = min(min_y, np.min(box_corners[:,1]))
            min_z = min(min_z, instance[2]-1/2*instance[5])
            max_z = max(max_z, instance[2]+1/2*instance[5])
        max_bound = [max_x, max_y, max_z]
        min_bound = [min_x, min_y, min_z]
        cluster_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cluster_box.color = [1,0,0]
        pts_id_in_cluster = cluster_box.get_point_indices_within_bounding_box(xyz)
        cluster_pts_idx.append(pts_id_in_cluster)
        obj.append(cluster_box)

    # plot gt data and opacity based on recall rate
    if plot_gt:
        for data in gt:
            bbox, confidence = data
            location = bbox[0:3]
            size = bbox[3:6]
            heading_angle = bbox[6] + math.pi # quaternions radians to angle in radians
            rotational_matrix = np.array([[math.cos(heading_angle), -math.sin(heading_angle), 0],
                                            [math.sin(heading_angle), math.cos(heading_angle), 0],
                                            [0, 0, 1]])
            obj.append(o3d.geometry.OrientedBoundingBox(location, rotational_matrix, size))
            obj[-1].color = [0,0,1]

    if plot_FP:
        for i,bbox in enumerate(FP):
            location = bbox[0:3]
            size = bbox[3:6]
            heading_angle = bbox[6] + math.pi
            rotational_matrix = np.array([[math.cos(heading_angle), -math.sin(heading_angle), 0],
                                            [math.sin(heading_angle), math.cos(heading_angle), 0],
                                            [0, 0, 1]])
            obj.append(o3d.geometry.OrientedBoundingBox(location, rotational_matrix, size))
            r, g, b = hsv_to_rgb(0.5, 1, 1)
            obj[-1].color = [r/255.0,g/255.0,b/255.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = xyz
    # set point colors in each cluster
    colors = np.zeros((len(xyz), 3))
    for i, idx in enumerate(cluster_pts_idx):
        colors[idx] = np.random.rand(3)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    obj.append(pcd)
    o3d.visualization.draw_geometries(obj)

def draw_active_FP_areas_points(pcd, FP_clusters):

    xyz = (pcd.T)[:, :3]
    xyz = o3d.utility.Vector3dVector(xyz)
    # compute max bound for each cluster
    obj = []
    cluster_pts_idx = []
    for cluster in FP_clusters:
        max_x,min_x,max_y,min_y,max_z,min_z = -1000,1000,-1000,1000,-1000,1000
        for instance in cluster:
            box_corners = get_box_corners(instance)
            max_x = max(max_x, np.max(box_corners[:,0]))
            max_y = max(max_y, np.max(box_corners[:,1]))
            min_x = min(min_x, np.min(box_corners[:,0]))
            min_y = min(min_y, np.min(box_corners[:,1]))
            min_z = min(min_z, instance[2]-1/2*instance[5])
            max_z = max(max_z, instance[2]+1/2*instance[5])
        max_bound = [max_x, max_y, max_z]
        min_bound = [min_x, min_y, -2]
        cluster_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cluster_box.color = [1,0,0]
        cluster_pts_idx.append(cluster_box.get_point_indices_within_bounding_box(xyz))
        obj.append(cluster_box)

    for i, idx in enumerate(cluster_pts_idx):
        obj.append(o3d.geometry.PointCloud())
        obj[-1].points = o3d.utility.Vector3dVector((pcd.T[:,:3])[idx])
        obj[-1].paint_uniform_color(np.random.rand(3))
    o3d.visualization.draw_geometries(obj)

def get_box_corners(bbox) -> np.ndarray:
    heading_angle = bbox[6]*180/math.pi + 180 # quaternions radians to angle in degrees
    location = (bbox[0], bbox[1])
    rect = (location,(bbox[3], bbox[4]), heading_angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)    
    return box

def visualize_cluster_TP_confidences(FP_clusters, TPs, show_results=True, scene = 0):
    """
    FP_clusters: n_clusters x n_instances x 8
    TPs: n_models x ([model_ids], [bboxes], [confidences])
    """
    
    categories = ["PP_MULTIHEAD","SECOND_MULTIHEAD","PP_CENTERPT","CENTERPT_VOX01","CENTERPT_VOX0075","VOXELNEXT","TRANSFUSION","BEVFUSION"]
    model_confidences = {}
    for i in range(len(FP_clusters)):
        model_confidences[f"Cluster {i}"] = [0 for _ in categories]
    model_confidences[f"TP"] = [0 for _ in categories]
    
    for i,cluster in enumerate(FP_clusters):
        confidence_sum = []
        for _ in range(8):
            confidence_sum.append([])
        for instance in cluster:
            confidence_sum[int(instance[-1])].append(instance[-2])
        confidence_sum = [np.mean(np.array(e)) if len(e) > 0 else 0 for e in confidence_sum]
        model_confidences[f"Cluster {i}"] = confidence_sum
    
    for i,TP in enumerate(TPs):
        model_confidences[f"TP"][i] = np.mean(TP[2])
    
    x = np.arange(len(categories))
    bar_width = 0.04
    multiplier = 0
    plt.close('all')
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(10, 6)
    for keys, values in model_confidences.items():
        offset = bar_width*multiplier
        if keys == "TP":
            rects = ax.bar(x + offset, values, bar_width, label=keys, color='r')
        else:
            rects = ax.bar(x + offset, values, bar_width, label=keys, alpha=0.5)
        multiplier += 1
    
    ax.set_ylabel('Confidence')
    ax.set_title('Cluster confidences by models')
    ax.set_xticks(x + bar_width, categories, fontsize=5)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', ncol=2, fontsize=5)

    if show_results:
        plt.show()
    else:
        plt.savefig(f"results/confidences_FP_TP_clusters/scene_{scene}.png")

def draw_num_pts_confidence_correlation(pcd, FP_clusters, show_results=True, scene = 0):
    categories = ["PP_MULTIHEAD","SECOND_MULTIHEAD","PP_CENTERPT","CENTERPT_VOX01","CENTERPT_VOX0075","VOXELNEXT","TRANSFUSION","BEVFUSION"]

    xyz = (pcd.T)[:, :3]
    xyz = o3d.utility.Vector3dVector(xyz)
    # compute max bound for each cluster
    cluster_pts_num = [[] for _ in range(8)]
    cluster_confidence = [[] for _ in range(8)]
    for cluster in FP_clusters:
        max_x,min_x,max_y,min_y,max_z,min_z = -1000,1000,-1000,1000,-1000,1000
        for instance in cluster:
            box_corners = get_box_corners(instance)
            max_x = max(max_x, np.max(box_corners[:,0]))
            max_y = max(max_y, np.max(box_corners[:,1]))
            min_x = min(min_x, np.min(box_corners[:,0]))
            min_y = min(min_y, np.min(box_corners[:,1]))
            min_z = min(min_z, instance[2]-1/2*instance[5])
            max_z = max(max_z, instance[2]+1/2*instance[5])
        max_bound = [max_x, max_y, max_z]
        min_bound = [min_x, min_y, -2]
        cluster_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pts_in_cluster = cluster_box.get_point_indices_within_bounding_box(xyz)
        for instance in cluster:
            cluster_pts_num[int(instance[-1])].append(len(pts_in_cluster))
            cluster_confidence[int(instance[-1])].append(instance[-2])

    plt.close('all')
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(10, 6)
    for i in range(8):
        ax.scatter(cluster_pts_num[i], cluster_confidence[i], label=categories[i], alpha=0.5)
    ax.set_xlabel('Number of points in cluster')
    ax.set_ylabel('Confidence')
    ax.set_title('Number of points in cluster vs confidence')
    ax.legend(loc='upper left', ncol=2, fontsize=5)
    
    if show_results:
        plt.show()
    else:
        plt.savefig(f"results/num_pts_confidence_correlation/scene_{scene}.png")

def draw_high_confidence_cluster(pcd, clusters, TPs, min_points=5):
    xyz = (pcd.T)[:, :3]
    xyz = o3d.utility.Vector3dVector(xyz)
    obj = []
    pc_points = []
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    for i,cluster in enumerate(clusters):
        cluster_confidence = []
        for _ in range(8):
            cluster_confidence.append([])
        for instance in cluster:
            cluster_confidence[int(instance[-1])].append(instance[-2])
        cluster_confidence = np.array([np.mean(np.array(e)) if len(e) > 0 else 0 for e in cluster_confidence])
        TP_confidence = np.zeros(8)
        for i,TP in enumerate(TPs):
            TP_confidence[i] = np.mean(TP[2])
        if any([cluster_confidence[i] > TP_confidence[i]*0.75 for i in range(8)]):
            max_x,min_x,max_y,min_y,max_z,min_z = -1000,1000,-1000,1000,-1000,1000
            for instance in cluster:
                box_corners = get_box_corners(instance)
                max_x = max(max_x, np.max(box_corners[:,0]))
                max_y = max(max_y, np.max(box_corners[:,1]))
                min_x = min(min_x, np.min(box_corners[:,0]))
                min_y = min(min_y, np.min(box_corners[:,1]))
                min_z = min(min_z, instance[2]-1/2*instance[5])
                max_z = max(max_z, instance[2]+1/2*instance[5])
            max_bound = [max_x, max_y, max_z]
            min_bound = [min_x, min_y, min_z]
            cluster_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pts_ids_inside_box = cluster_box.get_point_indices_within_bounding_box(xyz)
            if len(pts_ids_inside_box) < min_points:
                continue
            obj.append(cluster_box)
            obj[-1].color = [1,0,0]
            pc_points.append(o3d.geometry.PointCloud())
            pc_points[-1].points = o3d.utility.Vector3dVector((pcd.T[:,:3])[pts_ids_inside_box])
            pc_points[-1].paint_uniform_color(np.array([1,0,0]))
            print(f"confidence: {cluster_confidence},\n TP confidence: {TP_confidence}")
    
    obj.extend(pc_points)
    if len(obj) > 0:
        o3d.visualization.draw_geometries(obj)
    else:
        print("No high confidence clusters found.")

def FPs_per_model_per_scene(FPs, scene=0, save_results=True):
    num_FPs = [len(e[scene][1]) for e in FPs]
    models = ["PP_MULTIHEAD","SECOND_MULTIHEAD","PP_CENTERPT","CENTERPT_VOX01","CENTERPT_VOX0075","VOXELNEXT","TRANSFUSION","BEVFUSION"]
    plt.close('all')
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(10, 6)
    ax.bar(models, num_FPs)
    ax.set_xlabel('Model')
    ax.set_ylabel('Number of FPs')
    ax.set_title('Number of FPs per model per scene')
    if save_results:
        plt.savefig(f"results/FPs_per_model_per_scene/scene_{scene}.png")

FPs, scene_tokens = load_predictions_from_files(PATH_LIST, filter_class="car", filter_IoU=0)
scenes = np.arange(0,81)
TPs = load_TP_predictions_from_files(PATH_LIST, filter_class="car", filter_IoU=0)
for scene in scenes:

    ######################## Plot per model FP number ########################
    # FPs_per_model_per_scene(FPs, scene)

    ######################## cluster points ########################
    clusters = cluster_scene_FP(FPs, scene, print_result=False, visualize=False, save_results=False)
    
    ######################## Draw pointcloud with active FP areas ########################

    pcd = find_corresponding_pcd(scene_tokens[scene],DATA_INFO_PATH,PCD_PATH)
    with open(PATH_LIST[0], 'rb') as f:
        pred = pickle.load(f)
    gt_boxes = pred[scene]['gt_box_coverage']
    draw_pointcloud_with_active_FP_areas(pcd, clusters, plot_gt=True, gt=gt_boxes, plot_FP=True, FP=[x for e in FPs for x in e[scene][1]])
    draw_active_FP_areas_points(pcd, clusters)

    ######################## Visualize cluster TP confidences differences ########################

    # TPs = load_TP_predictions_from_files(PATH_LIST, filter_class="car", filter_IoU=0)
    # visualize_cluster_TP_confidences(clusters, [x[scene] for x in TPs[0]], show_results=False,scene=scene)

    ######################## Draw number of points in cluster vs confidence ########################

    # pcd = find_corresponding_pcd(scene_tokens[scene],DATA_INFO_PATH,PCD_PATH)
    # draw_num_pts_confidence_correlation(pcd, clusters, show_results=False, scene=scene)

    ######################## Draw cluster points for high confidence FP cluster ########################

    # pcd = find_corresponding_pcd(scene_tokens[scene],DATA_INFO_PATH,PCD_PATH)
    # draw_high_confidence_cluster(pcd, clusters, [x[scene] for x in TPs[0]])
    # #########################################################################################
