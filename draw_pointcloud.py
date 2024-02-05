import math
import cv2
import open3d as o3d
import numpy as np
from .utils import *
from .ensemble_FP import load_predictions_from_files, cluster_scene_FP
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

def draw_pointcloud_with_active_FP_areas(pcd, FP_clusters, plot_gt=False, gt=None, plot_FP=False, FP=None, scores=None):

    xyz = (pcd.T)[:, :3]
    xyz = o3d.utility.Vector3dVector(xyz)

    # compute max bound for each cluster
    obj = []
    cluster_pts_idx = []
    for cluster in FP_clusters:
        max_x,min_x,max_y,min_y,max_z = -1000,1000,-1000,1000,-1000
        for instance in cluster:
            box_corners = get_box_corners(instance)
            max_x = max(max_x, np.max(box_corners[:,0]))
            max_y = max(max_y, np.max(box_corners[:,1]))
            min_x = min(min_x, np.min(box_corners[:,0]))
            min_y = min(min_y, np.min(box_corners[:,1]))
            max_z = max(max_z, instance[5])
        max_bound = [max_x, max_y, max_z]
        min_bound = [min_x, min_y, -2]
        cluster_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cluster_box.color = [1,0,0]
        cluster_pts_idx.append(cluster_box.get_point_indices_within_bounding_box(xyz))
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
            r, g, b = hsv_to_rgb(0.5*(1-scores[i]), 1, 1)
            print(scores[i])
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
        max_x,min_x,max_y,min_y,max_z = -1000,1000,-1000,1000,-1000
        for instance in cluster:
            box_corners = get_box_corners(instance)
            max_x = max(max_x, np.max(box_corners[:,0]))
            max_y = max(max_y, np.max(box_corners[:,1]))
            min_x = min(min_x, np.min(box_corners[:,0]))
            min_y = min(min_y, np.min(box_corners[:,1]))
            max_z = max(max_z, instance[5])
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

FPs, scene_tokens = load_predictions_from_files(PATH_LIST, filter_class="car", filter_IoU=0)
scene = 0
clusters = cluster_scene_FP(FPs, scene, print_result=True, visualize=False)
pcd = find_corresponding_pcd(scene_tokens[scene],DATA_INFO_PATH,PCD_PATH)
with open(PATH_LIST[0], 'rb') as f:
    pred = pickle.load(f)
gt_boxes = pred[scene]['gt_box_coverage']
draw_pointcloud_with_active_FP_areas(pcd, clusters, plot_gt=True, gt=gt_boxes, plot_FP=True, FP=[x for e in FPs for x in e[scene][1]], scores=[x for e in FPs for x in e[scene][2]])
draw_active_FP_areas_points(pcd, clusters)
