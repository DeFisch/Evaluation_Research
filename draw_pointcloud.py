import open3d as o3d
import numpy as np
from .utils import *
 
def draw_pointcloud(pcd):
    xyz = (pcd.T)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

pcd = find_corresponding_pcd("ca9a282c9e77460f8360f564131a8af5",DATA_INFO_PATH,PCD_PATH)
draw_pointcloud(pcd)

