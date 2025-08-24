# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
参考：https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
 
import open3d as o3d
from copy import deepcopy
import numpy as np
 
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
 
    pcd1 = deepcopy(pcd)
    pcd1.translate((20, 0, 0)) #整体进行x轴方向平移
    points = np.array(pcd1.points)
    points = farthest_point_sample(points, 500)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    print(pcd1)
 
    # # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1], #点云列表
                                      window_name="最远点采样",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
 