# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import open3d as o3d
import numpy as np
from copy import deepcopy
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = pcd.uniform_down_sample(50)#每50个点采样一次
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
    
    pcd1 = deepcopy(pcd)
    pcd1.translate((20, 0, 0)) #整体进行x轴方向平移20
    mesh1 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd1, alpha=2)
    mesh1.paint_uniform_color([0, 1, 0])#指定显示为绿色
    print(mesh1)
    
    pcd2 = deepcopy(pcd)
    pcd2.translate((-20, 0, 0)) #整体进行x轴方向平移-20
    radius = 0.01 # 搜索半径
    max_nn = 10  # 邻域内用于估算法线的最大点数
    pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    radii = [1, 2]#半径列表
    mesh2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd2, o3d.utility.DoubleVector(radii))
    mesh2.paint_uniform_color([0, 0, 1])
    
    pcd3 = deepcopy(pcd)
    pcd3.translate((0, 20, 0)) #整体进行y轴方向平移20
    pcd3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    mesh3, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd3, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.35)
    mesh3.remove_vertices_by_mask(vertices_to_remove)
    mesh3.paint_uniform_color([1, 0, 0])
    
    pcd4 = deepcopy(pcd)
    pcd4.translate((0, -20, 0)) #整体进行y轴方向平移-30
    pcd4.paint_uniform_color([0, 1, 1])
    mesh4 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd4, voxel_size=1)
    
    o3d.visualization.draw_geometries([pcd, mesh1, mesh2, mesh3, mesh4], #点云列表
                                      window_name="点云重建",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600,
                                      mesh_show_wireframe=True,
                                      mesh_show_back_face=True,
                                      )  # 窗口高度