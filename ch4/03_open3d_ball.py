# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import open3d as o3d
import numpy as np
 
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
    #pcd需要有法向量
    pcd.estimate_normals()
    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 6 * avg_dist   
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))
    
    # 点云显示
    o3d.visualization.draw_geometries([mesh], #点云列表
                                      window_name="点云三角化",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600,
                                      mesh_show_wireframe=True,#显示三角网格线
                                      mesh_show_back_face=True,#显示法向量垂直被部的平面
                                      )  # 窗口高度
 
                                      