# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import numpy as np
import open3d as o3d
from copy import deepcopy
 
   
if __name__ == '__main__':
        file_path = 'bun000.ply'
        source = o3d.io.read_triangle_mesh(file_path)
        points1 = np.array(source.vertices) #转为矩阵
        file_path = 'bun045.ply'
        target = o3d.io.read_triangle_mesh(file_path)
        points2 = np.array(target.vertices) #转为矩阵

        threshold = 0.2 #距离阈值
        trans_init = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0,   0.0],
                                [0.0, 0.0, 1.0, 0],
                                [0.0, 0.0, 0.0,   1.0]])
        #计算两个重要指标，fitness计算重叠区域（内点对应关系/目标点数）。越高越好。
        #inlier_rmse计算所有内在对应关系的均方根误差RMSE。越低越好。
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(points1)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(points2)
        print("Initial alignment")
        icp = o3d.pipelines.registration.registration_icp(
                source, target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(icp)
        icp_pcd = deepcopy(source)
        icp_pcd.transform(icp.transformation)
        print(icp.transformation)
        points1 = np.array(icp_pcd.points)
        points3 = np.concatenate((points1, points2), axis=0)
        con_pcd = o3d.geometry.PointCloud()
        con_pcd.points = o3d.utility.Vector3dVector(points3)
        con_pcd1 = con_pcd.uniform_down_sample(2)

        print(source)
        print(target)
        print(con_pcd)
        print(con_pcd1)

        source.paint_uniform_color([0, 1, 0])#指定显示为绿色
        target.paint_uniform_color([0, 0, 1])#指定显示为蓝色
        target.translate((0.2, 0, 0))#整体沿X轴平移
        con_pcd.paint_uniform_color([1, 0, 0])#指定显示为红色
        con_pcd.translate((0.4, 0, 0))#整体沿X轴平移
        con_pcd1.paint_uniform_color([1, 1, 0])#指定显示为红色
        con_pcd1.translate((0.6, 0, 0))#整体沿X轴平移
        
      
        o3d.visualization.draw_geometries([source, target, con_pcd, con_pcd1], #点云列表
                                        window_name="点云ICP配准",
                                        point_show_normal=False,
                                        width=800,  # 窗口宽度
                                        height=600)  # 窗口高度