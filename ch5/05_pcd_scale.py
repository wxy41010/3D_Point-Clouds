# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 

import open3d as o3d
from copy import deepcopy
import numpy as np
 
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
    print('原始点云质心：', pcd.get_center())
    
    # 采用numpy计算
    points = np.array(pcd.points)
    points = points/2.0#缩小到原来的一半
    points[:, 0] = points[:, 0] + 20#质心平移到x=20处
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    print('数组平移后点云质心：', pcd1.get_center())
 
 
    # 采用scale函数
    pcd2 = deepcopy(pcd)
    pcd2.scale(2.0, (40, 0, 0))#点云放大两倍，质心平移至(-40, 0, 0)
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为绿色
    print('scale缩放后点云质心：', pcd2.get_center())
   
    # 采用仿射变换
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 80], [0, 0, 1, 0], [0, 0, 0, 3]])#点云缩小到1/3，质心平移到(0, 80, 0)
    pcd3 = deepcopy(pcd)
    pcd3.transform(T)
    pcd3.paint_uniform_color([1, 0, 0])#指定显示为红色
    print('仿射变换缩放后点云质心：', pcd3.get_center())
 
   
    # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3], #点云列表
                                      window_name="点云缩放",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度