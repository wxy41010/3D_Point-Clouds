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
 
    
    #采用欧拉角进行旋转
    R = pcd.get_rotation_matrix_from_xyz((0, np.pi/2, 0))#绕y轴旋转90°
    #旋转矩阵
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
   
    # 仿射变换
    T = np.array([[1, 0, 0, 20], [0, 1, 1, 20], [0, 0, 1, 0], [0, 0, 0, 1]])
    pcd1 = deepcopy(pcd)
    pcd1.transform(T)
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
 
    # 旋转矩阵R+x方向平移20个单位
    T = np.array([[0, 0, 1, 20], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]) #旋转矩阵R+x方向平移20个单位
    pcd2 = deepcopy(pcd)
    pcd2.transform(T)
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为绿色
 
 
    # y方向平移40个单位，并且缩小3倍
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 40], [0, 0, 1, 0], [0, 0, 0, 3]]) #y方向平移40个单位，并且缩小3倍
    pcd3 = deepcopy(pcd)
    pcd3.transform(T)
    pcd3.paint_uniform_color([1, 0, 0])#指定显示为红色
 
    # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3], #点云列表
                                      window_name="仿射变换",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度