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
    print(pcd.get_center())
 
    pcd1 = deepcopy(pcd)
    #采用欧拉角进行旋转
    R = pcd.get_rotation_matrix_from_xyz((0, np.pi/2, 0))#绕y轴旋转90°
    pcd1.rotate(R, center=(20, 0, 0))#旋转点位于x=20处，若不指定则默认为原始点云质心。
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    print(pcd1.get_center())
    print(R)
 
    #采用旋转向量（轴角）进行旋转
    pcd2 = deepcopy(pcd)
    R = pcd.get_rotation_matrix_from_axis_angle(np.array([0, -np.pi/2, 0]).T)#向量方向为旋转轴，模长等于旋转角度，绕y轴旋转-90°
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为绿色
    pcd2.rotate(R, center=(20, 0, 0))#旋转点位于x=20处，若不指定则默认为原始点云质心。
    print(pcd2.get_center())
    print(R)
   
    #采用四元数进行旋转
    pcd3 = deepcopy(pcd)
    R = pcd.get_rotation_matrix_from_quaternion(np.array([0, 0, 0, 1]).T)#绕x轴旋转180°
    pcd3.paint_uniform_color([1, 0, 0])#指定显示为红色
    pcd3.rotate(R, center=(0, 10, 0))#旋转点位于y=10处，若不指定则默认为原始点云质心。
    print(pcd3.get_center())
    print(R)
    # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3], #点云列表
                                      window_name="点云旋转",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
 