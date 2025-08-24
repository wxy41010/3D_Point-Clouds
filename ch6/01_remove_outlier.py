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
    pcd = pcd.uniform_down_sample(50)#每50个点采样一次
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
 
    #剔除无效值
    pcd1 = deepcopy(pcd)
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    pcd1.translate((20, 0, 0)) #整体进行x轴方向平移
    pcd1 = pcd1.remove_non_finite_points(True, True)#剔除无效值
    print(pcd1)
 
    #统计方法剔除
    pcd2 = deepcopy(pcd)
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为蓝色
    pcd2.translate((-20, 0, 0)) #整体进行x轴方向平移
    res = pcd2.remove_statistical_outlier(20, 0.5)#统计方法剔除
    pcd2 = res[0]#返回点云，和点云索引
    print(pcd2)
 
    #半径方法剔除
    pcd3 = deepcopy(pcd)
    pcd3.paint_uniform_color([1, 0, 0])#指定显示为蓝色
    pcd3.translate((0, 20, 0)) #整体进行y轴方向平移
    res = pcd3.remove_radius_outlier(nb_points=20, radius=2)#半径方法剔除
    pcd3 = res[0]#返回点云，和点云索引
    print(pcd3)
 
    # # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3], #点云列表
                                      window_name="离群点剔除",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
 