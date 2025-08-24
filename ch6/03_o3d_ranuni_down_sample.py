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
 
    pcd1 = deepcopy(pcd)
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    pcd1.translate((20, 0, 0)) #整体进行x轴方向平移
    pcd1 = pcd1.uniform_down_sample(100)#每100个点采样一次
    print(pcd1)
 
    pcd2 = deepcopy(pcd)
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为绿色
    pcd2.translate((0, 20, 0)) #整体进行y轴方向平移
    pcd2 = pcd2.random_down_sample(0.1)#采1/10的点云
    print(pcd2)
 
    #自定义随机采样
    pcd3 = deepcopy(pcd)
    pcd3.translate((-20, 0, 0)) #整体进行x轴方向平移
    points = np.array(pcd3.points)
    n = np.random.choice(len(points), 500, replace=False) #s随机采500个数据，这种随机方式也可以自己定义
    pcd3.points = o3d.utility.Vector3dVector(points[n])
    pcd3.paint_uniform_color([1, 0, 0])#指定显示为红色
    print(pcd3)
    
    # # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3], #点云列表
                                      window_name="均匀随机采样",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
 