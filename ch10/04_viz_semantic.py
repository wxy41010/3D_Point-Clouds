# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import open3d as o3d
import numpy as np
from copy import deepcopy
 
 
if __name__ == '__main__':
    preds = np.loadtxt('Area_5_office_33.txt')
    points = np.load('Area_5_office_33.npy')
    print(preds.shape, points.shape)
    print(set(preds))
    
    #随机生成13个类别的颜色
    colors_0 = np.random.randint(255, size=(13, 3))/255.
 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
   
    #为各个真实标签指定颜色
    colors = colors_0[points[:, -1].astype(np.uint8)]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    #显示预测结果
    pcd1 = deepcopy(pcd)
    pcd1.translate((0, 5, 0)) #整体进行y轴方向平移5
    #为各个预测标签指定颜色
    colors = colors_0[preds.astype(np.uint8)]
    pcd1.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
 
    #显示预测结果和真实结果对比
    pcd2 = deepcopy(pcd)
    pcd2.translate((0, -5, 0)) #整体进行y轴方向平移-5
    preds = preds.astype(np.uint8) == points[:, -1].astype(np.uint8)
    #为各个预测标签指定颜色
    colors = colors_0[preds.astype(np.uint8)]
    pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
 
    # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2], window_name="PointNet++语义分割结果",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度