# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import open3d as o3d
from copy import deepcopy
 
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
 
    pcd1 = deepcopy(pcd)
    pcd1.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    pcd1.translate((20, 0, 0)) #整体进行x轴方向平移
    pcd1 = pcd1.voxel_down_sample(voxel_size=1)
    print(pcd1)
 
    pcd2 = deepcopy(pcd)
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为绿色
    pcd2.translate((0, 20, 0)) #整体进行y轴方向平移
    res = pcd2.voxel_down_sample_and_trace(1, min_bound=pcd2.get_min_bound()-0.5, max_bound=pcd2.get_max_bound()+0.5, approximate_class=True)
    pcd2 = res[0]
    print(pcd2)
 
    
    # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2], #点云列表
                                      window_name="体素下采样",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
 