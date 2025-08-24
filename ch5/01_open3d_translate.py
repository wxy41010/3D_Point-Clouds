# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
from copy import deepcopy
import  open3d as o3d
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    print(pcd)
    pcd1 = deepcopy(pcd)
    #x方向平移
    pcd1.translate((20,0,0), relative=True)
    pcd2 = deepcopy(pcd)
    #y方向平移
    pcd2.translate((0,20,0), relative=True)
    #z方向平移
    pcd3 = deepcopy(pcd)
    pcd3.translate((0,0,20), relative=True)
    pcd4 = deepcopy(pcd)
    pcd4.translate((20,20,20), relative=True)
    #点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3, pcd4], #点云列表
                                      window_name="点云平移",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
 