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
    pcd.translate((0, 10, 0))#原始点云的质心为(0, 0, 0)，将其平移到(0, 10, 0)
    print('open3d get_center 质心计算结果', pcd.get_center())
    # 采用矩阵求平均的方法计算质心
    points = np.array(pcd.points)
    center = np.mean(points, axis=0)
    print('矩阵求平均 质心计算结果', center) 
 