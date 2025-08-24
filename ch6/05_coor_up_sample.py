# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
参考：https://github.com/yanx27/Pointnet_Pointnet2_pytorch
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


    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd1, alpha=2)
    pcd2 = mesh.sample_points_poisson_disk(number_of_points=3000, init_factor=5)
    pcd2.paint_uniform_color([0, 1, 0])#指定显示为绿色
    pcd2.translate((-40, 0, 0)) #整体进行x轴方向平移
    print(pcd2)
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2], #点云列表
                                      window_name="点云上采样",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度
    