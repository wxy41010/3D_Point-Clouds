# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import open3d as o3d
import numpy as np
 
if __name__ == '__main__':
    file_path = 'chair_0001.txt'
    points = np.loadtxt(file_path, delimiter=',')[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(pcd)
    # pcd = pcd.uniform_down_sample(50)#每50个点采样一次
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
    [A, B, C, D] = plane_model
    print(f"Plane equation: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")
    colors = np.array(pcd.colors)
    colors[inliers] = [0, 0, 1]#平面内的点设置为蓝色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd], #点云列表
                                      window_name="RANSAC平面分割",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度