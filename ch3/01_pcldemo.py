# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import open3d as o3d

if __name__ == '__main__':
    file_path = 'rabbit1.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    # 点云显示
    o3d.visualization.draw_geometries([pcd], #点云列表
                                      window_name="点云示例",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度