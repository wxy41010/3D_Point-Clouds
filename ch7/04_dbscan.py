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
    print(pcd)
    #labels返回聚类成功的类别，-1表示没有分到任何类中的点
    labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=60, print_progress=True))
    #最大值加1相当于共有多少个类别
    num_label = np.max(labels) + 1
    print('类别数量：', max(labels)+1)
    #生成n个类别的颜色，n表示聚类别成功的类，-1表示没有分类成功的类别
    colors = np.random.randint(255, size=(max(num_label, 1), 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
    # 点云显示
    o3d.visualization.draw_geometries([pcd], #点云列表
                                      window_name="DBSCAN聚类",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度