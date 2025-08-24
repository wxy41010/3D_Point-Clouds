# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
 
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
    points = np.array(pcd.points)
    result = KMeans(n_clusters=8).fit(points)
    #各个类别中心
    center = result.cluster_centers_
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label, 3))/255.
    colors = colors[labels]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
 
    # 点云显示
    o3d.visualization.draw_geometries([pcd], #点云列表
                                      window_name="Kmeans点云聚类",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度