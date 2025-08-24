# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 

import open3d as o3d
import numpy as np
from copy import deepcopy
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, estimate_bandwidth, MeanShift, Birch, AffinityPropagation
 
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = pcd.uniform_down_sample(10)#每50个点采样一次
    pcd.paint_uniform_color([0.5, 0.5, 0.5])#指定显示为灰色
    print(pcd)
    
    pcd1 = deepcopy(pcd)
    pcd1.translate((20, 0, 0)) #整体进行x轴方向平移20
    points = np.array(pcd1.points)
    result = OPTICS(min_samples=2, max_eps=5).fit(points)
    #各个类别中心
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label+1, 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd1.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    
    pcd2 = deepcopy(pcd)
    pcd2.translate((-20, 0, 0)) #整体进行x轴方向平移-20
    points = np.array(pcd2.points)
    result = SpectralClustering(n_clusters=8).fit(points)
    #各个类别中心
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label+1, 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    pcd3 = deepcopy(pcd)
    pcd3.translate((0, 20, 0)) #整体进行y轴方向平移20
    points = np.array(pcd3.points)
    result = AgglomerativeClustering(n_clusters=8).fit(points)
    #各个类别中心
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label+1, 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd3.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    pcd4 = deepcopy(pcd)
    pcd4.translate((0, -20, 0)) #整体进行y轴方向平移-20
    points = np.array(pcd4.points)
    #定义搜索半径，也可以直接初始化一个数值
    bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)
    result = MeanShift(bandwidth=bandwidth).fit(points)
    #各个类别中心
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label+1, 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd4.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    pcd5 = deepcopy(pcd)
    pcd5.translate((40, 0, 0)) #整体进行x轴方向平移40
    points = np.array(pcd5.points)
    result = Birch(n_clusters=8).fit(points)
    #各个类别中心
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label+1, 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd5.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    
    pcd6 = deepcopy(pcd)
    pcd6.translate((-40, 0, 0)) #整体进行x轴方向平移-40
    points = np.array(pcd6.points)
    result = AffinityPropagation(preference=-20).fit(points)
    #各个类别中心
    # labels返回聚类成功的类别，从0开始，每个数据表示一个类别
    labels = result.labels_
    #最大值相当于共有多少个类别
    max_label = np.max(labels) + 1 #从0开始计算标签
    print(max(labels))
    #生成k个类别的颜色，k表示聚类成功的类别
    colors = np.random.randint(255, size=(max_label+1, 3))/255.
    colors = colors[labels]
    #没有分类成功的点设置为黑色
    colors[labels < 0] = 0 
    pcd6.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
 
    # 点云显示
    o3d.visualization.draw_geometries([pcd, pcd1, pcd2, pcd3, pcd4, pcd5, pcd6], #点云列表
                                      window_name="点云聚类",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度