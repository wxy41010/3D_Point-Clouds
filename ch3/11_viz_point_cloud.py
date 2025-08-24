# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import pcl
import numpy as np
import open3d as o3d
from mayavi import mlab
import matplotlib.pyplot as plt
import pcl.pcl_visualization as viz

def viz_matplot(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z,  cmap='rainbow')
    ax.axis()
    plt.show()


def viz_mayavi(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z, z, mode="point", colormap='spectral', # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # 也可以使用固定的RGB值
                         )
    mlab.show()

def viz_open3d():
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    print(pcd)
    pcd.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    #点云显示
    o3d.visualization.draw_geometries([pcd],#待显示的点云列表
                                      window_name="点云显示",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度

def viz_pypcl(pcd):
    print('pcd shape: ', np.array(pcd).shape)
    vizcolor = viz.PointCloudColorHandleringCustom(pcd, 0, 255, 0)
    vs=viz.PCLVisualizering
    vizer=viz.PCLVisualizering()
    vs.AddPointCloud_ColorHandler(vizer, pcd, vizcolor, id=b'cloud', viewport=0)
    while not vs.WasStopped(vizer):
        vs.Spin(vizer)


if __name__ == '__main__':
    pcd = pcl.load('rabbit.pcd')
    points = np.array(pcd)
    # viz_matplot(points)
    # viz_mayavi(points)
    # viz_open3d()
    viz_pypcl(pcd)

    