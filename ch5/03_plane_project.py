# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

 
from mayavi import mlab
import numpy as np
 
def viz_mayavi(points):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z,
                          y,          # Values used for Color
                          mode="point",
                          colormap='spectral', # 'bone', 'copper', 'gnuplot'
                          # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                          figure=fig,
                          )
    mlab.show()
 
#定义平面方程Ax+By+Cz+D=0
#以z=0平面为例，即在xy平面上的投影A=0, B=0, C=1(任意值), D=0
#para[0, 0, 1, 0]
def point_project(points, para):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    d = para[0]**2 + para[1]**2 + para[2]**2
    t = -(para[0]*x  + para[1]*y + para[2]*z + para[3])/d
    x = para[0]*t + x
    y = para[1]*t + y
    z = para[2]*t + z
    return np.array([x, y, z]).T
 
#矩阵写法
#定义平面方程Ax+By+Cz+D=0
#以z=0平面为例，即在xy平面上的投影A=0, B=0, C=1(任意值), D=0
#para[0, 0, 1, 0]
def point_project_array(points, para):
    para  = np.array(para)
    d = para[0]**2 + para[1]**2 + para[2]**2
    t = -(np.matmul(points[:, :3], para[:3].T) + para[3])/d
    points = np.matmul(t[:, np.newaxis], para[np.newaxis, :3]) + points[:, :3]
    return points
    
    
if  __name__ == '__main__':
    points = np.loadtxt('airplane_0001.txt', delimiter=',')
    #显示原始点云
    #viz_mayavi(points)
 
    #定义平面方程Ax+By+Cz+D=0
    #以z=0平面为例，即在xy平面上的投影A=0, B=0, C=1(任意值), D=0
    project_pane = [0, 0, 1, 0]
    points_new = point_project_array(points, project_pane)
    viz_mayavi(points_new)
                                      