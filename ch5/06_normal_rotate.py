# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from copy import deepcopy
 
def viz_matplot(points, ax):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    ax.scatter(x,   # x
               y,   # y
               z,   # z
               c=z, # height data for color
               cmap='rainbow',
               marker=".")
    
 
def pcd_rotate_normal(pointcloud, n0, n1):
    """
    Parameters
    ----------
    pointcloud : open3d PointCloud, 输入点云
    n0 : array, 1x3, 原始法向量
    n1 : array, 1x3, 目标法向量
    Returns
    -------
    pcd : open3d PointCloud, 旋转后点云
    """
    pcd = deepcopy(pointcloud)
    n0_norm2 = np.sqrt(sum(n0 ** 2))
    n1_norm2 = np.sqrt(sum(n1 ** 2))
    theta = np.arccos(sum(n0 * n1) / n0_norm2 / n1_norm2)
    r_axis = np.array([n1[2]*n0[1]-n0[2]*n1[1], n0[2]*n1[0]-n1[2]*n0[0], n0[0]*n1[1]-n1[0]*n0[1]])
    r_axis = r_axis * theta / np.sqrt(sum(r_axis ** 2))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    R = pcd.get_rotation_matrix_from_axis_angle(r_axis.T)
    pcd.rotate(R)
    return pcd
 
if __name__ == '__main__':
    #生成平面点云
    x = np.arange(301).reshape(-1, 1).repeat(100, 0) / 100.
    y = np.arange(301).reshape(-1, 1).repeat(100, 1).T.reshape(-1, 1) / 100.
    #平面方程
    z = (12 - 4*x -4*y) / 3
    p = np.concatenate((x, y, z), 1)
    p = p[np.where(p[:,2]>=0)]
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    viz_matplot(p, ax)
    #原始法向量
    n0 = np.array([4, 4, 3])
    #目标法向量
    n1 = np.array([0, 0, 1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd = pcd_rotate_normal(pcd, n0, n1)
    p = np.array(pcd.points)
    #旋转后z的值应该基本相等
    print('min Z: ', np.min(p[:, -1]), 'max Z: ', np.max(p[:, -1]))
    ax = fig.add_subplot(122, projection='3d')
    viz_matplot(p, ax)
    ax.axis()
    plt.show()