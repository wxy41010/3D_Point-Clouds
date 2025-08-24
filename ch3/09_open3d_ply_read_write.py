# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import open3d as o3d
import numpy as np

 
if __name__ == '__main__':
    ply = o3d.io.read_triangle_mesh('bun_zipper.ply')
    points = np.array(ply.vertices) #转为矩阵
    print(points.shape)