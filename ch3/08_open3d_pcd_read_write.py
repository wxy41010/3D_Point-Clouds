# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import open3d as o3d
import numpy as np

 
if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud('rabbit.pcd')
    points = np.array(pcd.points) #转为矩阵