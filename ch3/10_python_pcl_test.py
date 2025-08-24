# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import pcl
import numpy as np
import pcl.pcl_visualization as viz

if __name__ == '__main__':
    pcd = pcl.load("rabbit.pcd")
    pcd = pcl.load("bun_zipper.ply")
    print('pcd shape: ', np.array(pcd).shape)
    vizcolor = viz.PointCloudColorHandleringCustom(pcd, 0, 255, 0)
    vs=viz.PCLVisualizering
    vizer=viz.PCLVisualizering()
    vs.AddPointCloud_ColorHandler(vizer, pcd, vizcolor, id=b'cloud', viewport=0)
    while not vs.WasStopped(vizer):
        vs.Spin(vizer)