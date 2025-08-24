# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import pcl
import pcl.pcl_visualization as viz
cloud = pcl.load("rabbit.pcd")
vizcolor = viz.PointCloudColorHandleringCustom(cloud, 0, 255, 0)
vs=viz.PCLVisualizering
vizer=viz.PCLVisualizering()
vs.AddPointCloud_ColorHandler(vizer, cloud, vizcolor, id=b'cloud', viewport=0)
while not vs.WasStopped(vizer):
    vs.Spin(vizer)