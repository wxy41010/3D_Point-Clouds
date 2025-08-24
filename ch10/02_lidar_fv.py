# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import os
import cv2
import numpy as np

def lidar_fv():
    lidar_path = os.path.join('./data/KITTI/training', "velodyne/")
    lidar_file = lidar_path + '/' + '000010' + '.bin'
    #加载雷达数据
    print("Processing: ", lidar_file)
    lidar = np.fromfile(lidar_file, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    v_res = 26.8/64
    h_res = 0.09
    # 转换为弧度
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)
    angels = np.zeros((lidar.shape[0], 2))
    angels[:, 0] = np.arctan2(lidar[:, 1], lidar[:, 0])
    angels[:, 1] = np.arctan2(lidar[:, 2], np.sqrt(lidar[:, 0]**2+lidar[:, 1]**2))
    img_x = angels[:, 0]/h_res_rad
    img_x = img_x.astype(np.int)
    img_x = img_x - min(img_x)
    img_y = angels[:, 1]/v_res_rad
    img_y = img_y.astype(np.int)
    print(min(img_y), max(img_y))
    img_y = img_y - min(img_y)
    img_y = max(img_y) - img_y
    print(min(img_x), max(img_x))
    print(min(img_y), max(img_y))
    fv_img = np.zeros((max(img_y)+1, max(img_x)+1))
    fv_img[img_y, img_x] = lidar[:, -1]
    print(fv_img.shape)
    cv2.namedWindow('FV', 0)
    cv2.imshow('FV', fv_img)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    lidar_fv()