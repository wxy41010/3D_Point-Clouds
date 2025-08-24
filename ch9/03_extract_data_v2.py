# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
二维、三维、量化投资等深度学习算法分享。
"""

import os
import cv2
import h5py
import shutil
from scipy.io import loadmat, savemat
import numpy as np
from tqdm import tqdm

def read3dPoints(data):
    depthpath = data['depthpath']
    rgbpath   = data['rgbpath']
    depthVis  = cv2.imread(depthpath, -1)
    imsize    = depthVis.shape
    depthInpaint = (depthVis >> 3) | (depthVis << 13)
    depthInpaint = depthInpaint / 1000.
    depthInpaint[depthInpaint >8] = 8
    
    # K is [fx 0 cx; 0 fy cy; 0 0 1];  
    # for uncrop image crop =[1,1];
    # imageName is the full path to image
    K = data['K']
    cx = K[0, 2]
    cy = K[1, 2]  
    fx = K[0, 0]
    fy = K[1, 1]
    invalid = depthInpaint==0
    imageName = rgbpath
    if len(imageName) > 0:
        im = cv2.imread(imageName, -1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rgb = im.astype(np.float)
    else:
        ch1 = np.zeros(imsize, dtype=np.float)
        ch2 = np.ones(imsize, dtype=np.float)
        rgb = np.concatenate([ch1[:, :, None], ch2[:, :, None], ch1[:, :, None]], 2)
    rgb = rgb.reshape(-1, 3)
    
    #3D points
    x, y = np.meshgrid(range(imsize[1]), range(imsize[0]))
    x3 = (x - cx) * depthInpaint / fx
    y3 = (y - cy) * depthInpaint / fy
    z3 = depthInpaint
    points3dMatrix = np.concatenate([x3[:, :, None], z3[:, :, None], -y3[:, :, None]], 2)
    invalid_ = np.concatenate([invalid[:, :, None], invalid[:, :, None], invalid[:, :, None]], 2).astype(np.uint)
    points3dMatrix[np.where(invalid_>0)]=np.nan
    x3 = x3.reshape(-1, 1)
    y3 = y3.reshape(-1, 1)
    z3 = z3.reshape(-1, 1)
    invalid_ = invalid.reshape(-1, 1).astype(np.uint)
    points3d = np.concatenate([x3, z3, -y3], 1)
    points3d[np.where(invalid_>0), :] = np.nan
    points3d = np.dot(points3d, data['Rtilt']) 
    return rgb, points3d, points3dMatrix, imsize


if __name__ == '__main__':
    root_dir = r'..' + '/'
    train_ids = np.loadtxt('../sunrgbd_trainval/train_data_idx.txt')
    val_ids = np.loadtxt('../sunrgbd_trainval/val_data_idx.txt')
    trainval_ids = np.concatenate([train_ids, val_ids]).astype(np.int)
    sunrgbdm    = loadmat(root_dir + 'OFFICIAL_SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
    sunrgbd2dm  = h5py.File(root_dir + 'OFFICIAL_SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
    # SUNRGBDMeta = sunrgbdm['SUNRGBDMeta'][0]
    # SUNRGBD2Dseg = sunrgbd2dm['SUNRGBD2Dseg'][0]
    sunmeta3_v2  = loadmat(root_dir + 'OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat')
    sunmeta2_v2  = loadmat(root_dir + 'OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat')
    SUNRGBDMeta = sunmeta3_v2['SUNRGBDMeta'][0]
    SUNRGBDMeta2DBB = sunmeta2_v2['SUNRGBDMeta2DBB'][0]
    depth_folder = '../sunrgbd_trainval/depth/';
    image_folder = '../sunrgbd_trainval/image/';
    calib_folder = '../sunrgbd_trainval/calib/';
    det_label_folder = '../sunrgbd_trainval/label/';
    seg_label_folder = '../sunrgbd_trainval/seg_label/';
    if not os.path.exists(depth_folder):
        os.makedirs(depth_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(calib_folder):
        os.makedirs(calib_folder)
    if not os.path.exists(det_label_folder):
        os.makedirs(det_label_folder)
    if not os.path.exists(seg_label_folder):
        os.makedirs(seg_label_folder)

    for imageId in tqdm(range(10355)):
        if imageId not in trainval_ids:
            continue
        data = SUNRGBDMeta[imageId]
        data['depthpath'] = root_dir + '/OFFICIAL_SUNRGBD/' + data['depthpath'][0][17:]
        data['rgbpath']   = root_dir + '/OFFICIAL_SUNRGBD/' + data['rgbpath'][0][17:]
        rgb, points3d, depthInpaint, imsize = read3dPoints(data)
        points3d_rgb = np.concatenate([points3d, rgb], 1)
        points3d_rgb = np.delete(points3d_rgb, np.where(np.isnan(points3d[:, 0])), axis=0)
        points3d_rgb = points3d_rgb.astype(np.float32)
        # MAT files are 3x smaller than TXT files.
        mat_filename = str(imageId).zfill(6) + '.mat'
        txt_filename = str(imageId).zfill(6) + '.txt'
        savemat(depth_folder+mat_filename, {'instance': points3d_rgb})
        #  Write images
        shutil.copy(data['rgbpath'], image_folder+str(imageId).zfill(6)+'.jpg')
        # Write calibration
        calib_info = np.vstack([data['Rtilt'].reshape(1, -1), data['K'].reshape(1, -1)])
        np.savetxt(calib_folder+txt_filename, calib_info)
        # Write 2D and 3D box labe
        data2d = SUNRGBDMeta2DBB[imageId]
        if len(data['groundtruth3DBB']) < 1:
            continue
        gt3d = data['groundtruth3DBB'][0]
        gt3d_info = []
        for j in range(len(gt3d)):
            centroid    = gt3d[j]['centroid'][0]
            classname   = gt3d[j]['classname'][0]
            orientation = gt3d[j]['orientation'][0]
            coeffs      = gt3d[j]['coeffs'][0]
            if len(data2d['groundtruth2DBB']) <1 or j > len(data2d['groundtruth2DBB'][0]) - 1 or classname != data2d['groundtruth2DBB'][0][j]['classname'][0]:
                continue
            box2d       = data2d['groundtruth2DBB'][0][j]['gtBb2D'][0]
            tmp         = [classname, str(box2d[0]), str(box2d[1]), str(box2d[2]), str(box2d[3]), \
                            str(centroid[0]), str(centroid[1]), str(centroid[2]), str(coeffs[0]), str(coeffs[1]), str(coeffs[2]), \
                                str(orientation[0]), str(orientation[1]), '\n']
            gt3d_info.append(' '.join(tmp))
        with open(det_label_folder+txt_filename, 'w', encoding='utf8') as f:
            f.writelines(gt3d_info)
