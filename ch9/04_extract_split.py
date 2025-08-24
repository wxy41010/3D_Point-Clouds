# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
二维、三维、量化投资等深度学习算法分享。
"""

import os
from scipy.io import loadmat
from tqdm import tqdm

if __name__ == '__main__':
    ratio = 0.01
    root_dir = r'..'
    split = loadmat(root_dir + '/OFFICIAL_SUNRGBD/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat');
    N_train = len(split['alltrain'][0])
    N_val = len(split['alltest'][0])

    train_files = []
    val_files   = []
    for i in range(N_train):
        folder_path = split['alltrain'][0][i]
        folder_path = folder_path[0][17:]
        train_files.append(folder_path)

    for i in range(N_val):
        folder_path = split['alltest'][0][i]
        folder_path = folder_path[0][17:]
        val_files.append(folder_path)

    train_ids = []
    val_ids   = []
    if not os.path.exists('../sunrgbd_trainval'):
        os.makedirs('../sunrgbd_trainval')
    sunmeta  = loadmat(root_dir + '/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat')
    SUNRGBDMeta = sunmeta['SUNRGBDMeta'][0]
    for imageId in tqdm(range(N_train + N_val)):
        data = SUNRGBDMeta[imageId]
        depthpath = data['depthpath'][0][17:].rsplit('/', 2)[0]
        if depthpath in train_files:
            train_ids.append(str(imageId) + '\n')
        elif depthpath in val_files:
            val_ids.append(str(imageId) + '\n')
    N_train = int(ratio * N_train)
    N_val   = int(ratio * N_val)
    with open('../sunrgbd_trainval/train_data_idx.txt', 'w', encoding='utf-8')  as f:
                f.writelines(train_ids[:N_train])
    with open('../sunrgbd_trainval/val_data_idx.txt', 'w', encoding='utf-8')  as f:
                f.writelines(val_ids[:N_val])
