# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""
 
import numpy as np
import pandas as pd
from plyfile import PlyData
 
def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # 读取文件
    data = plydata.elements[0].data  # 读取文件中数据
    data_pd = pd.DataFrame(data)  # 转换为DataFrame格式
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 新建矩阵用于存储点云数据
    property_names = data[0].dtype.names  # 读取属性名称
    for i, name in enumerate(property_names):  # 根据属性名称逐一读取
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)
 
if __name__ == '__main__':
    convert_ply('bun_zipper.ply', 'bun_zipper.bin')