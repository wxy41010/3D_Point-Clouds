# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

def pcd_read(file_path):
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines
 
if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    points = pcd_read(file_path)
    for p in points[:15]:
        print(p)