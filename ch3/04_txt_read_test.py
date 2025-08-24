# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import numpy as np
if __name__ == '__main__':
    points = np.loadtxt('airplane_0001.txt', delimiter=',')
    print(points.shape)
    print(points[:5, :])
