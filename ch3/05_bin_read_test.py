# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import numpy as np
if __name__ == '__main__':
    points = np.fromfile('000001.bin', dtype=np.float32)
    print(points[:20])
    points = points.reshape(-1, 4)
    print(points)