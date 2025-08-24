# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

import torch
import torch.nn as nn

   
if __name__ == '__main__':
    x1d     = torch.Tensor([[[1, 2, 3, 4, 5]]])
    print('x1d.shape: ', x1d.shape)
    conv1d  = nn.Conv1d(1, 1, 3, padding=0)
    conv1d.weight.data = torch.Tensor([[[2, 1, 1]]])
    conv1d.bias.data   = torch.Tensor([0.1])
    y1d     = conv1d(x1d)
    print('y1d: ', y1d)


    x1d     = torch.Tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    print('x1d.shape: ', x1d.shape)
    conv1d  = nn.Conv1d(2, 1, 3, padding=0)
    conv1d.weight.data = torch.Tensor([[[2, 1, 1], [2, 1, 1]]])
    conv1d.bias.data   = torch.Tensor([0.1])
    y1d     = conv1d(x1d)
    print('y1d: ', y1d)

    x1d     = torch.Tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    print('x1d.shape: ', x1d.shape)
    conv1d  = nn.Conv1d(2, 2, 3, padding=0)
    conv1d.weight.data = torch.Tensor([[[2, 1, 1], [2, 1, 1]], [[2, 1, 1], [2, 1, 1]]])
    conv1d.bias.data   = torch.Tensor([0.1, 0.2])
    print(conv1d.weight.data.shape)
    y1d     = conv1d(x1d)
    print('y1d: ', y1d)

    x1d     = torch.Tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    print('x1d.shape: ', x1d.shape)
    conv1d  = nn.Conv1d(2, 2, 3, padding=2)
    conv1d.weight.data = torch.Tensor([[[2, 1, 1], [2, 1, 1]], [[2, 1, 1], [2, 1, 1]]])
    conv1d.bias.data   = torch.Tensor([0.1, 0.2])
    print(conv1d.weight.data.shape)
    y1d     = conv1d(x1d)
    print('y1d: ', y1d)

    x1d     = torch.Tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    print('x1d.shape: ', x1d.shape)
    conv1d  = nn.Conv1d(2, 2, 3, 2, padding=1)
    conv1d.weight.data = torch.Tensor([[[2, 1, 1], [2, 1, 1]], [[2, 1, 1], [2, 1, 1]]])
    conv1d.bias.data   = torch.Tensor([0.1, 0.2])
    y1d     = conv1d(x1d)
    print('y1d: ', y1d)
    #最大池化
    x1d     = torch.Tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    print('x1d.shape: ', x1d.shape)
    mpool  = nn.MaxPool1d(3, 2)
    y1d     = mpool(x1d)
    print('y1d: ', y1d)

 