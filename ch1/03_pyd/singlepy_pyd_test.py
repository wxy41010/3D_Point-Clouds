# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:43:36 2022

@author: Administrator
"""

#调用basics.pyd中的类或函数
import singlepy #pyd包导入与py文件导入完全一致
 
 
if __name__ == '__main__':
   x = 3.1
   #调用basics.py中的函数
   y = singlepy.sum_test(x, 1.2)
   print(y)