# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:17:06 2022

@author: Administrator
"""

import numpy as np

#定义一个求和函数
def sum_test(a, b=1):
    c = a + b
    return c


class Cls_test():
    def __init__(self, a):
        self.a = a
    
    #定义一个求和函数
    def sum_test(self, a, b=1):
        d = a + b + self.a
        return d
        

if __name__ == '__main__':
    #调用sun_test函数
    x = sum_test(1, 5)
    #输出：6
    print(x)
    x = sum_test(1)
    #输出：2
    print(x)
    
    #实例化类
    Clt = Cls_test(3)
    #输出：3
    print(Clt.a)
    #调用类中函数sum_test
    x = Clt.sum_test(1, 5)
    #输出：9
    print(x)
    
    #以上语句顺序执行，属于顺序结构
    
    
    a = 10
    #判断结构，输出：a>5
    if a > 5:
        print('a > 5')
    elif a > 10:
        print('a > 10')
    else:
        print('a <= 5')
        
    #循环结构
    for x in [1, 3, 5, 7]:
        print(x)
        
    a = 10
    #循环结构
    while a > 0:
        print(a)
        a = a - 1