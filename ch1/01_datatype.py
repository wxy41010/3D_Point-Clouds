# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:50:14 2022

@author: https://blog.csdn.net/suiyingy?type=blog

推荐书籍：《Python三维点云 》
"""

if __name__ == '__main__':
    #1、数值类型
    #定义整型变量a
    a = 1
    #打印结果，输出：<class 'int'>
    print(type(a))
    #将a强制转换成浮点型
    b = float(a)
    #输出：1.0 <class 'float'>，可以看到b中有小数点存在
    print(b, type(b))
    #将b转换回整型
    c = int(b)
    #输出：1 <class 'int'>，c中不再有小数点，即c为整型
    print(c, type(c))
    
    #2、字符串类型
    #输出：python	三维点云，\t为转义字符。
    d = 'python\t三维点云'
    print(d)
    #输出：python\t三维点云，\t不当作转义字符处理
    e = r"python\t三维点云"
    print(e)
    f = d + e
    #字符串拼接，输出：python	三维点云python\t三维点云
    print(f)
    
    #3、list列表类型
    g = [1, 'a', [2, 3]]
    #list列表切片，输出：1 [2, 3] ['a', [2, 3]]
    print(g[0], g[-1], g[1:3])
    #计算数组长度，输出
    print(len(g))
    #删除列表元素
    g.remove(1)
    #输出：['a', [2, 3]]
    print(g)
    #删除列表元素
    del g[-1]
    #输出：['a']
    print(g)

    #4、dict字典类型
    f = {'a': 1, 0:'abc', 'b':[1, 2, 3]}
    #获取数据，输出：1 abc [1, 2, 3]
    print(f['a'], f[0], f['b'])
    #获取所有键，并转换为list
    keys = list(f.keys())
    #输出：['a', 0, 'b']
    print(keys)
    #获取所有取值，并转换为list
    values = list(f.values())
    #输出：[1, 'abc', [1, 2, 3]]
    print(values)
    #获取所有键值对，并转换为list
    items = list(f.items())
    #输出：[('a', 1), (0, 'abc'), ('b', [1, 2, 3])]
    print(items)
    #删除字典元素
    del f['b']
    #删除：{'a': 1, 0: 'abc'}
    print(f)

if __name__ == '__main__':    
    #5、元组类型
    g = (1, '2', [3])
    #元组切片，与列表类似，输出：2
    print(g[1])
    
    #6、集合类型
    h = set([1, 1, 2, '2'])
    #输出：{1, 2, '2'}
    print(h)
    
    #7、数组类型
    #导入numpy包
    import numpy as np
    #定义数组
    i = np.array([[1, 2], [3, 4]])
    #输出：[[1 2][3 4]]
    print(i)
    #计算矩阵维度，输出:(2, 2)
    print(i.shape)