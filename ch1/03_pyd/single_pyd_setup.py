# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:43:36 2022

@author: Administrator
"""

import os
import sys
import shutil
from distutils.core import setup
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
def get_export_symbols_fixed(self, ext):
    pass  # return [] also does the job!
# replace wrong version with the fixed:
build_ext.get_export_symbols = get_export_symbols_fixed
 
 
if __name__ == '__main__':
    #在命令行运行：python single_pyd_setup.py build_ext --inplace appname pypath
    #appname名称可以自己任意定义，pypath是需要转为pyd的py文件路径
    #例如：python single_pyd_setup.py build_ext --inplace appname1 singlepy.py
    
    #env_config：定义python版本和系统类型，如果不知道版本和系统类型，可先执行命令，然后会产生对应的pyd文件，文件名会包含该信息
    env_config = '.cp37-win_amd64'
    if os.path.exists('build'):
        shutil.rmtree('build')
    # print('argv: ', sys.argv, len(sys.argv))
    appname = sys.argv[3]
    pypath = sys.argv[4]
    sys.argv = sys.argv[:3]
    # print('argv: ', sys.argv, len(sys.argv))
    cpath = pypath.replace('.py', '.c')
    if os.path.exists(cpath):
        os.remove(cpath)
    setup(
        name=appname,
        ext_modules=cythonize(pypath)
    )
    if os.path.exists(cpath):
        os.remove(cpath)
    if os.path.exists('build'):
        shutil.rmtree('build')
    pypath = os.path.basename(pypath)
    temp_path = pypath.replace('.py', env_config + '.exp')
    if os.path.exists(temp_path):
        os.remove(temp_path)
    temp_path = pypath.replace('.py', env_config + '.lib')
    if os.path.exists(temp_path):
        os.remove(temp_path)
    temp_path = pypath.replace('.py', env_config + '.pyd')    
    if os.path.exists(temp_path):
        os.rename(temp_path, pypath.replace('.py', '.pyd')    )