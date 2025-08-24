# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:44:02 2022

@author: Administrator
"""

import os
import glob
import shutil
from pathlib import Path
 
if __name__ == '__main__':
    #proj_name可自定义
    proj_name = 'py2pyd_batch_test'
    
    #带批量打包的工程所在文件目录
    proj_dir = r'testproject'
    proj_dir = str(Path(proj_dir)) + '/'
    #pyd结果保存目录
    respyd_dir = r'respyd'
    #编译环境
    pyd_suffix = '.cp37-win_amd64'
    if os.path.exists(respyd_dir):
        shutil.rmtree(respyd_dir)
    shutil.copytree(proj_dir, respyd_dir)
    py_files = glob.glob(proj_dir + '**', recursive=True)
    i = 1
    for py in py_files:
        py_ = os.path.splitext(py)
        filename = os.path.basename(py).rsplit('.', 1)[0]
        dirname  = os.path.dirname(py)
        dst_dir  = dirname.replace(str(Path(proj_dir)), respyd_dir) + '/'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        if py_[-1] != '.py':
            continue
        print(str(i).zfill(3), ': building', py)
        command = 'python single_pyd_setup.py build_ext --inplace ' + proj_name + ' ' + py
        os.system(command)
        if os.path.exists(proj_dir+filename + pyd_suffix+'.pyd'):
            dst = dst_dir + filename + '.pyd'
            shutil.copy(proj_dir+filename + pyd_suffix+'.pyd', dst)
            os.remove(proj_dir+filename + pyd_suffix+'.pyd')
        if os.path.exists(proj_dir+filename + pyd_suffix+'.exp'):
            os.remove(proj_dir+filename + pyd_suffix+'.exp')
        if os.path.exists(proj_dir+filename + pyd_suffix+'.lib'):
            os.remove(proj_dir+filename + pyd_suffix+'.lib')
        if os.path.exists(dst_dir + filename +'.py'):
            os.remove(dst_dir + filename +'.py')
        
        if os.path.exists(filename+'.pyd'):
            dst = dst_dir + filename +'.pyd'
            shutil.copy(filename+'.pyd', dst)
            
        
        if os.path.exists(filename+'.pyd'):
            os.remove(filename+'.pyd')
        if os.path.exists(filename+pyd_suffix+'.pyd'):
            os.remove(filename+pyd_suffix+'.pyd')
        
        i += 1
        # break