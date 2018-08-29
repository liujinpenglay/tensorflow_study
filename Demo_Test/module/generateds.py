#coding:utf-8
#0 导入模块，生成模拟数据集
import numpy as np
import matplotlib.pyplot as plt
SEED = 2;
def generateds():
    rdm = np.random.RandomState(SEED)
    # rand:均匀分布  randn:正态分布
    X = rdm.randn(300, 2)
    Y_ = [int(x0*x0 + x1*x1 < 2) for (x0, x1) in X]
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据 X, Y_ 进行形状整理
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1) 
    return X, Y_,Y_c

#print(X)
#print(Y_)
#print(Y_c)
