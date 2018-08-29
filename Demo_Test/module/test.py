#coding:utf-8
# 反向传播模块化
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateds
import forward

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY =0.999
REGULARIZER = 0.01

xx, yy = np.mgrid[0:3:1, 0:3:1]
print(xx,'\n',xx.ravel())
print(yy,'\n',yy.ravel())








