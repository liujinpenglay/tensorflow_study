#coding:utf-8
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

# 基于seed产生随机数
rng = np.random.RandomState(seed)
X = rng.rand(32, 2)
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print('X:\n',X)
print('Y:\n',Y)

#1 定义输入，参数，前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2 损失函数和反向传播过程
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#3 生成会话 
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))
    print('\n')

    #loss_val, y_square = sess.run([loss, tf.square(y-y_)], feed_dict={x: X, y_: Y})
    #print('square:\n', y_square)
    #print('loss:\n',loss_val)
    
    # 训练模型
    STEPS = 30000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("Atfer %d training step(s), loss on all data is %g"%(i, total_loss))
    
    # 输出参数
    print('\n')
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))



