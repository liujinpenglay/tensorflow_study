#coding:utf-8
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT =9

# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
#Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
# 加入噪音 -0.05 ~ 0.05
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]
print('X:\n',X)
print('Y:\n',Y_)

#1 定义输入，参数，前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w1)

#w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
#w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))
#a = tf.matmul(x, w1)
#y = tf.matmul(a, w2)

#2 损失函数和反向传播过程
#loss = tf.reduce_mean(tf.square(y - y_))
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3 生成会话 
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print('w1:\n', sess.run(w1))
    #print('w2:\n', sess.run(w2))
    print('\n')

    #loss_val, y_square = sess.run([loss, tf.square(y-y_)], feed_dict={x: X, y_: Y})
    #print('square:\n', y_square)
    #print('loss:\n',loss_val)
    
    # 训练模型
    STEPS = 200
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            #total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("Atfer %d training steps, w1 is: " %i)
            print(sess.run(w1), '\n')
            #print("Atfer %d training step(s), loss on all data is %g"%(i, total_loss))
    
    # 输出参数
    print('\n')
    print('w1:\n', sess.run(w1))
    #print('softmax:\n',sess.run(tf.nn.softmax([1.1,1.1,1.1])))
    #print('w2:\n', sess.run(w2))



