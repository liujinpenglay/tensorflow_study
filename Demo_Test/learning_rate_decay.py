#coding:utf-8
# 设损失函数 loss=(w+1)^2, 令w初值是常数10，反向传播就是求最优w
# 使用指数衰减的学习率， 在初期较高的迭代， 可以在较小的轮数得到更好的收敛度
import tensorflow as tf

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1    # 总样本数/BATH_SIZE

#BATH_SIZE 的计数器
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,  LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w+1)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('After %s steps: global_step is %f, w is %f, learning_rate is %f, loss is %f.'%(i, global_step_val, w_val, learning_rate_val, loss_val))
