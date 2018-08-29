#coding:utf-8
import tensorflow as tf

#1 定义变量及滑动平均
# 定义一个变量，初始值为0，不断更新优化 w1 参数，滑动平均做了一个 w1 的影子
w1 = tf.Variable(0, dtype=tf.float32)
# 定义 num_updates(NN的迭代轮数)，初始值为0，不可被训练
global_step = tf.Variable(0, trainable=False)
# 实例化滑动平均类，衰减率为0.99，当前轮数 global_step
MOVING_AVERAGE_DECAY =0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 打印当前参数 w1 和 w1 滑动平均值
    print(sess.run([w1, ema.average(w1)]))

    # 参数手动赋值
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    
    # 模拟更新参数
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
