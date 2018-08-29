#coding:utf-8



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
SEED = 2
rdm = np.random.RandomState(SEED)

X = rdm.randn(300, 2) # 矩阵是二维的[[...]]
Y_ = [int(x1*x1 + x2*x2 < 2) for (x1, x2) in X] #行向量 是一维的[...]
Y_c = [['red' if y else 'blue'] for y in Y_]
#print(X)
#print(Y_)
i_ = np.vstack(Y_).reshape(-1, 1)
# 对数据 X 和 Y_ 进行 shape 整理
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)
print(X)
print(Y_)
print(Y_c)

# 使用 plt.scatter 画图
plt.scatter(X[:, 0],X[:, 1], c=np.squeeze(Y_c))
plt.show()

# 定义神经网络
def get_weight(shape, regularizer):
    W = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(W))
    return W

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape = shape))
    return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2) + b2

# 损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

# 方向传播 不含正则化
#train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)



with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = i*(BATCH_SIZE) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 200 == 0:
            loss_mse_v = sess.run(loss_mse, feed_dict={x: X, y_:Y_})
            print('After %d steps, loss is: %ff' %(i, loss_mse_v))
    #在 -3~3 之间生成步长为0.01 的二维网格坐标点, 构建新数据
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)
    print('w1:\n', sess.run(w1))
    print('b1:\n', sess.run(b1))
    print('w2:\n', sess.run(w2))
    print('b2:\n', sess.run(b2))

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=0.5)
plt.show()



