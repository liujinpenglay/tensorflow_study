# 导入 input_data
import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
#交叉熵和梯度下降
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#开始训练模型1000次
for i in range(1):
    batch_xs, batch_ys = mnist.train.next_batch(1)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #print([sess.run(tf.matmul(x,W), feed_dict={x:batch_xs, y_:batch_ys})])
    #print([sess.run(b)])
    #print([sess.run(tf.matmul(x,W)+b, feed_dict={x:batch_xs, y_:batch_ys})])

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
#print([sess.run(tf.matmul(x,W))])
sess.close()

#测试用 无效
a0 = np.array([1,2,3,4,5,6,7,8,9,10])
a1 = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
a2 = a1 + a0
#print(x)
#print(W)
#print(b)
#print(y)
print('a0:',a0)
print('a1:',a1)
print('a2:',a2)


