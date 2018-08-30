#conding:utf-8
import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

print(mnist.train.num_examples)
print(mnist.validation.num_examples)
print(mnist.test.num_examples)
print(mnist.train.labels[0])
print(mnist.train.images[0])

BATCH_SIZE = 200
xs, ys = mnist.train.next_batch(BATCH_SIZE)
print('xs shape:', xs.shape)
print('ys shape:', ys.shape)
