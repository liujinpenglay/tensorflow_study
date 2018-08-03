
import tensorflow as tf

# 创建一个变量
state =tf.Variable(0, name ='counter')

# 创建一个op，使state 加 1 

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图前, 变量必须先经过`初始化` (init) op 初始化, 

init_op = tf.global_variables_initializer()

# 启动图
with tf.Session() as sess:
    sess.run(init_op)
    # 打印初始化的值    
    print(sess.run(state))
    #print(update)
    # 运行 op 更新 'state', 并打印
    for xx in range(3):
        result = sess.run(update)
        print(result)
    # 测试多个参数 Fetch
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.multiply(input1, intermed)
    resultt = sess.run([mul, intermed])
    print(sess.run(mul),resultt)
    
    # Feed 临时替换一个 op 的输出结果
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output  = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run([output], feed_dict={input1:[7], input2:[2]}))
