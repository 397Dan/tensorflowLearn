#coding=utf-8
'''
https://cuiqingcai.com/4893.html
'''
import tensorflow as tf

import numpy as np

x_data = np.float32(np.random.rand(2, 100))
# print x_data
y_data = np.dot([0.300, 0.200], x_data) + 0.400
# print y_data

x = tf.placeholder(tf.float32, [2, 100])
y_label = tf.placeholder(tf.float32, [100])
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([2], -1.0, 1.0))
y = tf.matmul(tf.reshape(w, [1, 2]), x) + b

loss = tf.reduce_mean(tf.square(y - y_label))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # tensorflow改版后更新了方法,文章里的不再适用
    for step in range(201):
        sess.run(train, feed_dict={x: x_data, y_label: y_data})
        if step % 10 == 0:
            print(step, sess.run(w), sess.run(b))