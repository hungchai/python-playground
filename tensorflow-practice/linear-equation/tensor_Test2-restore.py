#-*- coding: utf-8 -*-　
# http://darren1231.pixnet.net/blog/post/336860963-tensorflow%E6%95%99%E5%AD%B8%282%29----%E5%BB%BA%E7%BD%AE%E4%B8%80%E5%80%8B%E6%9C%89%E9%9A%B1%E8%97%8F%E5%B1%A4%E7%9A%84%E7%A5%9E%E7%B6%93%E7%B6%B2

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    logits = tf.matmul(inputs, Weights) + biases   

    #自由選擇激活函數
    if activation_function is None:
        outputs = logits
    else:
        outputs = activation_function(logits)
    

    return outputs

# 給tensorflow 一個placeholder 隨時置換數據 None 表示會自己計算出放了多少組數據
# 像這裡 None 就會自動放入300組 因為我們等等會放入300組數據訓練 
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


#組裝神經網路
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.softmax)
# l2 = add_layer(l1, 3, 6, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
# loss = tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1])
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
saver = tf.train.Saver()
graph = tf.Graph()
print matplotlib.rcParams['backend']

# sess = tf.InteractiveSession()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    x_data2 = np.linspace(-2,2,1000)[:, np.newaxis]
    noise = np.random.normal(-0.05, 0.1, x_data2.shape)
    y_data2 = np.sqrt((np.square(x_data2)) + 1)+noise
    feed_dict = {xs : x_data2}
    
    prediction_value = sess.run(prediction, feed_dict)
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    ax.scatter(x_data2, y_data2)
    lines = ax.plot(x_data2, prediction_value, 'r-', lw=5)
    plt.ion()
    plt.show()
    plt.savefig('test')
    plt.close();    

