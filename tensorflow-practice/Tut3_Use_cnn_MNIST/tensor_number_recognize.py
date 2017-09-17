#-*- coding: utf-8 -*-　
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
GLOG_logtostderr=1
#定義準確度函數 如果不懂這部份代碼看完前面MNIST教學 應該就懂了
def compute_accuracy(v_xs, v_ys):
    print("i am here")
    global prediction
    print("i am here2")
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    #print(y_pre);
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
 
#定義初始化變數 採用normal distribution , 標準差為0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
#定義初始化變數 採用常數 , 皆為為0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
#定義conv 層 layer padding 方法採用"一樣"
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
#movement代表移動的步數，如下圖移動步數為4，所以剛好是filter的一半，每次做conv都是移動4步
     
 
 
# 定義placeholder
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
 
#將圖片reshape， -1表示會自動算幾組
#28,28,1 分別代表寬 高 channel數(像RGB的話這裡就要改3)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
 
 
#開始組裝神經網路
## conv1 layer ##
#1:表示 input_size  32:表示output_size 所以這裡表示一張圖總共訓練出32個filter
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32
 
#max_pooling 示意圖：就是2*2的矩陣中 挑出一個最大的數字 所以大小會變一半

## conv2 layer ##
#這裡表示 一張圖訓練出2個filter
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64
 
## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
 
#這裡將第2層max_pool 過後的神經元 全部攤平
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
## func2 layer ##
#倒數第二層為1024個神經元 最後一層為10個神經元 採用softmax當成最後一層的激活函數
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
# 定義loss function 以及 優化函數
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))# loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 
#定義Sess 以及初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
#開始訓練，dropout 0.5代表隨機隱藏掉一半神經元的資訊
#科學家們發現這樣可以有效的減少overfitting
#有關dropout的相關資訊可以參考這篇
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print(i)

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 ==0 and i > 0:
        #y_pre = sess.run(prediction, feed_dict={xs:mnist.test.images, keep_prob: 0.5})
        #print(y_pre)
        print(compute_accuracy(mnist.test.images, mnist.test.labels))