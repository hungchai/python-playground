#-*- coding: utf-8 -*-　
#引用函數
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# Import data
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
 
#設定資料儲存地方
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
 
#如果電腦沒有數據庫的話，開始執行下載
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
sess = tf.InteractiveSession()
 
# 建造模型 這裡的模型就只是一個輸入神經元為784 輸出為10個神經元的網路  採用softmax 當激活函數

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
 
# 定義loss function 和優化器
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
 
# 開始訓練囉~
tf.global_variables_initializer().run()
for i in range(1000):
  print(i)
  batch_xs, batch_ys = mnist.train.next_batch(100)
  #把我們剛剛設定的placeholder 開始丟資料進去
  train_step.run({x: batch_xs, y_: batch_ys})
  
# 訓練完後 測試準確度
# correct_prediction 出來是boolean 資料 有相等就是1 沒有就是0
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#經由cast 轉變為float32的型式，然後reduce_mean 取準確度
#例如 總共有90個1，10個0 經由reduce_mean 會把所有數字加起來除以總共個數 也就是90/100=0.9
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#採用eval 方式 列印出來
print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))