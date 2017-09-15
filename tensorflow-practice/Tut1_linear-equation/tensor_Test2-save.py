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
        outputs = activation_function(logits=logits)
    

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

#接下來製造一些數據和雜訊吧 
#製造出範圍為-1~1之間的 row:300 col:1 矩陣
x_data = np.linspace(-5,5,1000)[:, np.newaxis]
noise = np.random.normal(-0.05, 0.05, x_data.shape)

#製造出要讓網路學習的Y 並加上雜訊
# y_data = np.square(x_data) - 0.5 + noise
y_data = np.sqrt((np.square(x_data)) + 1) + noise

# 定義loss function 並且選擇減低loss 的函數 這裡選擇GradientDescentOptimizer
# 其他方法再這裡可以找到 https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
loss = tf.reduce_mean(tf.squared_difference(prediction, ys))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

#全部設定好了之後 記得初始化喔
init = tf.global_variables_initializer()
#sess = tf.Session()
sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver()
  
print(matplotlib.rcParams['backend'])

# 為了可以可視化我們訓練的結果
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()
# plt.savefig('start')

# 之後就可以用for迴圈訓練了
for i in range(10000):
     # 整個訓練最核心的code , feed_dict 表示餵入 輸入與輸出
     # x_data:[300,1]   y_data:[300,1]
    _,prediction_val,loss_val= sess.run([train_step,prediction,loss], feed_dict={xs: x_data, ys: y_data})
    print(loss.eval(feed_dict={xs: x_data, ys: y_data},session=sess))
    #print(prediction.eval(feed_dict={xs: x_data, ys: y_data},session=sess))
    #print(prediction_val);
    # print(loss_val)
    if i % 50 == 0:
        # 畫出下一條線之前 必須把前一條線刪除掉 不然會看不出學習成果
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        
        # 要取出預測的數值 必須再run 一次才能取出
        # print(loss.eval())
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        print(i)
        # 每隔0.1 秒畫出來
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # plt.pause(0.1)
        if i % (5000) == 0:
            print(loss_val)
#             plt.savefig('filename_'+str(i))

saver.save(sess, './model/' + 'tommodel.ckpt', global_step=i+1)        
        
        
# plt.savefig('filename')
# plt.close()

x_data2 = np.linspace(-2,2,1000)[:, np.newaxis]
noise = np.random.normal(-0.05, 0.1, x_data2.shape)

y_data2 = np.sqrt((np.square(x_data2)) + 1)+noise

# predict new set of x_data
prediction_value = sess.run(prediction, feed_dict={xs:x_data2})
fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
ax.scatter(x_data2, y_data2)
lines = ax.plot(x_data2, prediction_value, 'r-', lw=5)
plt.ion()
plt.show()
plt.savefig('test')
plt.close();