import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#define layer function
def add_layer(inputs, in_size, out_size, activation_func=None):
    outputs = None
    weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]))
    logits = tf.matmul(inputs, weights) + biases   
     
    if activation_func is None:
        outputs = logits
    else:
        outputs = activation_func(logits=logits)
        
    return outputs
    
#load raw data    
iris = datasets.load_iris()

#count the set of results.(Iris Setosa, Iris Versicolour, Iris Virginica)
num_labels = len(set(iris.target))

input_data = iris.data.astype(np.float32)

labels = (np.arange(num_labels) == np.array(iris.target)[:,None]).astype(np.float32)

print(input_data.shape, labels.shape)
x_data = input_data
y_data = labels


#sess = tf.Session()
graph = tf.Graph()
#create input and output tf feed

with graph.as_default():
    xs = tf.placeholder(tf.float32, [None, 4])
    ys = tf.placeholder(tf.float32, [None, 3])
        
    # add hidden layer
    l1 = add_layer(xs, 4, 30, activation_func=tf.nn.softmax)
    
    # add output layer
    prediction = add_layer(l1, 30, 3, activation_func=None)

    # calculate loss
    loss = tf.reduce_mean(tf.squared_difference(prediction, ys))
    # loss = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    
    # set train function
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
    
# init = tf.global_variables_initializer()
# sess = tf.InteractiveSession(graph=graph)
# sess.run(init)


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    for step in range(10000):
        _,prediction_val,loss_val= sess.run([train_step,prediction,loss], feed_dict={xs: x_data, ys: y_data})
        print(loss.eval(feed_dict={xs: x_data, ys: y_data},session=sess))

    #test
    prediction_val= sess.run(prediction, feed_dict={xs: x_data})
    np.savetxt("prediction_answer.csv", prediction_val, delimiter=",")
    np.savetxt("model_answer.csv", y_data, delimiter=",")