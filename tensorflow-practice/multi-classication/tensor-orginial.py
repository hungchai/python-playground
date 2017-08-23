import numpy as np
from sklearn import datasets
import tensorflow as tf

iris = datasets.load_iris()
# d = iris.data
# print(iris.data)

# print(iris.target)
# print(set(iris.target_name))
# print(set(iris.target))
# print(list(iris.target))
num_labels = len(set(iris.target))
# print(num_labels)
# num_labels = len(list(iris.target))
data = iris.data.astype(np.float32)
print(np.array(iris.target)[:,None])
print(np.arange(num_labels))
labels = (np.arange(num_labels) == np.array(iris.target)[:,None]).astype(np.float32)
print(labels)
print(data.shape, labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
            
feature_size = data.shape[1]
print(feature_size)

graph = tf.Graph()


with graph.as_default():
    tf_train_dataset = tf.constant(data)
    tf_train_labels = tf.constant(labels)

    weights = tf.Variable(tf.truncated_normal([feature_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(10001):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if step % 500 == 0:
            print('step:{} loss:{:.6f} accuracy: {:.2f}'.format(
                    step, l, accuracy(predictions, labels)))