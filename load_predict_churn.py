import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

trainX = np.loadtxt('/d_drive/userChurnList.csv', delimiter=',', dtype=np.float32)

x_data = trainX[:, :-1]
y_data = trainX[:, [-1]]

numFeatures = x_data.shape[1]

print(trainX.shape)
print(x_data.shape)
print(y_data.shape)
print(numFeatures)

numLabels = 1

learningRate = 0.01
iteration = 1

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, numFeatures])
Y = tf.placeholder(tf.float32, shape=[None, numLabels])

W = tf.Variable(tf.random_normal([numFeatures, numFeatures],
                                 name="weights"))

b = tf.Variable(tf.random_normal([1, numLabels],
                                 name="bias"))

layer1 = tf.sigmoid(tf.matmul(X, W) + b)

W2 = tf.Variable(tf.random_normal([numFeatures, numFeatures]), name='weight2')
b2 = tf.Variable(tf.random_normal([numFeatures]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([numFeatures, numFeatures]), name='weight3')
b3 = tf.Variable(tf.random_normal([numFeatures]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([numFeatures, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cost/loss function
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    # sess.run(tf.global_variables_initializer())

    saver.restore(sess, "/d_drive/model.ckpt")

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/mylogi2")
    writer.add_graph(sess.graph)

    for step in range(iteration):
        accr, summary = sess.run([accuracy, merged_summary], feed_dict={X: x_data, Y: y_data})

        writer.add_summary(summary, global_step=step)

        print(accr)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)