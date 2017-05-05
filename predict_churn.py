import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

trainX = np.loadtxt('/home/lks21c/repo/spark/userChurnList.csv', delimiter=',', dtype=np.float32)

x_data = trainX[:, :-1]
y_data = trainX[:, [-1]]

numFeatures = x_data.shape[1]

numLabels = 1

learningRate = 0.01

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, numFeatures])
Y = tf.placeholder(tf.float32, shape=[None, numLabels])

W = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                 name="weights"))

b = tf.Variable(tf.random_normal([1, numLabels],
                                 name="bias"))

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

cost_summ = tf.summary.scalar("cost", cost)

train = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/mylogi")
    writer.add_graph(sess.graph)

    for step in range(27000):
        cost_val, _, accr, summary = sess.run([cost, train, accuracy, merged_summary], feed_dict={X: x_data, Y: y_data})

        writer.add_summary(summary, global_step=step)

        if step % 200 == 0:
            print(step, cost_val, accr)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
