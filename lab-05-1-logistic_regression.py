b 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
trainX = np.loadtxt('data/trainX.csv', delimiter='\t', dtype=np.float32)
x_data = trainX
numFeatures = x_data.shape[1]

print("x_data shape = ", x_data.shape[0], x_data.shape[1])

y_data = [[0], [0], [0], [1], [1], [1]]
# print("y_data shape = ", y_data.shape[0], y_data.shape[1])

testY = np.loadtxt('data/trainY.csv', delimiter='\t', dtype=np.float32)
yshape = testY[:, [0]]
print("yshape shape = ", yshape.shape[0], yshape.shape[1])

y_data = yshape
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
# cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
#                        tf.log(1 - hypothesis))

cost = tf.nn.l2_loss(hypothesis - Y, name="squared_error_cost")

train = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(27000):
        cost_val, _, accr = sess.run([cost, train, accuracy], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val, accr)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
0 1.73078
200 0.571512
400 0.507414
600 0.471824
800 0.447585
...
9200 0.159066
9400 0.15656
9600 0.154132
9800 0.151778
10000 0.149496

Hypothesis:  [[ 0.03074029]
 [ 0.15884677]
 [ 0.30486736]
 [ 0.78138196]
 [ 0.93957496]
 [ 0.98016882]]
Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
Accuracy:  1.0
'''

