import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

cvX = np.loadtxt('/d_drive/refined_data/logIdCnt/cvUserChurnList.csv', delimiter=',', dtype=np.float32)

x_cvdata = cvX[:, :-1]
y_cvdata = cvX[:, [-1]]

testX = np.loadtxt('/d_drive/refined_data/logIdCnt/testUserChurnList.csv', delimiter=',', dtype=np.float32)

x_testdata = testX[:, :-1]
y_testdata = testX[:, [-1]]

numFeatures = x_cvdata.shape[1]

print(cvX.shape)
print(x_cvdata.shape)
print(y_cvdata.shape)
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
cv_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Launch graph
with tf.Session() as sess:
    # Restore Variables
    saver.restore(sess, "/d_drive/model/model.ckpt")

    # Accuracy report
    cv_h, cv_c, cv_a = sess.run([hypothesis, predicted, cv_accuracy], feed_dict={X: x_cvdata, Y: y_cvdata})
    print("\nCross Validation Hypothesis: ", cv_h, "\nCross Validation Correct (Y): ", cv_c, "\nCross Validation Accuracy: ", cv_a)

    test_accr = sess.run([test_accuracy], feed_dict={X: x_testdata, Y: y_testdata})
    print("Test Validation set Accuracy: ", test_accr)
