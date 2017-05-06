import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

# Set Training Data
trainX = np.loadtxt('/d_drive/refined_data/logIdCnt/trainUserChurnList.csv', delimiter=',', dtype=np.float32)
# feature : 1st column ~ the column before the last column
x_data = trainX[:, :-1]
# y value : the last column
y_data = trainX[:, [-1]]

# Set numFeatures
numFeatures = x_data.shape[1]

# Set Cross Validation Data
cvX = np.loadtxt('/d_drive/refined_data/logIdCnt/cvUserChurnList.csv', delimiter=',', dtype=np.float32)
x_cvdata = cvX[:, :-1]
y_cvdata = cvX[:, [-1]]

# Set Test Data
testX = np.loadtxt('/d_drive/refined_data/logIdCnt/testUserChurnList.csv', delimiter=',', dtype=np.float32)
x_testdata = testX[:, :-1]
y_testdata = testX[:, [-1]]

# 1 because it's binary classification
numLabels = 1

# Set Learning Rate
learningRate = 0.01

# Iteration
iteration = 200000

# X is feature and Y is the right answer.
X = tf.placeholder(tf.float32, shape=[None, numFeatures])
Y = tf.placeholder(tf.float32, shape=[None, numLabels])

# Set Hidden Layer 1
W = tf.Variable(tf.random_normal([numFeatures, numFeatures],
                                 name="weights"))
b = tf.Variable(tf.random_normal([1, numLabels],
                                 name="bias"))
layer1 = tf.nn.relu(tf.matmul(X, W) + b)

# Set Hidden Layer 2
W2 = tf.Variable(tf.random_normal([numFeatures, numFeatures]), name='weight2')
b2 = tf.Variable(tf.random_normal([numFeatures]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

# Set Hidden Layer 3
W3 = tf.Variable(tf.random_normal([numFeatures, numFeatures]), name='weight3')
b3 = tf.Variable(tf.random_normal([numFeatures]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

# Set Output Layer
W4 = tf.Variable(tf.random_normal([numFeatures, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.matmul(layer3, W4) + b4

# Set Cost Function as Cross Entropy.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                              labels=Y))
# cost tensor for tensorboard.
cost_summ = tf.summary.scalar("cost", cost)

# Set default optimizer as GradientDescentOptimizer.
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# assums predicted if h(x) exceeds 0.5
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

# accuaracy for training dataset
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# accuracy for cv dataset
cv_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# accuracy for test dataset
test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# accracy tensor for tensorboard.
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Set saver tensor for saving the model
saver = tf.train.Saver()

# Run Graph
with tf.Session() as sess:
    # Initialize tensorboard variables
    sess.run(tf.global_variables_initializer())

    # Merge tensorboard summary
    merged_summary = tf.summary.merge_all()
    # Set tensorboard log path
    writer = tf.summary.FileWriter("./logs/predictChurn")
    # Write graph info to tensorboard.
    writer.add_graph(sess.graph)

    # Traing the model until interation ends.
    for step in range(iteration):
        # train is run wtih x_data, y_data.
        cost_val, _, accr, summary = sess.run([cost, train, accuracy, merged_summary], feed_dict={X: x_data, Y: y_data})

        # Save tensorboard summary while training in in progress.
        writer.add_summary(summary, global_step=step)

        # pring debug info by every 200 step.
        if step % 200 == 0:
            print(step, cost_val, accr)

    # Save ML Model after training.
    save_path = saver.save(sess, "/d_drive/model/model.ckpt")
    print("Model saved in file: %s" % save_path)

    # Accuracy Report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("Trains Set Accuracy: ", a)

    cv_h, cv_c, cv_a = sess.run([hypothesis, predicted, cv_accuracy], feed_dict={X: x_cvdata, Y: y_cvdata})
    print("Cross Validation Accuracy: ", cv_a)

    test_accr = sess.run([test_accuracy], feed_dict={X: x_testdata, Y: y_testdata})
    print("Test Validation set Accuracy: ", test_accr)