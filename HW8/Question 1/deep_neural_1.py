from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', '/tmp/deep_logs', 'Summaries directory')

if tf.gfile.Exists(FLAGS.summaries_dir):
  tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var, name):
  with tf.name_scope("summaries"):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W_conv1 = weight_variable([5, 5, 1, 32])
# variable_summaries(W_conv1, 'conv1' + '/weights')

b_conv1 = bias_variable([32])
# variable_summaries(b_conv1, 'conv1' + '/bias')

x_image = tf.reshape(x, [-1,28,28,1])
# variable_summaries(b_conv1, 'conv1' + '/bias')

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# tf.histogram_summary('conv1' + '/activations', h_conv1)

h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
# variable_summaries(W_conv2, 'conv2' + '/weights')

b_conv2 = bias_variable([64])
# variable_summaries(b_conv2, 'conv2' + '/bias')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# tf.histogram_summary('conv2' + '/activations', h_conv2)

h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
# variable_summaries(W_fc1, 'fc1' + '/weights')

b_fc1 = bias_variable([1024])
# variable_summaries(b_fc1, 'fc1' + '/bias')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# tf.histogram_summary('fc1' + '/activations', h_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
# variable_summaries(W_fc2, 'fc2' + '/weights')
b_fc2 = bias_variable([10])
# variable_summaries(b_fc2, 'fc2' + '/bias')

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

tf.initialize_all_variables().run()

#sess.run(tf.initialize_all_variables())
for i in range(2001):
  batch = mnist.train.next_batch(100)
  print i
  if i%50 == 0:
    summary, acc = sess.run([merged, accuracy], feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    test_writer.add_summary(summary, i)
    # train_accuracy = accuracy.eval(feed_dict={
    #     x:batch[0], y_: batch[1], keep_prob: 1.0})
    # print("step %d, training accuracy %g"%(i, train_accuracy))
  #   summary, _ = sess.run([merged,accuracy],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  #   train_writer.add_summary(summary, i)
  # else:  
  summary, _ = sess.run([merged,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

