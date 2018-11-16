import tensorflow as tf
from tensorflow.python import debug as tf_debug


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = x + y

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

op1 = sess.run(z, feed_dict={x: 3, y: 4.5})
print("Result:", op1)
