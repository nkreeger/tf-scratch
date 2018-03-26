import tensorflow as tf

with tf.Graph().as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    a = tf.constant([0.14773492515087128, -0.11258933693170547, 0.00])

    greater_t = tf.greater(a, 0.0)
    ones_t = tf.ones(tf.shape(a))
    alpha = tf.constant(0.55555, shape=a.shape)
    d = tf.where(greater_t, ones_t, alpha)

    print d.eval()
