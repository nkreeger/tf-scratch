import tensorflow as tf
import os

def generate_gol_example(size):
  if size < 3:
    raise ValueError("Size must be greater than 2, received %d" % size)

  with tf.name_scope("generate_gol_example"):
    world = tf.random_uniform(
        (size - 2, size - 2), minval=0, maxval=2, dtype=tf.int32)
    world_padded = tf.pad(world, [[1, 1], [1, 1]])

    num_neighbors = (
        world_padded[:-2, :-2] + world_padded[:-2, 1:-1] + world_padded[:-2, 2:]
        + world_padded[1:-1, :-2] + world_padded[1:-1, 2:] +
        world_padded[2:, :-2] + world_padded[2:, 1:-1] + world_padded[2:, 2:])

    cell_survives = tf.logical_or(
        tf.equal(num_neighbors, 3), tf.equal(num_neighbors, 2))
    cell_rebirths = tf.equal(num_neighbors, 3)

    survivors = tf.where(cell_survives, world_padded[1:-1, 1:-1],
                         tf.zeros_like(world))
    world_next = tf.where(cell_rebirths, tf.ones_like(world), survivors)

    world_next_padded = tf.pad(world_next, [[1, 1], [1, 1]])

    return world_padded, world_next_padded


def main():
  size = 5
  dir = os.path.dirname(os.path.realpath(__file__))

  with tf.Graph().as_default():
    input_world = tf.placeholder(tf.float32, [1, size * size])
    target_world = tf.placeholder(tf.float32, [1, size * size])

    hidden_layer = tf.contrib.layers.fully_connected(
        inputs=input_world, num_outputs=(size * size))
    hidden_layer = tf.contrib.layers.fully_connected(
        inputs=hidden_layer, num_outputs=(size * size))
    hidden_layer = tf.contrib.layers.fully_connected(
        inputs=hidden_layer, num_outputs=(size * size),
        activation_fn=tf.nn.sigmoid)

    loss = tf.losses.log_loss(target_world, hidden_layer)

    prediction = tf.to_int32(tf.greater(hidden_layer, 0.5))
    prediction = tf.reshape(prediction, [size, size])

    optimizer = tf.train.AdagradOptimizer(0.01)
    train = optimizer.minimize(loss)

    saver = tf.train.Saver()


    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      print '... Generating samples'
      sample_sets = []
      samples = 1000
      for i in range(samples):
        if i % 10 == 0:
          print '{0} samples'.format(i)
        world, world_next = generate_gol_example(size)
        sample_sets.append(tf.to_float(tf.reshape(world, [1, -1])).eval())
        sample_sets.append(tf.to_float(tf.reshape(world_next, [1, -1])).eval())

      print '... Training'
      for i in range(samples):
        #world, world_next = generate_gol_example(size)
        #world_flat = tf.to_float(tf.reshape(world, [1, -1])).eval()
        #world_next_flat = tf.to_float(tf.reshape(world_next, [1, -1])).eval()

        world_next_flat = sample_sets.pop()
        world_flat = sample_sets.pop()
        _, loss_value = sess.run([train, loss], feed_dict={
            input_world: world_flat, target_world: world_next_flat})

        if i % 10 == 0:
          print '{0} @ Step {1}'.format(loss_value, i)

      # 1.) Export as 'Frozen Graph' - using inference model format.
      tf.train.write_graph(sess.graph_def, '.', 'gol.pbtxt')
      tf.train.write_graph(sess.graph_def, '.', 'gol.pb', False)

      # 2.) Export as Saved Model (model saver TODO)
      saver.save(sess, dir + '/gol-data/gol')

if __name__ == '__main__':
  main()
