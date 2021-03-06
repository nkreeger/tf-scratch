import argparse
import numpy as np
import os
import tensorflow as tf

# manually put back imported modules
# https://github.com/tensorflow/tensorflow/issues/15410
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

def calc_world_next(world, size):
  world_next = np.zeros_like(world)
  for i in range(1, size - 1):
    for j in range(1, size - 1):
      num_neighbors = np.sum(world[i-1:i+2, j-1:j+2]) - world[i, j]
      if num_neighbors == 3:
        world_next[i, j] = 1
      elif num_neighbors == 2:
        world_next[i, j] = world[i, j]
  return world_next


def calc_world(size):
  world = np.random.random_integers(0, 1, [size - 2, size -2])
  world = np.lib.pad(world, (1, 1), 'constant')
  return world, calc_world_next(world, size)


def train_model(size, saver, dir, train, input_world, target_world, prediction, loss, sess):
  for i in range(1):
    world, world_next = calc_world(size)

    _, loss_value = sess.run([train, loss], feed_dict={
        input_world: world.reshape(1, size * size),
        target_world: world_next.reshape(1, size * size)})

    if i % 300 == 0:
      print '{0} @ Step {1}'.format(loss_value, i)

  # 1.) Export as 'Frozen Graph' - using inference model format.
  tf.train.write_graph(sess.graph_def, '.', 'gol.pbtxt')
  tf.train.write_graph(sess.graph_def, '.', 'gol.pb', False)

  # 2.) Export as Saved Model (model saver TODO)
  saver.save(sess, dir + '/gol-data/gol')

  # 3.) Save as TFLite format
  print 'prediction {}'.format(prediction)
  tflite_model = tf.contrib.lite.toco_convert(sess.graph_def,
                                              [input_world],
                                              [prediction])
  open("gol.tflite", "wb").write(tflite_model)


def infer(size, saver, dir, sess, prediction, input_world):
  # Infer a couple of examples:
  for i in range(5):
    world, world_next = calc_world(size)

    print 'prediction'
    print sess.run(prediction, feed_dict={input_world: world.reshape(1, size * size)})
    print 'world_next'
    print world_next
    print '---------------------'


def main(should_train):
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

      saver.restore(sess, dir + '/gol-data/gol')
      print 'model restored'

      if should_train == True:
        train_model(size, saver, dir, train, input_world, target_world, prediction, loss, sess)
      else:
        infer(size, saver, dir, sess, prediction, input_world)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', action='store_true', help='Perform training')

  args = parser.parse_args()
  main(args.train)

