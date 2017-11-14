import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, input_size, output_size, name='main'):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def _build_network(self, h_size=256, l_rate=1e-1):
        with tf.variable_scope(self.net_name):

            self._X = tf.placeholder(
                tf.float32, [None, self.input_size], name='input_x'
            )

            W1 = tf.get_variable('W1', shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            # layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            layer1 = tf.nn.relu(tf.matmul(self._X, W1))

            W2 = tf.get_variable('W2', shape=[h_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            # layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
            layer2 = tf.nn.relu(tf.matmul(layer1, W2))

            W3 = tf.get_variable("W3", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            self._Qpred = tf.matmul(layer2, W3)

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        self._train = tf.train.AdamOptimizer(
            learning_rate=l_rate
        ).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        predict_value = self.session.run(self._Qpred, feed_dict={self._X: x})
        # print('predict_value', predict_value)

        return predict_value

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack, self._Y: y_stack})
