import tensorflow as tf
import numpy as np


class DQN2:
    def __init__(self, session, input_size, output_size, name='main'):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def weight_variable(self, name, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(name=name, initial_value=initial)
        # return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, name, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(name=name, initial_value=initial)
        # return tf.constant(0.01, shape=shape)

    def conv2d(self, name, x, W, stride):
        # return tf.nn.convolution(x, W, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.conv2d(input=x, filter=W, strides=[1, stride, stride, 1], padding="SAME", name=name)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def _build_network(self, l_rate=1e-6):

        # self._X = tf.placeholder(tf.float32, [None, self.input_size], name='input_x')
        self._X = tf.placeholder(tf.float32, [None, 23, 10, 1], name='input_x')

        W1 = self.weight_variable('W1', [8, 8, 1, 32])
        b1 = self.bias_variable('b1', [32])

        conv1 = tf.nn.relu(self.conv2d('conv1', self._X, W1, 4) + b1)
        pool1 = self.max_pool_2x2(conv1)

        ## layer2
        W2 = self.weight_variable('W2', [4, 4, 32, 32])
        b2 = self.bias_variable('b2', [32])

        conv2 = tf.nn.relu(self.conv2d('conv2', pool1, W2, 2) + b2)
        # h_pool2 = max_pool_2x2(h_conv2)

        ## layer3
        W3 = self.weight_variable('W3', [3, 3, 32, 64])
        b3 = self.bias_variable('b3', [64])

        conv3 = tf.nn.relu(self.conv2d('conv3', conv2, W3, 1) + b3)
        # h_pool3 = max_pool_2x2(h_conv3)

        ## layer1
        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        # conv3_flat = tf.reshape(conv3, [-1, 1600])
        conv3_flat = tf.reshape(conv3, [-1, 128])

        # W_fc1 = self.weight_variable('W_fc1', [1600, 512])
        W_fc1 = self.weight_variable('W_fc1', [128, 64])
        b_fc1 = self.bias_variable('b_fc1', [64])

        W_fc2 = self.weight_variable('W_fc2', [64, self.output_size])
        b_fc2 = self.bias_variable('b_fc2', [self.output_size])

        self.fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)
        self.readout = tf.matmul(self.fc1, W_fc2) + b_fc2

        a = tf.placeholder("float", [None, self.output_size])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(self.readout, a), reduction_indices=1)
        # readout_action = tf.reduce_sum(tf.matmul(readout, a), reduction_indices=1)
        self._loss = tf.reduce_mean(tf.square(y - readout_action))
        self._train = tf.train.AdamOptimizer(1e-6).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [23, 10, 1])
        predict_value = self.session.run(self.readout, feed_dict={self._X: [x]})
        return predict_value

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

if __name__ == "__main__":
    with tf.Session() as sess:
        dqn = DQN2(sess, 23*10, 5, 'name')

        init_op = tf.global_variables_initializer()
        init_op.run()

        value = dqn.predict(np.zeros([23, 10]))[0]
        print(value)

