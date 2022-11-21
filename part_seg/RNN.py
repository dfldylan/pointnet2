import tensorflow as tf


class FluidRNN(object):
    def __init__(self, time_step, rnn_unit, input_size, output_size, lr):
        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size], name='rnn_x')
        Y = tf.placeholder(tf.float32, shape=[None, output_size], name='rnn_y')
        with tf.variable_scope('sec_lstm'):
            batch_size = tf.shape(X)[0]
            time_step = tf.shape(X)[1]
            input = tf.reshape(X, [-1, time_step, input_size])
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
            # init_state = cell.zero_state(batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
            value = tf.transpose(output_rnn, [1, 0, 2])
            last = value[-1]
            pred = tf.layers.dense(last, output_size, name='rnn_fcn')

            loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
            train_op = tf.train.AdamOptimizer(lr, name='rnn_op').minimize(loss)


        self.opt = {'X': X, 'Y': Y, 'pred': pred, 'loss': loss, 'train_op': train_op}

    def __call__(self, sess, train_part, test_part):
        _, pred, loss = sess.run([self.opt['train_op'], self.opt['pred'], self.opt['loss']],
                 feed_dict={self.opt['X']: train_part, self.opt['Y']: test_part})
        return loss, pred