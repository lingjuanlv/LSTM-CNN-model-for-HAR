#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : model.py
import tensorflow as tf
import numpy as np
import random


class HAR(object):
    """"""
    def __init__(self,
                 rnn_hidden,
                 num_layers,
                 cnn_filter,
                 classes,
                 batch_size,
                 steps,
                 features,
                 lr,
                 dtype=tf.float32,
                 bidir=False):
        """HAR constructor
        
        Args:
          rnn_hidden: hidden units for rnn cell
        """
        self.batch_size = batch_size
        self.steps = steps
        self.features = features
        self.num_layers = num_layers
        self.rnn_hidden = rnn_hidden
        self.lr = lr

        self.inputs = tf.placeholder(shape=[self.batch_size,
                                            self.steps,
                                            self.features],
                                     dtype=dtype)
        self.labels = tf.placeholder(shape=[self.batch_size], dtype=tf.int64)
        self.dropout = tf.placeholder(dtype)

        xavier_init = tf.contrib.layers.xavier_initializer(seed=1)

        if bidir:
            num_cell = 2
        else:
            num_cell = 1

        # RNN
        with tf.variable_scope("RNN", initializer=xavier_init):
            cell = [tf.nn.rnn_cell.BasicLSTMCell(rnn_hidden, state_is_tuple=True) for _ in range(num_cell)]
            cell = [tf.nn.rnn_cell.DropoutWrapper(cell[i], output_keep_prob=self.dropout) for i in range(num_cell)]
            cell = [tf.nn.rnn_cell.MultiRNNCell([cell[i]] * num_layers, state_is_tuple=True) for i in range(num_cell)]
            initial_state = [cell[i].zero_state(self.batch_size, dtype=dtype) for i in range(num_cell)]

            if bidir:
                rnn_outputs, _, _ = tf.nn.bidirectional_rnn(cell[0],
                                                            cell[1],
                                                            tf.unpack(self.inputs, axis=1),
                                                            initial_state[0],
                                                            initial_state[1])
                rnn_hidden *= 2
            else:
                rnn_outputs, _ = tf.nn.rnn(cell[0],
                                                   tf.unpack(self.inputs, axis=1),
                                                   initial_state[0])

        # CNN 
        with tf.variable_scope("CNN", initializer=xavier_init):
            inputs = tf.transpose(tf.pack(rnn_outputs), perm=[1, 0, 2])
            inputs = tf.reshape(inputs, shape=[batch_size, steps, rnn_hidden, -1])
            n_grams = []
            for kernel, num in cnn_filter:
                conv2_w = tf.get_variable("conv2_w_%d" % kernel,
                                          shape=[kernel, rnn_hidden, 1, num]) 
                conv2_b = tf.get_variable("conv2_b_%d" % kernel, 
                                          initializer=tf.zeros_initializer([num],
                                                                           dtype=dtype))

                cnn_layer1 = tf.nn.conv2d(inputs, conv2_w,
                                          strides=[1,1,1,1], padding='VALID') 
                cnn_layer1 = tf.nn.relu(cnn_layer1 + conv2_b)
                n_grams.append(tf.reduce_max(cnn_layer1, [1,2]))

            cnn_outputs = tf.concat(1, n_grams)
        with tf.variable_scope("output", initializer=xavier_init):
            output_units = cnn_outputs.get_shape().as_list()[1]
            self.output_w = tf.get_variable("output_w",
                                            shape=[output_units,
                                                   classes])
            output_b = tf.get_variable("output_b",
                                       initializer=tf.zeros_initializer([classes],
                                                                        dtype=dtype))
            outputs = tf.matmul(cnn_outputs, self.output_w) + output_b
            pred = tf.nn.dropout(outputs, self.dropout)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred,
                                                                                      self.labels))

            correct_pred = tf.equal(tf.argmax(pred,1), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.update = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.saver = tf.train.Saver(tf.all_variables())

    def get_batch(self, dataset, isTrain):
        if isTrain:
            random.shuffle(dataset)
        else:
            dataset = random.sample(dataset, self.batch_size)

        batches = len(dataset) // self.batch_size
        for i in range(batches):
            st = i*self.batch_size
            ed = (i+1)*self.batch_size
            yield self.pack(dataset[st:ed])


    def pack(self, minibatch):
        xs, ys = [], []
        for x, y in minibatch:
            xs.append(x)
            ys.append(y)

        return (np.array(xs, dtype=np.float32),
                np.array(ys, dtype=np.int32))

    def get_weights(self, sess):
        weights = sess.run([self.Weights, self.conv2_w, self.output_w])
        return weights

    def step(self, sess, inputs, labels, dropout, isTrain):
        feed_dict = {self.inputs: inputs,
                     self.labels: labels,
                     self.dropout: dropout}
        if isTrain:
            _, loss, accu = sess.run([self.update, self.loss, self.accuracy],
                                      feed_dict=feed_dict)
            return loss, accu
        else:
            accu = sess.run(self.accuracy, feed_dict=feed_dict)
            return accu