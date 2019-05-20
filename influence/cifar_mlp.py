from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import numpy as np
import os.path as osp
import tensorflow as tf
import math

from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from influence.dataset import DataSet

from models.nn import CIFAR_ResNet_20 as ConvNet
from models.layers import global_avg_pool


class CIFAR_MLP(GenericNeuralNet):
    """ The last FC layer of CIFAR_ResNet20 """

    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(CIFAR_MLP, self).__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        for layer in ['fc6']:
            for var_name in ['weights', 'biases']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                all_params.append(temp_tensor)
        return all_params

    def retrain(self, num_steps, feed_dict):
        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])

        for step in range(num_steps):
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def inference(self, input_x):
        with tf.variable_scope('fc6'):
            weights = variable(
                'weights',
                [self.input_dim * self.num_classes],
                tf.initializers.truncated_normal(mean=0.0, stddev=0.01))
            biases = variable(
                'biases',
                [self.num_classes],
                tf.initializers.constant(value=0.01))

            logits = tf.matmul(input_x, tf.reshape(weights, [self.input_dim, self.num_classes])) + biases

        return logits

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds

