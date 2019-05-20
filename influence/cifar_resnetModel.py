from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import numpy as np
import os.path as osp
import tensorflow as tf
import math

from influence.genericNeuralNet import GenericNeuralNet, variable_with_weight_decay

from models.nn import CIFAR_ResNet_20 as ConvNet
from models.layers import global_avg_pool


class CIFAR_ResNetModel(GenericNeuralNet):

    def __init__(self, img_side, num_channels, feature_layer_name, weight_decay, ckpt_dir, **kwargs):
        self.weight_decay = weight_decay

        self.img_side = img_side
        self.num_channels = num_channels
        self.feature_layer_name = feature_layer_name
        self.input_dim = img_side * img_side * num_channels
        self.num_features = 64  # Hardcoded for CIFAR_ResNet20
        self.ckpt_dir = ckpt_dir

        super(CIFAR_ResNetModel, self).__init__(**kwargs)

        self.load_weights()
        self.set_params_op = self.set_params()

    def get_all_params(self):
        all_params = []
        for layer in ['softmax_linear']:
            # for var_name in ['weights', 'biases']:
            for var_name in ['weights']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                all_params.append(temp_tensor)
        return all_params

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

    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed
        }
        return feed_dict

    def load_weights(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, osp.join(self.ckpt_dir, 'model.ckpt'))

    def inference(self, input):
        reshaped_input = tf.reshape(input, [-1, self.img_side, self.img_side, self.num_channels])
        reshaped_labels = tf.one_hot(self.labels_placeholder, depth=self.num_classes)
        self.model = ConvNet(reshaped_input, reshaped_labels, self.num_classes)
        raw_features = self.model.d[self.feature_layer_name]

        pooled_features = global_avg_pool(raw_features)
        dim = int(np.prod(pooled_features.get_shape().as_list()[1:]))
        self.features = tf.reshape(pooled_features, (-1, dim))

        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights',
                [self.num_features],
                stddev=1.0 / math.sqrt(float(self.num_features)),
                wd=self.weight_decay)

            logits = tf.matmul(self.features, tf.reshape(weights, [-1, 1]))
            zeros = tf.zeros_like(logits)
            logits_with_zeros = tf.concat([zeros, logits], 1)

        self.weights = weights

        return logits_with_zeros

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds

    def set_params(self):
        # See if we can automatically infer weight shape
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.num_features],
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]

