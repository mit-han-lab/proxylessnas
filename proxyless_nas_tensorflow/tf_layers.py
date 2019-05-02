import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import avg_pool2d


def conv2d(
        _input,
        out_features,
        kernel_size,
        stride=1,
        padding='SAME',
        param_initializer=None):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.variable_scope('conv'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.get_variable(
            name='weight',
            shape=[
                kernel_size,
                kernel_size,
                in_features,
                out_features],
            initializer=initializer)
        if padding == 'SAME':
            pad = kernel_size // 2
            paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
            output = tf.pad(output, paddings, 'CONSTANT')

        output = tf.nn.conv2d(
            output, weight, [
                1, stride, stride, 1], 'VALID', data_format='NHWC')
    return output


def depthwise_conv2d(
        _input,
        kernel_size,
        stride=1,
        padding='SAME',
        param_initializer=None):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input
    with tf.variable_scope('conv'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.get_variable(
            name='weight', shape=[kernel_size, kernel_size, in_features, 1],
            initializer=initializer
        )
        if padding == 'SAME':
            pad = kernel_size // 2
            paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
            output = tf.pad(output, paddings, 'CONSTANT')

        output = tf.nn.depthwise_conv2d(
            output, weight, [
                1, stride, stride, 1], 'VALID', data_format='NHWC')
    return output


def avg_pool(_input, k=2, s=2):
    padding = 'VALID'
    output = avg_pool2d(
        _input, kernel_size=[
            k, k], stride=[
            s, s], padding=padding, data_format='NHWC')
    return output


def fc_layer(_input, out_units, use_bias=False, param_initializer=None):
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.variable_scope('linear'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(
            init_key, tf.contrib.layers.xavier_initializer())
        weight = tf.get_variable(
            name='weight', shape=[features_total, out_units],
            initializer=initializer
        )
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = '%s/bias' % tf.get_variable_scope().name
            initializer = param_initializer.get(
                init_key, tf.constant_initializer(
                    [0.0] * out_units))
            bias = tf.get_variable(
                name='bias', shape=[out_units],
                initializer=initializer
            )
            output = output + bias
    return output


def batch_norm(
        _input,
        is_training,
        epsilon=1e-3,
        decay=0.9,
        param_initializer=None):
    with tf.variable_scope('bn'):
        scope = tf.get_variable_scope().name
        bn_init = {
            'beta': param_initializer['%s/bias' % scope],
            'gamma': param_initializer['%s/weight' % scope],
            'moving_mean': param_initializer['%s/running_mean' % scope],
            'moving_variance': param_initializer['%s/running_var' % scope],
        }
        output = tf.contrib.layers.batch_norm(
            _input,
            scale=True,
            is_training=is_training,
            param_initializers=bn_init,
            updates_collections=None,
            epsilon=epsilon,
            decay=decay,
            data_format='NHWC',
        )
    return output


def activation(_input, activation='relu6'):
    if activation == 'relu6':
        return tf.nn.relu6(_input)
    else:
        raise ValueError('Do not support %s' % activation)


def flatten(_input):
    input_shape = _input.shape.as_list()
    if len(input_shape) != 2:
        return tf.reshape(_input, [-1, np.prod(input_shape[1:])])
    else:
        return _input


def dropout(_input, keep_prob, is_training):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
        )
    else:
        output = _input
    return output


class MBInvertedConvLayer:
    def __init__(
            self,
            _id,
            filter_num,
            kernel_size=3,
            stride=1,
            expand_ratio=6):
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio

    def build(self, _input, net, init=None):
        output = _input
        in_features = int(_input.get_shape()[3])
        with tf.variable_scope(self.id):
            if self.expand_ratio > 1:
                feature_dim = round(in_features * self.expand_ratio)
                with tf.variable_scope('inverted_bottleneck'):
                    output = conv2d(
                        output, feature_dim, 1, 1, param_initializer=init)
                    output = batch_norm(
                        output,
                        net.is_training,
                        epsilon=net.bn_eps,
                        decay=net.bn_decay,
                        param_initializer=init)
                    output = activation(output, 'relu6')
            with tf.variable_scope('depth_conv'):
                output = depthwise_conv2d(
                    output,
                    self.kernel_size,
                    self.stride,
                    param_initializer=init)
                output = batch_norm(
                    output,
                    net.is_training,
                    epsilon=net.bn_eps,
                    decay=net.bn_decay,
                    param_initializer=init)
                output = activation(output, 'relu6')
            with tf.variable_scope('point_linear'):
                output = conv2d(output, self.filter_num, 1,
                                1, param_initializer=init)
                output = batch_norm(
                    output,
                    net.is_training,
                    epsilon=net.bn_eps,
                    decay=net.bn_decay,
                    param_initializer=init)
        return output


class ConvLayer:
    def __init__(self, _id, filter_num, kernel_size=3, stride=1):
        self.id = _id
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, _input, net, init=None):
        output = _input
        with tf.variable_scope(self.id):
            output = conv2d(
                output,
                self.filter_num,
                self.kernel_size,
                self.stride,
                param_initializer=init)
            output = batch_norm(
                output,
                net.is_training,
                epsilon=net.bn_eps,
                decay=net.bn_decay,
                param_initializer=init)
            output = activation(output, 'relu6')
        return output


class LinearLayer:
    def __init__(self, _id, n_units, drop_rate=0):
        self.id = _id
        self.n_units = n_units
        self.drop_rate = drop_rate

    def build(self, _input, net, init=None):
        output = _input
        with tf.variable_scope(self.id):
            if self.drop_rate > 0:
                output = dropout(output, 1 - self.drop_rate, net.is_training)
            output = fc_layer(
                output,
                self.n_units,
                use_bias=True,
                param_initializer=init)
        return output
