from .tf_layers import *


class MobileInvertedResidualBlock:

    def __init__(self, _id, mobile_inverted_conv, has_residual):
        self.id = _id
        self.mobile_inverted_conv = mobile_inverted_conv
        self.has_residual = has_residual

    def build(self, _input, net, init=None):
        output = _input
        with tf.variable_scope(self.id):
            output = self.mobile_inverted_conv.build(output, net, init)
            if self.has_residual:
                output = output + _input
        return output


class ProxylessNASNets:

    def __init__(self, net_config, net_weights=None, graph=None, sess=None, is_training=True, images=None,
                 img_size=None, n_classes=1001):
        if graph is not None:
            self.graph = graph
            slim = True
        else:
            self.graph = tf.Graph()
            slim = False

        self.net_config = net_config
        self.n_classes = n_classes

        with self.graph.as_default():
            self._define_inputs(slim=slim, is_training=is_training, images=images, img_size=img_size)
            logits = self.build(init=net_weights)
            self.logits = logits
            soft_logit = tf.nn.softmax(logits, dim=1)

            prediction = logits
            # losses
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.labels))
            self.cross_entropy = cross_entropy

            correct_prediction = tf.equal(
                tf.argmax(prediction, 1),
                tf.argmax(self.labels, 1)
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

            self.global_variables_initializer = tf.global_variables_initializer()
        self._initialize_session(sess)

    @property
    def bn_eps(self):
        return self.net_config['bn']['eps']

    @property
    def bn_decay(self):
        return 1 - self.net_config['bn']['momentum']

    def _initialize_session(self, sess):
        """ Initialize session, variables """
        config = tf.ConfigProto()  # allow_soft_placement=True, log_device_placement=False
        # restrict model GPU memory utilization to min required
        # config.gpu_options.allow_growth = True
        if sess is None:
            # config.gpu_options.visible_device_list = str(mgw.local_rank())
            self.sess = tf.Session(graph=self.graph, config=config)
        else:
            self.sess = sess
        self.sess.run(self.global_variables_initializer)

    def _define_inputs(self, slim=False, is_training=True, images=None, img_size=None):
        if isinstance(img_size, list) or isinstance(img_size, tuple):
            assert len(img_size) == 2
            shape = [None, img_size[0], img_size[1], 3]
        else:
            shape = [None, img_size, img_size, 3]
        if images is not None:
            self.images = images
        else:
            self.images = tf.placeholder(
                tf.float32,
                shape=shape,
                name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        # self.learning_rate = tf.placeholder(
        #     tf.float32,
        #     shape=[],
        #     name='learning_rate')
        if slim:
            self.is_training = is_training
        else:
            self.is_training = tf.placeholder(
                tf.bool, shape=[], name='is_training')

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def build(self, init=None):
        output = self.images
        if init is not None:
            for key in init:
                init[key] = tf.constant_initializer(init[key])

        # first conv
        first_conv = ConvLayer(
            'first_conv',
            self.net_config['first_conv']['out_channels'],
            3,
            2)
        output = first_conv.build(output, self, init)

        for i, block_config in enumerate(self.net_config['blocks']):
            if block_config['mobile_inverted_conv']['name'] == 'ZeroLayer':
                continue
            mobile_inverted_conv = MBInvertedConvLayer(
                'mobile_inverted_conv',
                block_config['mobile_inverted_conv']['out_channels'],
                block_config['mobile_inverted_conv']['kernel_size'],
                block_config['mobile_inverted_conv']['stride'],
                block_config['mobile_inverted_conv']['expand_ratio'],
            )
            if block_config['shortcut'] is None or block_config['shortcut']['name'] == 'ZeroLayer':
                has_residual = False
            else:
                has_residual = True
            block = MobileInvertedResidualBlock(
                'blocks/%d' %
                i, mobile_inverted_conv, has_residual)
            output = block.build(output, self, init)

        # feature mix layer
        if self.net_config['feature_mix_layer'] is not None:
            feature_mix_layer = ConvLayer(
                'feature_mix_layer',
                self.net_config['feature_mix_layer']['out_channels'],
                1,
                1)
            output = feature_mix_layer.build(output, self, init)
        # print(output.get_shape()[1])
        output = avg_pool(output, output.get_shape()[1], output.get_shape()[2])
        # output = flatten(output)
        # classifier = LinearLayer(
        #     'classifier',
        #     self.n_classes,
        #     self.net_config['classifier']['dropout_rate'])
        classifier = ConvLayer_fc(
            'classifier',
            self.n_classes,
            1,
            1)
        output = classifier.build(output, self, init)
        output = tf.reshape(output, shape=[-1, self.n_classes])
        return output

    # def build(self, init=None):
    #     output = self.images
    #     if init is not None:
    #         for key in init:
    #             init[key] = tf.constant_initializer(init[key])
    #
    #     # first conv
    #     first_conv = ConvLayer(
    #         'Conv',
    #         self.net_config['first_conv']['out_channels'],
    #         3,
    #         2)
    #     output = first_conv.build(output, self, init)
    #
    #     for i, block_config in enumerate(self.net_config['blocks']):
    #         if block_config['mobile_inverted_conv']['name'] == 'ZeroLayer':
    #             continue
    #         mobile_inverted_conv = MBInvertedConvLayer(
    #             '',
    #             block_config['mobile_inverted_conv']['out_channels'],
    #             block_config['mobile_inverted_conv']['kernel_size'],
    #             block_config['mobile_inverted_conv']['stride'],
    #             block_config['mobile_inverted_conv']['expand_ratio'],
    #         )
    #         if block_config['shortcut'] is None or block_config['shortcut']['name'] == 'ZeroLayer':
    #             has_residual = False
    #         else:
    #             has_residual = True
    #         if i == 0:
    #             block = MobileInvertedResidualBlock(
    #                 'expanded_conv'
    #                 , mobile_inverted_conv, has_residual)
    #         elif i <= 3:
    #             block = MobileInvertedResidualBlock(
    #                 'expanded_conv_%d' %
    #                 i, mobile_inverted_conv, has_residual)
    #         else:
    #             block = MobileInvertedResidualBlock(
    #                 'expanded_conv_%d' %
    #                 (i-2), mobile_inverted_conv, has_residual)
    #         output = block.build(output, self, init)
    #
    #     # feature mix layer
    #     feature_mix_layer = ConvLayer(
    #         'Conv_1',
    #         self.net_config['feature_mix_layer']['out_channels'],
    #         1,
    #         1)
    #     output = feature_mix_layer.build(output, self, init)
    #
    #     output = avg_pool(output, 7, 7)
    #     output = flatten(output)
    #     classifier = LinearLayer(
    #         'Logits/Conv2d_1c_1x1',
    #         self.n_classes,
    #         self.net_config['classifier']['dropout_rate'])
    #     output = classifier.build(output, self, init)
    #     return output
