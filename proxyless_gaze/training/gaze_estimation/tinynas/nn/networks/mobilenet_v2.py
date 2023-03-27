from .proxyless_nets import ProxylessNASNets, MobileInvertedResidualBlock
from ..modules import *
from utils import make_divisible, val2list

__all__ = ['MobileNetV2']


class MobileNetV2(ProxylessNASNets):

    def __init__(self, n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0.2,
                 ks=None, expand_ratio=None, depth_param=None, stage_width_list=None, no_mix_layer=False,
                 disable_keep_last_channel=False):

        if ks is None:
            ks = 3
        if expand_ratio is None:
            expand_ratio = 6

        input_channel = 32
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult, 8)
        if disable_keep_last_channel:
            last_channel = make_divisible(last_channel * width_mult, 8)
        else:
            last_channel = make_divisible(last_channel * width_mult,
                                          8) if width_mult > 1.0 else last_channel

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expand_ratio, 24, 2, 2],
            [expand_ratio, 32, 3, 2],
            [expand_ratio, 64, 4, 2],
            [expand_ratio, 96, 3, 1],
            [expand_ratio, 160, 3, 2],
            [expand_ratio, 320, 1, 1],
        ]

        if depth_param is not None:
            assert isinstance(depth_param, int)
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        ks = val2list(ks, sum([n for _, _, n, _ in inverted_residual_setting]) - 1)
        _pt = 0

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        # inverted residual blocks
        blocks = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, 8)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if t == 1:
                    kernel_size = 3
                else:
                    kernel_size = ks[_pt]
                    _pt += 1
                mobile_inverted_conv = MBInvertedConvLayer(
                    in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                    expand_ratio=t,
                )
                if t > 1 and stride == 1:  # NOTICE: we enforce no residual for the first block
                    if input_channel == output_channel:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                else:
                    shortcut = None
                blocks.append(
                    MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                )
                input_channel = output_channel
        # 1x1_conv before global average pooling
        if no_mix_layer:
            feature_mix_layer = None
            classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)
        else:
            feature_mix_layer = ConvLayer(
                input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
            )

            classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileNetV2, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
