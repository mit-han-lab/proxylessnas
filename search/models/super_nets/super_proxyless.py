# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from queue import Queue
import copy

from modules.mix_op import *
from models.normal_nets.proxyless_nets import *
from utils import LatencyEstimator


class SuperProxylessNASNets(ProxylessNASNets):

    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0):
        self._redundant_modules = None
        self._unused_modules = None

        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )

        # first block
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'],
            input_channel, first_cell_width, 1, 'weight_bn_act',
        ), )
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # blocks
        blocks = [first_block]
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # conv
                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act',
                ), )
                # shortcut
                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

        # feature mix layer
        last_channel = make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)
        super(SuperProxylessNASNets, self).__init__(first_conv, blocks, feature_mix_layer, classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, SuperProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_latency(self, latency_model: LatencyEstimator):
        expected_latency = 0
        # first conv
        expected_latency += latency_model.predict('Conv', [224, 224, 3], [112, 112, self.first_conv.out_channels])
        # feature mix layer
        expected_latency += latency_model.predict(
            'Conv_1', [7, 7, self.feature_mix_layer.in_channels], [7, 7, self.feature_mix_layer.out_channels]
        )
        # classifier
        expected_latency += latency_model.predict(
            'Logits', [7, 7, self.classifier.in_features], [self.classifier.out_features]  # 1000
        )
        # blocks
        fsize = 112
        for block in self.blocks:
            shortcut = block.shortcut
            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1

            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                if not mb_conv.is_zero_layer():
                    out_fz = fsize // mb_conv.stride
                    op_latency = latency_model.predict(
                        'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
                        expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
                    )
                    expected_latency = expected_latency + op_latency
                    fsize = out_fz
                continue

            probs_over_ops = mb_conv.current_prob_over_ops
            out_fsize = fsize
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                out_fsize = fsize // op.stride
                op_latency = latency_model.predict(
                    'expanded_conv', [fsize, fsize, op.in_channels], [out_fsize, out_fsize, op.out_channels],
                    expand=op.expand_ratio, kernel=op.kernel_size, stride=op.stride, idskip=idskip
                )
                expected_latency = expected_latency + op_latency * probs_over_ops[i]
            fsize = out_fsize
        return expected_latency

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop, x = self.first_conv.get_flops(x)
        expected_flops += flop
        # blocks
        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops = expected_flops + shortcut_flop

            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops = expected_flops + op_flops * probs_over_ops[i]
            x = block(x)
        # feature mix layer
        delta_flop, x = self.feature_mix_layer.get_flops(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return ProxylessNASNets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)
