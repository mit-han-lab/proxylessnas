import random
import copy
from queue import Queue

from .layers import *
from .utils import *


def set_block_from_config(config):
    name2block = {
        TransitionBlock.__name__: TransitionBlock,
        ResidualTreeBlock.__name__: ResidualTreeBlock,
    }

    block_name = config.pop('name')
    block = name2block[block_name]
    return block.build_from_config(config)


class TreeNode(BasicUnit):
    def __init__(self, edges, child_nodes, in_channels, out_channels,
                 split_type='copy', merge_type='add', has_branch_bn=False, path_drop_rate=0):
        super(TreeNode, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_type = split_type
        self.merge_type = merge_type

        self.has_branch_bn = has_branch_bn

        self.path_drop_rate = path_drop_rate

        assert len(edges) == len(child_nodes)

        """ add modules """
        self.edges = nn.ModuleList(edges)
        self.child_nodes = nn.ModuleList(child_nodes)

        # branch batch norm (skip node bn)
        if self.has_branch_bn:
            branch_bns = []
            for out_dim in self.out_dim_list:
                branch_bns.append(nn.BatchNorm2d(out_dim))
        else:
            branch_bns = [None] * self.child_num
        self.branch_bns = nn.ModuleList(branch_bns)

    @property
    def child_num(self):
        return len(self.edges)

    @property
    def in_dim_list(self):
        if self.split_type == 'copy':
            in_dim_list = [self.in_channels] * self.child_num
        elif self.split_type == 'split':
            in_dim_list = get_split_list(self.in_channels, self.child_num)
        else:
            assert self.child_num == 1
            in_dim_list = [self.in_channels]
        return in_dim_list

    @property
    def out_dim_list(self):
        if self.merge_type == 'add':
            out_dim_list = [self.out_channels] * self.child_num
        elif self.merge_type == 'concat':
            out_dim_list = get_split_list(self.out_channels, self.child_num)
        else:
            assert self.child_num == 1
            out_dim_list = [self.out_channels]
        return out_dim_list

    def get_node(self, path2node):
        node = self
        for step in path2node:
            node = node.child_nodes[step]
        return node

    def allocation_scheme(self, x):
        if self.split_type == 'copy':
            child_inputs = [x] * self.child_num
        elif self.split_type == 'split':
            child_inputs, _pt = [], 0
            for seg_size in self.in_dim_list:
                seg_x = x[:, _pt:_pt + seg_size, :, :].contiguous()  # split in the channel dimension
                child_inputs.append(seg_x)
                _pt += seg_size
        else:
            child_inputs = [x]
        return child_inputs

    def merge_scheme(self, child_outputs):
        if self.merge_type == 'concat':
            output = torch.cat(child_outputs, dim=1)
        elif self.merge_type == 'add':
            output = list_sum(child_outputs)
        else:
            assert len(child_outputs) == 1
            output = child_outputs[0]
        return output

    @staticmethod
    def path_normal_forward(x, edge=None, child=None, branch_bn=None):
        if edge is not None:
            x = edge(x)
        edge_x = x
        if child is not None:
            x = child(x)
        if branch_bn is not None:
            x = branch_bn(x)
            x = x + edge_x
        return x

    def path_drop_forward(self, x, branch_idx):
        edge, child, branch_bn = self.edges[branch_idx], self.child_nodes[branch_idx], self.branch_bns[branch_idx]

        if self.path_drop_rate > 0:
            if self.training:
                # train
                p = random.uniform(0, 1)
                drop_flag = p < self.path_drop_rate
                if drop_flag:
                    batch_size = x.size()[0]
                    feature_map_size = x.size()[2:4]
                    stride = edge.__dict__.get('stride', 1)
                    out_channels = self.out_dim_list[branch_idx]
                    padding = torch.zeros(batch_size, out_channels,
                                          feature_map_size[0] // stride, feature_map_size[1] // stride, device=x.device)
                    path_out = padding
                else:
                    path_out = self.path_normal_forward(x, edge, child, branch_bn)
            else:
                # test
                path_out = self.path_normal_forward(x, edge, child, branch_bn)
                path_out = path_out * (1 - self.path_drop_rate)
        else:
            path_out = self.path_normal_forward(x, edge, child, branch_bn)
        return path_out

    """ required methods """

    def forward(self, x):
        child_inputs = self.allocation_scheme(x)

        child_outputs = []
        for branch_idx in range(self.child_num):
            path_out = self.path_drop_forward(child_inputs[branch_idx], branch_idx)
            child_outputs.append(path_out)

        output = self.merge_scheme(child_outputs)
        return output

    @property
    def unit_str(self):
        if self.child_num > 0:
            children_str = []
            for _i, child in enumerate(self.child_nodes):
                child_str = None if child is None else child.unit_str
                children_str.append('%s=>%s' % (self.edges[_i].unit_str, child_str))
            children_str = '[%s]' % ', '.join(children_str)
        else:
            children_str = None
        return 'T(%s-%s, %s)' % (self.merge_type, self.split_type, children_str)

    @property
    def config(self):
        return {
            'name': TreeNode.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'split_type': self.split_type,
            'merge_type': self.merge_type,
            'has_branch_bn': self.has_branch_bn,
            'path_drop_rate': self.path_drop_rate,
            'edges': [
                edge.config if edge is not None else None for edge in self.edges
            ],
            'child_nodes': [
                child.config if child is not None else None for child in self.child_nodes
            ],
        }

    @staticmethod
    def build_from_config(config):
        if 'name' in config:
            config.pop('name')
        edges = []
        for edge_config in config.pop('edges'):
            edges.append(set_layer_from_config(edge_config))
        child_nodes = []
        for child_config in config.pop('child_nodes'):
            child_nodes.append(TreeNode.build_from_config(child_config) if child_config is not None else None)
        return TreeNode(edges=edges, child_nodes=child_nodes, **config)

    def get_flops(self, x):
        child_inputs = self.allocation_scheme(x)

        child_outputs = []
        flops = 0
        for branch_idx in range(self.child_num):
            edge, child, branch_x = self.edges[branch_idx], self.child_nodes[branch_idx], child_inputs[branch_idx]
            if edge is not None:
                edge_flop, branch_x = edge.get_flops(branch_x)
                flops += edge_flop
            if child is not None:
                child_flop, branch_x = child.get_flops(branch_x)
                flops += child_flop
            child_outputs.append(branch_x)
        output = self.merge_scheme(child_outputs)
        return flops, output


class TransitionBlock(BasicUnit):
    def __init__(self, layers):
        super(TransitionBlock, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def unit_str(self):
        return 'TransitionBlock'

    @property
    def config(self):
        return {
            'name': TransitionBlock.__name__,
            'layers': [
                layer.config for layer in self.layers
            ]
        }

    @staticmethod
    def build_from_config(config):
        layers = []
        for layer_config in config.get('layers'):
            layer = set_layer_from_config(layer_config)
            layers.append(layer)
        block = TransitionBlock(layers)
        return block

    def get_flops(self, x):
        flop = 0
        for layer in self.layers:
            delta_flop, x = layer.get_flops(x)
            flop += delta_flop
        return flop, x


class ResidualTreeBlock(BasicUnit):
    def __init__(self, cell, in_bottle, out_bottle, shortcut, final_bn=True, cell_drop_rate=0):
        super(ResidualTreeBlock, self).__init__()

        self.in_bottle = in_bottle
        self.out_bottle = out_bottle
        self.shortcut = shortcut
        self.cell = cell

        if final_bn:
            self.final_bn = nn.BatchNorm2d(self.out_channels)
        else:
            self.final_bn = None

        self.cell_drop_rate = cell_drop_rate

    @property
    def out_channels(self):
        if self.out_bottle is None:
            out_channels = self.cell.out_channels
        else:
            out_channels = self.out_bottle.out_channels
        return out_channels

    def cell_normal_forward(self, x):
        if self.in_bottle is not None:
            x = self.in_bottle(x)

        x = self.cell(x)

        if self.out_bottle is not None:
            x = self.out_bottle(x)
        if self.final_bn:
            x = self.final_bn(x)
        return x

    def forward(self, x):
        _x = self.shortcut(x)
        batch_size = _x.size()[0]
        feature_map = _x.size()[2:4]

        if self.cell_drop_rate > 0:
            if self.training:
                # train
                p = random.uniform(0, 1)
                drop_flag = p < self.cell_drop_rate
                if drop_flag:
                    x = torch.zeros(batch_size, self.out_channels, feature_map[0], feature_map[1], x.device)
                else:
                    x = self.cell_normal_forward(x)
            else:
                # test
                x = self.cell_normal_forward(x) * (1 - self.cell_drop_rate)
        else:
            x = self.cell_normal_forward(x)

        residual_channel = x.size()[1]
        shortcut_channel = _x.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel, feature_map[0], feature_map[1],
                                  device=x.device)
            _x = torch.cat((_x, padding), 1)

        return _x + x

    @property
    def unit_str(self):
        return 'ResidualTreeBlock'

    @property
    def config(self):
        return {
            'name': ResidualTreeBlock.__name__,
            'cell_drop_rate': self.cell_drop_rate,
            'final_bn': self.final_bn is not None,
            'shortcut': self.shortcut.config,
            'in_bottle': self.in_bottle.config,
            'out_bottle': self.out_bottle.config,
            'cell': self.cell.config,
        }

    @staticmethod
    def build_from_config(config):
        in_bottle = set_layer_from_config(config.get('in_bottle'))
        out_bottle = set_layer_from_config(config.get('out_bottle'))

        shortcut = set_layer_from_config(config.get('shortcut'))
        cell = TreeNode.build_from_config(config.get('cell'))
        final_bn = config.get('final_bn')
        cell_drop_rate = config.get('cell_drop_rate')

        return ResidualTreeBlock(cell, in_bottle, out_bottle, shortcut, final_bn, cell_drop_rate)

    def get_flops(self, x):
        flop, _x = self.shortcut.get_flops(x)
        batch_size = _x.size()[0]
        feature_map = _x.size()[2:4]

        if self.in_bottle is not None:
            delta_flop, x = self.in_bottle.get_flops(x)
            flop += delta_flop

        delta_flop, x = self.cell.get_flops(x)
        flop += delta_flop

        if self.out_bottle is not None:
            delta_flop, x = self.out_bottle.get_flops(x)
            flop += delta_flop

        residual_channel = x.size()[1]
        shortcut_channel = _x.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.zeros(batch_size, residual_channel - shortcut_channel, feature_map[0], feature_map[1],
                                  device=x.device)
            _x = torch.cat((_x, padding), 1)

        return flop, _x + x


class PyramidTreeNet(BasicUnit):
    def __init__(self, blocks, classifier,
                 tree_node_config=None, cell_drop_rate=0, cell_drop_scheme='linear'):
        super(PyramidTreeNet, self).__init__()

        self.blocks = nn.ModuleList(blocks)
        self.classifier = classifier

        self.tree_node_config = tree_node_config if tree_node_config is not None else {}
        self.cell_drop_rate = cell_drop_rate
        self.cell_drop_scheme = cell_drop_scheme

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def unit_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': PyramidTreeNet.__name__,
            'tree_node_config': self.tree_node_config,
            'cell_drop_rate': self.cell_drop_rate,
            'cell_drop_scheme': self.cell_drop_scheme,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        if 'name' in config:
            config.pop('name')
        blocks = []
        total_blocks = 0
        for block_config in config.pop('blocks'):
            if 'cell' in block_config:
                total_blocks += 1
                tree_node_config = copy.deepcopy(config.get('tree_node_config', {}))
                root_node = block_config['cell']['child_nodes'][0]
                if root_node is not None:
                    root_node.update(tree_node_config)
                    tree_node_config['has_branch_bn'] = False
                    to_updates = Queue()
                    for child_config in root_node['child_nodes']:
                        to_updates.put(child_config)
                    while not to_updates.empty():
                        child_config = to_updates.get()
                        if child_config is not None:
                            child_config.update(tree_node_config)
                            for new_config in child_config['child_nodes']:
                                to_updates.put(new_config)
            block = set_block_from_config(block_config)
            blocks.append(block)

        _l = 0
        for block in blocks:
            if 'cell_drop_rate' in block.__dict__:
                _l += 1
                if config.get('cell_drop_scheme', 'linear') == 'linear':
                    block.cell_drop_rate = 2 * _l * config.get('cell_drop_rate', 0) / (total_blocks + 1)
                else:
                    block.cell_drop_rate = config.get('cell_drop_rate', 0)

        classifier = set_layer_from_config(config.pop('classifier'))
        return PyramidTreeNet(blocks, classifier, **config)

    def get_flops(self, x):
        flop = 0
        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        return flop + delta_flop, x
