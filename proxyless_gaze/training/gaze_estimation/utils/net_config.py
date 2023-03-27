# generate the detailed network configuration with input and output shape, which is easier for profile
import torch
import torch.nn as nn

__all__ = ['get_network_config_with_activation_shape']


def record_in_out_shape(m, x, y):
    x = x[0]
    m.input_shape = torch.Tensor(list(x.shape))
    m.output_shape = torch.Tensor(list(y.shape))


def record_residual_shape(m, x, y):
    from tinynas.nn.modules import ZeroLayer
    if m.mobile_inverted_conv is None or isinstance(m.mobile_inverted_conv, ZeroLayer):
        pass
    elif m.shortcut is None or isinstance(m.shortcut, ZeroLayer):
        pass
    else:  # only record the shape if we actually use the residual connection
        m.output_shape = torch.Tensor(list(y.shape))


def add_activation_shape_hook(m_):
    from tinynas.nn.networks import MobileInvertedResidualBlock
    m_type = type(m_)
    if m_type == nn.Conv2d:
        m_.register_buffer('input_shape', torch.zeros(4))
        m_.register_buffer('output_shape', torch.zeros(4))
        m_.register_forward_hook(record_in_out_shape)
    elif m_type == MobileInvertedResidualBlock:
        m_.register_buffer('output_shape', torch.zeros(4))
        m_.register_forward_hook(record_residual_shape)


def get_network_config_with_activation_shape(model, device='cpu', data_shape=(1, 3, 224, 224)):
    from tinynas.nn.networks import ProxylessNASNets
    assert isinstance(model, ProxylessNASNets)
    import copy
    model = copy.deepcopy(model).to(device)
    model.eval()
    model.apply(add_activation_shape_hook)

    with torch.no_grad():
        _ = model(torch.randn(*data_shape).to(device))

    def get_conv_cfg(conv):
        conv = conv.conv
        return {
            'in_channel': int(conv.input_shape[1]),
            'in_shape': int(conv.input_shape[2]),
            'out_channel': int(conv.output_shape[1]),
            'out_shape': int(conv.output_shape[2]),
            'kernel_size': conv.kernel_size[0],
            'stride': conv.stride[0],
            'groups': conv.groups,
            'depthwise': conv.groups == int(conv.input_shape[1]),
        }

    def get_linear_cfg(op):
        return {
            'in_channel': op.in_features,
            'out_channel': op.out_features,
        }

    def get_block_cfg(block):
        # (inverted_bottleneck), depth_conv, point_linear
        pdp = block.mobile_inverted_conv
        if int(block.output_shape[0]) == 0:
            residual = None
        else:
            assert block.output_shape[2] == block.output_shape[3]
            residual = {'in_channel': int(block.output_shape[1]), 'in_shape': int(block.output_shape[2])}

        return {
            'pointwise1': get_conv_cfg(pdp.inverted_bottleneck) if pdp.inverted_bottleneck is not None else None,
            'depthwise': get_conv_cfg(pdp.depth_conv),
            'pointwise2': get_conv_cfg(pdp.point_linear),
            'residual': residual
        }

    cfg = {}
    cfg['first_conv'] = get_conv_cfg(model.first_conv)
    cfg['classifier'] = get_linear_cfg(model.classifier)
    # assert input_model.feature_mix_layer is None
    if model.feature_mix_layer is not None:
        cfg['feature_mix'] = get_conv_cfg(model.feature_mix_layer)
    else:
        cfg['feature_mix'] = None

    block_cfg = []

    # now check input_model.blocks
    for block in model.blocks:
        from tinynas.nn.modules import ZeroLayer
        if block.mobile_inverted_conv is None or isinstance(block.mobile_inverted_conv, ZeroLayer):  # empty block
            continue
        block_cfg.append(get_block_cfg(block))

    cfg['blocks'] = block_cfg

    del model

    return cfg
