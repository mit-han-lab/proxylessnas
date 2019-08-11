from functools import partial
import json

import torch

from .utils import download_url
from .nas_modules import ProxylessNASNets


def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])

    return net


proxyless_cpu = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth")

proxyless_gpu = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth")

proxyless_mobile = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth")

proxyless_mobile_14 = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.pth")
