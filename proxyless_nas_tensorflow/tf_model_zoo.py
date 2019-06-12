from functools import partial
import json
import pickle

from proxyless_nas.utils import download_url
from .tf_modules import ProxylessNASNets


def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = pickle.load(open(init_path, 'rb'))
    else:
        init = None
    net = ProxylessNASNets(net_config_json, init)

    return net


proxyless_cpu = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.tfinit")

proxyless_gpu = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.tfinit")

proxyless_mobile = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.tfinit")

proxyless_mobile_14 = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.tfinit")
