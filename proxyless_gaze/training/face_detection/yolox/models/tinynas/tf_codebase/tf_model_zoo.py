import json
import pickle

from .tf_modules import ProxylessNASNets


def proxyless_base(pretrained=True, net_config=None, net_weight=None, graph=None, sess=None, is_training=True,
                   images=None, img_size=None, only_train=True, latency=-1):
    if only_train:
        print('#' * 50, 'TRAIN {}ms'.format(latency), '#' * 50)
    else:
        print('#' * 50, 'ALL {}ms'.format(latency), '#' * 50)
    assert net_config is not None, "Please input a network config"
    prefix = 'train' if only_train else 'train+val'
    net_config_json = json.load(open('finetune/{}/{}ms/net.config'.format(prefix, latency), 'r'))

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init = pickle.load(open('finetune/{}/{}ms/1001.tfinit'.format(prefix, latency), 'rb'))
    else:
        init = None
    net = ProxylessNASNets(net_config_json, init, graph, sess, is_training, images, img_size)

    return net

