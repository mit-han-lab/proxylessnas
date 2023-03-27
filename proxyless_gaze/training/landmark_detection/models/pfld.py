#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import math

from torch.nn.modules.container import Sequential
from torch.nn.modules.pooling import AdaptiveAvgPool2d

def _make_backbone():
    from tinynas.nn.networks import ProxylessNASNets
    import json
    json_file = "pretrained/proxyless-w0.3-r176_imagenet.json"
    with open(json_file) as f:
        config = json.load(f)
    _model = ProxylessNASNets.build_from_config(config)
    ckpt = torch.load("pretrained/proxyless-w0.3-r176_imagenet.pth")
    _model.load_state_dict(ckpt['state_dict'], strict=False)
    return _model


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

from torchvision.models.mobilenetv2 import mobilenet_v2

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class PFLDInference(nn.Module):
    
    def x1_hook(self, module, input, output):
        self.x1 = output
    
    def x2_hook(self, module, input, output):
        self.x2 = output
    
    def out1_hook(self, module, input, output):
        self.out1 = output
        
    def __init__(self, landmark):
        super(PFLDInference, self).__init__()
        self.backbone = _make_backbone()
        # self.backbone.blocks[8].register_forward_hook(self.out1_hook)
        # self.backbone.blocks[16].register_forward_hook(self.x1_hook)
        # self.backbone.blocks[-1].register_forward_hook(self.x2_hook)
        
        self.backbone.blocks[4].register_forward_hook(self.out1_hook)
        self.backbone.blocks[8].register_forward_hook(self.x1_hook)
        self.backbone.blocks[16].register_forward_hook(self.x2_hook)

        self.avg_pool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(432, landmark*2)

    def forward(self, x):  # x: 3, 112, 112
        x3 = self.backbone(x)
        x1 = self.avg_pool1(self.x1)
        x2 = self.avg_pool2(self.x2)
        x3 = self.avg_pool3(x3)
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        landmarks = torch.cat([x1, x2, x3], 1)
        
        landmarks = self.fc(landmarks)
        self.x1 = None
        self.x2 = None
        return self.out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(16, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class PFLDInferenceOriginal(nn.Module):
    def __init__(self, L):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class AuxiliaryNetOriginal(nn.Module):
    def __init__(self):
        super(AuxiliaryNetOriginal, self).__init__()
        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def pretty_print(x):
    if x is None:
        return ""
    if x < 1000:
        return str(round(x, 3))
    elif x < 1000000:
        return str(round(x/1000, 3))+"K"
    elif x < 1000000000:
        return str(round(x/1000000, 3))+"M"
    else:
        return str(round(x/1000000000, 3))+"G"
    
if __name__ == '__main__':
    import sys
    sys.path.append("./")
    import numpy as np
    import matplotlib.pyplot as plt
    
    input = torch.randn(1, 3, 112, 112)
    pfld_backbone = PFLDInference(98)
    auxiliarynet = AuxiliaryNet()
    features, landmarks = pfld_backbone(input)
    angle = auxiliarynet(features)

    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))

    from thop import profile
    from torchprofile import profile_macs
    # _, param = profile(pfld_backbone, inputs=(input, ))
    macs = profile_macs(pfld_backbone, args=(input, ))
    print(pretty_print(macs))

    # mac_list = []
    # for op in macs:
    #     # print(f'{op.operator:<30} {str(op.outputs[0].shape):<20} {pretty_print(macs[op]):<10} {pretty_print(np.prod(op.inputs[1].shape))}')
    #     if str(op.operator) == "aten::_convolution":
    #         mac_list.append(macs[op])
    # macs = profile_macs(pfld_backbone, args=(input, ))
    # plt.subplot(2,1,1)
    # plt.bar(list(range(len(mac_list))), mac_list)
    # plt.grid()
    # plt.ylim(0, 20*10**6)
    
    
    # plt.subplot(2,1,2)
    # pfld_backbone = PFLDInferenceOriginal()
    # # _, param = profile(pfld_backbone, inputs=(input, ))
    # macs = profile_macs(pfld_backbone, args=(input, ), reduction=None)
    # mac_list = []
    # for op in macs:
    #     # print(f'{op.operator:<30} {str(op.outputs[0].shape):<20} {pretty_print(macs[op]):<10} {pretty_print(np.prod(op.inputs[1].shape))}')
    #     if str(op.operator) == "aten::_convolution":
    #         mac_list.append(macs[op])
    # macs = profile_macs(pfld_backbone, args=(input, ))
    # plt.bar(list(range(len(mac_list))), mac_list)
    # plt.grid()
    # plt.ylim(0, 20*10**6)
    
    # plt.show()
    
    # import onnx
    # save_name = "pfld_proxyless_v2.onnx"
    # print(f"save ONNX model to {save_name}")
    # torch.onnx.export(pfld_backbone, input, save_name)
    # onnx.save(onnx.shape_inference.infer_shapes(onnx.load(save_name)), save_name)
    # print("ONNX model saved")
    
