#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


# ==================== Results ====================
# Easy   Val AP: 0.6804611527355322
# Medium Val AP: 0.5184280410666231
# Hard   Val AP: 0.2325707899936543
# =================================================


import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (160, 128)
        self.multiscale_range = 0
        self.mosaic_scale = (0.8, 1.6)
        self.test_size = (160, 128)
        self.mosaic_prob = 0.5
        self.enable_mixup = True
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.max_epoch = 500
        self.no_aug_epochs = 50

        self.data_dir = "/dev/shm/widerface_coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.num_classes = 1
        self.eval_interval = 5

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels=[64, 96, 320]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True, backbone="MyBackbone"
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
