#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import argparse
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference, AuxiliaryNet
from pfld.loss import PFLDLoss
from tqdm import tqdm
from test import compute_nme

pl.seed_everything(6)
convert_106_to_98_list = list(range(0,55))+[58,59,60,61,62]+list(range(66,74))+list(range(75,83))+list(range(84,104))+[104,105]
assert len(convert_106_to_98_list) == 98

def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

class TrainModel(pl.LightningModule):
    def __init__(self, args, logger):
        super(TrainModel, self).__init__()
        self.pfld_backbone = PFLDInference(98)
        self.auxiliarynet = AuxiliaryNet()
        self.criterion = PFLDLoss(98)
        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.learning_rate = args.base_lr
    
    def forward(self, img):
        landmarks = self.pfld_backbone(img)
        return landmarks

    def training_step(self, batch, batch_idx):
        img, landmark_gt, attribute_gt, euler_angle_gt = batch
        features, landmarks = self.pfld_backbone(img)
        angle = self.auxiliarynet(features)
        weighted_loss, loss, angle_loss = self.criterion(attribute_gt, landmark_gt,
                                        euler_angle_gt, angle, landmarks,
                                        args.train_batchsize)
        self.log("train_loss", loss.item())
        self.log("train_weighted_loss", weighted_loss.item())
        self.log("train_angle_loss", angle_loss.item())
        return weighted_loss
    
    def validation_step(self, batch, batch_idx):
        img, landmark_gt, _, _ = batch
        _, landmarks = self.pfld_backbone(img)
        # landmarks = landmarks.reshape(-1, 106, 2)[:,convert_106_to_98_list].reshape(landmarks.shape[0], -1)
        loss = torch.mean(torch.sum((landmark_gt - landmarks)**2, axis=1))
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
        landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
        nme = compute_nme(landmarks, landmark_gt)
        return {"val_nme": nme, "val_loss": loss.item()}
    
    def validation_epoch_end(self, outputs):
        nme = torch.hstack([torch.tensor(x['val_nme']).unsqueeze(0) for x in outputs]).reshape(-1)
        loss = torch.hstack([torch.tensor(x['val_loss']).unsqueeze(0) for x in outputs]).reshape(-1)
        self.log("val_nme", nme.mean().item())
        self.log("val_loss", loss.mean().item())
    
    def test_step(self, batch, batch_idx):
        img, landmark_gt, _, _ = batch
        _, landmark = self.pfld_backbone(img)
        loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
        return {"test_loss": loss}
    
    def test_epoch_end(self, outputs):
        losses = torch.tensor([x['test_loss'] for x in outputs])
        self.log("test_loss", losses.mean().item())
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{
            'params': self.pfld_backbone.parameters()
        }, {
            'params': self.auxiliarynet.parameters()
        }],
            lr=self.learning_rate,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.lr_patience, verbose=True, factor=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        return optim_dict
    
    def train_dataloader(self):
        trainset = WLFWDatasets(args.dataroot, 98, self.transform)
        trainloader = DataLoader(trainset, batch_size=args.train_batchsize, 
                                 shuffle=True, num_workers=args.workers, drop_last=False)
        return trainloader

    def val_dataloader(self):
        valset = WLFWDatasets(args.val_dataroot, 98, self.transform)
        valloader = DataLoader(valset, batch_size=args.val_batchsize,
                               shuffle=False, num_workers=args.workers)
        return valloader
    
    def test_dataloader(self):
        testset = WLFWDatasets(args.val_dataroot, 98, self.transform)
        testloader = DataLoader(testset, batch_size=args.val_batchsize,
                               shuffle=False, num_workers=args.workers)
        return testloader

def main(args):
    mylogger = WandbLogger(project=args.project,
                           log_model=False, name=args.run_name)
    mylogger.log_hyperparams(args)
    print_args(args)

    model = TrainModel(args, mylogger)
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir,
                                        filename='{epoch}-{val_loss:.4f}-{val_nme:.5f}',
                                        monitor='val_nme',
                                        save_last=True,
                                        save_top_k=3,
                                        verbose=False)
    trainer = Trainer(default_root_dir=args.ckpt_dir,
                        gpus=2,
                        num_nodes=args.num_nodes,
                        precision=32,
                        callbacks=[checkpoint_callback],
                        progress_bar_refresh_rate=1,
                        max_epochs=args.epoch,
                        benchmark=True,
                        strategy="ddp",
                        # strategy="ddp_find_unused_parameters_false",
                        logger=mylogger)
    trainer.fit(model)

def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('-p', '--project', type=str, default="PFLD")
    parser.add_argument('-r', '--run-name', type=str, default="exp")
    
    # training
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=float)
    parser.add_argument('--weight-decay', '--wd', default=1e-2, type=float)

    # -- lr
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- checkpoint
    parser.add_argument('--ckpt_dir',
                        default='./checkpoint',
                        type=str,
                        metavar='PATH')
    # --dataset
    parser.add_argument('--dataroot',
                        default='/dev/shm/train_list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--val_dataroot',
                        default='/dev/shm/WFLW_ready/test_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=128, type=int)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    
    args = parser.parse_args()
    args.base_lr = args.base_lr*2*args.num_nodes
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
