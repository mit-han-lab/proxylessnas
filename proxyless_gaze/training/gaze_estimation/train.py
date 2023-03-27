# must import pytorch_lightning before numpy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import argparse
import os
import os.path as osp
from pprint import pprint

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from dataloader import MPIIGazeDataset, XGazeDataset


def euler_to_vec(theta, phi):
    x = -1 * np.cos(theta) * np.sin(phi)
    y = -1 * np.sin(theta)
    z = -1 * np.cos(theta) * np.cos(phi)
    vec = np.array([x, y, z])
    vec = vec / np.linalg.norm(vec)
    return vec
    

class TrainModel(pl.LightningModule):
    def __init__(self, args):
        super(TrainModel, self).__init__()
        if args.model == "MyModelv7":
            self.model = models.MyModelv7(arch=args.arch)
        elif args.model == 'MyModelv8':
            self.model = models.MyModelv8(arch=args.arch)
        else:
            raise NotImplementedError
        self.criterion = getattr(torch.nn, args.criterion)()
        self.args = args
    
    def calc_angle_error(self, preds, gts):
        # in degree
        preds = np.deg2rad(preds.detach().cpu())
        gts = np.deg2rad(gts.detach().cpu())
        errors = []
        for pred, gt in zip(preds, gts):
            pred_vec = euler_to_vec(pred[0], pred[1])
            gt_vec = euler_to_vec(gt[0], gt[1])
            error = np.rad2deg(np.arccos(np.clip(np.dot(pred_vec, gt_vec), -1.0, 1.0)))
            errors.append(error)
        return errors
    
    def forward(self, left_eye, right_eye, face):
        return self.model(left_eye, right_eye, face)

    def training_step(self, batch, batch_idx):
        left_eye, right_eye, face, label = batch
        output = self.model(left_eye, right_eye, face)
        loss = self.criterion(output, label)
        if batch_idx % 100 == 0:
            angle_error = np.mean(self.calc_angle_error(output, label))
            self.log("train_angle_error", angle_error, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        left_eye, right_eye, face, label = batch
        output = self.model(left_eye, right_eye, face)
        loss = self.criterion(output, label)
        angle_error = np.mean(self.calc_angle_error(output, label))
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_angle_error", angle_error, on_epoch=True, logger=True)
    
    def validation_epoch_end(self, outputs):
        pass
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.args.optimizer)(self.model.parameters(), **self.args.optimizer_parameters)
        return optimizer
    
    def train_dataloader(self):
        trainset = XGazeDataset(self.args.dataset_dir, is_train=True)
        trainloader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        return trainloader

    def val_dataloader(self):
        valset = XGazeDataset(self.args.dataset_dir, is_train=False)
        valloader = DataLoader(valset, batch_size=4*self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        return valloader
    
    def test_dataloader(self):
        testset = XGazeDataset(self.args.dataset_dir, is_train=False)
        testloader = DataLoader(testset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        return testloader

pl.seed_everything(47)
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=False, type=str, default="./configs/config.yaml")
parser.add_argument('--resume', action="store_true", dest="resume")
args = parser.parse_args()

with open(args.config) as f:
    yaml_args = yaml.load(f, Loader=yaml.FullLoader)
yaml_args.update(vars(args))
args = argparse.Namespace(**yaml_args)

model = TrainModel(args)

if args.logger == 'wandb':
    mylogger = WandbLogger(project=args.project, 
                           log_model=False, 
                           name=args.run_name,
                           id=args.run_name)
    mylogger.log_hyperparams(args)
    mylogger.watch(model, None, 10000, log_graph=False)
else:
    raise NotImplementedError

checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_dir,
                                      filename='{epoch}-{val_loss:.4f}-{val_angle_error:.2f}',
                                      monitor='val_angle_error',
                                      save_last=True,
                                      save_top_k=3,
                                      verbose=False)

trainer = Trainer(default_root_dir=args.ckpt_dir,
                  gpus=-1,
                  precision=32,
                  callbacks=[checkpoint_callback],
                  max_epochs=args.epoch,
                  benchmark=True,
                  strategy=DDPStrategy(find_unused_parameters=False),
                  logger=mylogger
                  )

if args.resume:
    trainer.fit(model, ckpt_path=osp.join(args.ckpt_dir, "last.ckpt"))
else:
    trainer.fit(model)
trainer.validate(model)
