from __future__ import print_function

import os, os.path as osp
import math
import argparse

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd
import tensorboardX
from tqdm import tqdm

import net224x224 as models
from utils.bags_of_tricks import cross_encropy_with_label_smoothing

import subprocess
subprocess.call("ulimit -n 65536", shell=True)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='The number of classes in the dataset.')

parser.add_argument('--train-dir', default=os.path.expanduser('/ssd/dataset/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/ssd/dataset/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=64,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# tricks to boost accuracy
parser.add_argument('--lr-scheduler', type=str, default="cosine", choices=["linear", "cosine"],
                    help='how to schedule learning rate')
parser.add_argument("--color-jitter", action='store_true', default=False,
                    help="To apply color augmentation or not.")
parser.add_argument("--label-smoothing", action='store_true', default=False,
                    help="To use label smoothing or not.")
parser.add_argument("--no-wd-bn", action='store_true', default=False,
                    help="Whether to remove the weight decay on BN")

args = parser.parse_args()
name_componenets = [args.arch, str(args.epochs), args.lr_scheduler]
if args.color_jitter:
    name_componenets.append("color_jitter")
if args.label_smoothing:
    name_componenets.append("label_smoothing")

args.log_dir = osp.join(args.log_dir, "-".join(name_componenets))
args.checkpoint_format = osp.join(args.log_dir, args.format)
# linearly scale the learning rate.
args.base_lr = args.base_lr * (args.batch_size / 64)

args.cuda = not args.no_cuda and torch.cuda.is_available()

hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
best_val_acc = 0.0

kwargs = {'num_workers': 5, 'pin_memory': True} if args.cuda else {}
# Training transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
pre_process = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
]
if args.color_jitter:
    pre_process += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]
pre_process += [
    transforms.ToTensor(),
    normalize
]

train_dataset = datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose(pre_process))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

val_dataset = datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             normalize
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


# Set up standard ResNet-50 model.
# model = models.resnet50()
model = models.__dict__[args.arch](num_classes=args.num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.base_lr * hvd.size(),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.label_smoothing:
    criterion = cross_encropy_with_label_smoothing
else:
    criterion = nn.CrossEntropyLoss()


# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            lr_cur = adjust_learning_rate(epoch, batch_idx, type=args.lr_scheduler)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss)
            train_accuracy.update(accuracy(output, target))
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item(),
                           'lr': lr_cur})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch, ):
    global best_val_acc
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(criterion(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
        best_val_acc = max(best_val_acc, val_accuracy.avg)
        log_writer.add_scalar('val/best_acc', best_val_acc, epoch)


    return val_accuracy.avg

import torch.optim.lr_scheduler as lr_scheduler
# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, type="cosine"):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif type == "linear":
        if epoch < 30:
            lr_adj = 1.
        elif epoch < 60:
            lr_adj = 1e-1
        elif epoch < 90:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
    elif type == "cosine":
        # self.init_lr * 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
        run_epochs = epoch - args.warmup_epochs
        total_epochs = args.epochs - args.warmup_epochs
        T_cur = float(run_epochs * len(train_loader)) + batch_idx
        T_total = float(total_epochs * len(train_loader))

        lr_adj = 0.5  * (1 + math.cos(math.pi * T_cur / T_total))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * lr_adj
    return args.base_lr * hvd.size() * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        os.remove(args.checkpoint_format.format(epoch=epoch))
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


best_acc = 0.0
last_saved_epoch = None
for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    val_acc = validate(epoch)

    # save checkpoint for the master
    if hvd.rank() == 0:
        if last_saved_epoch is not None:
            os.remove(args.checkpoint_format.format(epoch=last_saved_epoch))
        filepath = args.checkpoint_format.format(epoch=epoch)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)
        last_saved_epoch = epoch
