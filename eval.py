import os
import os.path as osp

import argparse
import time
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from proxyless_nas.utils import AverageMeter, accuracy

from proxyless_nas import model_zoo

model_names = sorted(name for name in model_zoo.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(model_zoo.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    '--path',
    help='The path of imagenet',
    type=str,
    default="/dataset/imagenet")
parser.add_argument(
    "-g",
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=32)
parser.add_argument(
    "-j",
    "--workers",
    help="Number of workers of data loader",
    type=int,
    default=4)
parser.add_argument(
    '-a',
    '--arch',
    metavar='ARCH',
    default='proxyless_mobile_14',
    choices=model_names,
    help='model architecture: ' +
    ' | '.join(model_names) +
    ' (default: proxyless_mobile_14)')
parser.add_argument(
    '-d',
    '--dataset',
    default='imagenet',
    choices=['cifar10', 'imagenet'],
    help='Dataset'
)
parser.add_argument('--manual_seed', default=0, type=int)
args = parser.parse_args()

if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.dataset == 'imagenet':
    net = model_zoo.__dict__[args.arch](pretrained=True)
else:
    net = model_zoo.__dict__['proxyless_cifar'](pretrained=True)

# linear scale the devices
args.batch_size = args.batch_size * max(len(device_list), 1)
args.workers = args.workers * max(len(device_list), 1)

if args.dataset == 'imagenet':
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            osp.join(
                args.path,
                "val"),
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[
                            0.485,
                            0.456,
                            0.406],
                        std=[
                            0.229,
                            0.224,
                            0.225]),
                ])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
else:
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        ),
    ])
    data_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('.dataset/cifar10', train=False, transform=test_transforms, download=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

device = torch.device('cuda:0')

net = torch.nn.DataParallel(net).to(device)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().to(device)

net.eval()

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

with torch.no_grad():
    with tqdm(total=len(data_loader), desc='Test') as t:
        for i, (_input, target) in enumerate(data_loader):
            target = target.to(device)
            _input = _input.to(device)

            # compute output
            output = net(_input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), _input.size(0))
            top1.update(acc1[0].item(), _input.size(0))
            top5.update(acc5[0].item(), _input.size(0))

            t.set_postfix({
                'Loss': losses.avg,
                'Top1': top1.avg,
                'Top5': top5.avg
            })
            t.update(1)

print('Loss:', losses.avg, '\t Top1:', top1.avg, '\t Top5:', top5.avg)
