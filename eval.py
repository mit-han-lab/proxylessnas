import os
import os.path as osp

import argparse
import time

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
    default="/ssd/dataset/imagenet")
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
    help="The batch on every device for validation",
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
parser.add_argument('--manual_seed', default=0, type=int)
args = parser.parse_args()

if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

net = model_zoo.__dict__[args.arch](pretrained=True)

# linear scale the devices
args.batch_size = args.batch_size * max(len(device_list), 1)
args.workers = args.workers * max(len(device_list), 1)

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

net = torch.nn.DataParallel(net).cuda()
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()

net.eval()

batch_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

end = time.time()
for i, (_input, target) in enumerate(data_loader):
    if torch.cuda.is_available():
        target = target.cuda(async=True)
        _input = _input.cuda(async=True)
    input_var = torch.autograd.Variable(_input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = net(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.data[0], _input.size(0))
    top1.update(acc1[0], _input.size(0))
    top5.update(acc5[0], _input.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 10 == 0 or i + 1 == len(data_loader):
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'top 1-acc {top1.val:.3f} ({top1.avg:.3f})\t'
              'top 5-acc {top5.val:.3f} ({top5.avg:.3f})'. format(i,
                                                                  len(data_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses,
                                                                  top1=top1,
                                                                  top5=top5))

print(losses.avg, top1.avg, top5.avg)
