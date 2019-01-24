import os
import os.path as osp
import sys
import argparse
import time

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import datasets as datasets

import proxyless_nas.utils as utils
from proxyless_nas.utils import AverageMeter, accuracy

from proxyless_nas import model_zoo

model_names = sorted(name for name in model_zoo.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(model_zoo.__dict__[name]))

def get_arguments():
    """
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ProxyLessNAS")

    parser.add_argument("-p", '--path', type=str, default="datasets/",
                        help='The path of cifar10')
    parser.add_argument("-g", '--gpu', type=str, default='all',
                        help='The gpu(s) to use')
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="The batch on every device for validation")
    parser.add_argument('--cutout', action='store_true', default=False, 
                        help='use cutout')
    parser.add_argument("-j", "--workers", type=int, default=4,
                        help="The batch on every device for validation")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='proxyless_mobile_14', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: proxyless_mobile_14)')
    parser.add_argument('--manual_seed', default=0, type=int)
    
    return parser.parse_args()


def eval(test_queue, net, criterion):
    """
    
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()

    end = time.time()
    for i, (_input, target) in enumerate(test_queue):
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

        if i % 10 == 0 or i + 1 == len(test_queue):
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'top 1-acc {top1.val:.3f} ({top1.avg:.3f})\t'
                'top 5-acc {top5.val:.3f} ({top5.avg:.3f})'.
                format(i, len(test_queue), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
    print(("Average Loss: {losses:.3f}, Top1 Acc: {top1:.3f}, Top5 Acc: {top5:.3f}").
            format(losses=losses.avg, top1=top1.avg, top5=top5.avg))


if __name__ == "__main__":
    """
    Main Function
    """
    global args
    args = get_arguments()

    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # linear scale the devices
    args.batch_size = args.batch_size * max(len(device_list), 1)
    args.workers = args.workers * max(len(device_list), 1)
    
    net = model_zoo.__dict__[args.arch](pretrained=True)
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    _, test_transform = utils._data_transforms_cifar10(args)
    test_data = datasets.CIFAR10(root=args.path, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    print("Start testing")
    eval(test_queue, net, criterion)
    print("Done")