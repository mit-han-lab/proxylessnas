import os
import os.path as osp
import sys
import argparse
import time
import logging

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets as datasets

from proxyless_nas import utils as utils
from proxyless_nas import model_zoo
from proxyless_nas import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14
from proxyless_nas.utils import AverageMeter, accuracy


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
    parser.add_argument("-b", "--batch-size", type=int, default=512,
                        help="The batch on every device for training and validating")
    parser.add_argument('--cutout', action='store_true', default=False, 
                        help='use cutout')
    parser.add_argument("-j", "--workers", type=int, default=4,
                        help="The batch on every device for validation")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='proxyless_mobile_14', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: proxyless_mobile_14)')
    parser.add_argument('--epochs', type=int, default=500, 
                        help='num of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.025, 
                        help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, 
                        help='weight decay')
    parser.add_argument('--save', type=str, default='EXP', 
                        help='experiment name')
    parser.add_argument('--report_freq', type=float, default=50, 
                        help='report frequency')
    parser.add_argument('--init_channels', type=int, default=36, 
                        help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, 
                        help='total number of layers')
    parser.add_argument('--model_path', type=str, default='pretrained/weights.pt', 
                        help='path of pretrained model')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, 
                        help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, 
                        help='gradient clipping')
    parser.add_argument('--manual_seed', default=0, type=int)
    
    return parser.parse_args()


def train(train_queue, net, criterion, optimizer):
    """
    Training
    """
    logger.info("Start training")
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.train()

    for step, (_input, target) in enumerate(train_queue):
        if torch.cuda.is_available():
            target = target.cuda(async=True)
            _input = _input.cuda(async=True)
        input_var = Variable(_input, requires_grad=False)
        target_var = Variable(target, requires_grad=False)

        optimizer.zero_grad()

        output = net(input_var)

        loss = criterion(output, target_var)
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        n = _input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1[0], n)
        top5.update(prec5[0], n)

        if step % args.report_freq == 0:
            logger.info('train: step %03d, objs.avg %e, top1.avg %f, top5.avg %f', 
                        step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def valid(valid_queue, net, criterion):
    """
    Validating
    """
    logger.info("Start validating")
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    net.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits = net(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            logger.info('valid: step %03d, objs.avg %e, top1.avg %f, top5.avg %f', 
                        step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def main():
    """
    Main Function
    """
    global args, logger
    args = get_arguments()

    logger = utils.setup_logger("ProxylessNAS", "output/", 0)
    logger = logging.getLogger("ProxylessNAS.train_cifar10")

    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)
    
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.info('gpu device = %s', args.gpu)
    logger.info("args = %s", args)

    # linear scale the devices
    args.batch_size = args.batch_size * max(len(device_list), 1)
    args.workers = args.workers * max(len(device_list), 1)

    args.save = 'output/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    
    net = proxyless_gpu(pretrained=True)
    net = torch.nn.DataParallel(net).cuda()
    if os.path.exists(args.model_path):
        utils.load(net, args.model_path)
    cudnn.benchmark = True
    
    logger.info("param size = %fMB", utils.count_parameters_in_MB(net))
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = datasets.CIFAR10(root=args.path, train=True, download=True, transform=train_transform)
    valid_data = datasets.CIFAR10(root=args.path, train=False, download=True, transform=valid_transform)
    
    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    
    for epoch in range(args.epochs):
        scheduler.step()
        logger.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        net.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, _ = train(train_queue, net, criterion, optimizer)
        logger.info('train_acc %f', train_acc)

        valid_acc, _ = valid(valid_queue, net, criterion)
        logger.info('valid_acc %f', valid_acc)

        utils.save(net, os.path.join(args.save, 'weights.pt'))
    
    logger.info("Done")


if __name__ == "__main__":
    main()
