import os.path as osp
import numpy as np

import argparse

import torch.utils.data
from torchvision import transforms, datasets

from proxyless_nas.utils import AverageMeter

from proxyless_nas_tensorflow import tf_model_zoo

model_names = sorted(name for name in tf_model_zoo.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(tf_model_zoo.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    '--path',
    help='The path of imagenet',
    type=str,
    default="/ssd/dataset/imagenet")
parser.add_argument(
    "-b",
    "--batch-size",
    help="The batch on every device for validation",
    type=int,
    default=64)
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

net = tf_model_zoo.__dict__[args.arch](pretrained=True)

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

losses = AverageMeter()
top1 = AverageMeter()
for i, (_input, target) in enumerate(data_loader):
    images = _input.numpy()
    images = np.transpose(images, axes=[0, 2, 3, 1])
    labels = net.labels_to_one_hot(1000, target.numpy())

    feed_dict = {
        net.images: images,
        net.labels: labels,
        net.is_training: False,
    }
    fetches = [net.cross_entropy, net.accuracy]
    loss, accuracy = net.sess.run(fetches, feed_dict=feed_dict)

    losses.update(loss, images.shape[0])
    top1.update(accuracy * 100, images.shape[0])
    if i % 50 == 0:
        print(i, '\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses),
              '\tTop 1-acc {top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1))
print('Loss: %.4f' % losses.avg, '\tTop-1: %.3f' % top1.avg)
