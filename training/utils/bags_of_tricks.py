import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
#     n_classes = pred.size(1)
#     # convert to one-hot
#     target = torch.unsqueeze(target, 1)
#     soft_target = torch.zeros_like(pred)
#     soft_target.scatter_(1, target, 1)
#     # label smoothing
#     soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes * 1
#     return torch.mean(torch.sum(- soft_target * F.log_softmax(pred, dim=-1), 1))

def flip_along_batch(input, step=-1):
    inv_idx = torch.arange(input.size(0) - 1, -1, step).long()
    return input[inv_idx]


def label_smoothing(pred, target, eta=0.1):
    '''
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    '''
    n_classes = pred.size(1)
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_classes * 1


def cross_encropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    return cross_entropy_for_onehot(pred, onehot_target)


# MIXUP
def mixup_data(inputs, lam=1):
    flipped_inputs = inputs[::-1]  # flip over batch dimensions
    return lam * inputs + (1 - lam) * flipped_inputs


def mixup_label(pred, target, lam=1, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    flipped_target = onehot_target[::-1]  # flip over batch dimensions
    return lam * onehot_target + (1 - lam) * flipped_target


def cross_encropy_with_mixup(pred, target, lam=1, eta=0.0):
    mixup_target = mixup_label(pred, target, lam=lam, eta=eta)
    return cross_entropy_for_onehot(pred, mixup_target)


if __name__ == '__main__':
    import functools

    batch_size = 15
    num_classes = 100
    c1 = nn.CrossEntropyLoss()
    c2 = functools.partial(cross_encropy_with_label_smoothing, eta=0)

    for i in range(100):
        pred = torch.randn(batch_size, num_classes)
        target = torch.randn(batch_size, ).long().random_(num_classes)
        o1 = c1(pred, target)
        o2 = c2(pred, target)
        print(o1, o2, o2 - o1)
