import torch as th
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' \
        else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class MS_SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.84, data_range=1):
        super(MS_SSIM_L1_Loss, self).__init__()
        self.alpha = alpha
        self.L1 = nn.L1Loss()
        self.MS_SSIM = MS_SSIM(data_range=data_range, size_average=True, channel=3)

    def forward(self, x, target):
        loss = self.alpha * self.MS_SSIM(x, target) + (1 - self.alpha) * self.L1(x, target)
        return loss


def build_criterion(config, train=True):
    if config.AUG.MIXUP_PROB > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = SoftTargetCrossEntropy() \
            if train else nn.CrossEntropyLoss()
    elif config.LOSS.LABEL_SMOOTHING > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = LabelSmoothingCrossEntropy(config.LOSS.LABEL_SMOOTHING)
    elif config.LOSS.LOSS == 'softmax':
        criterion = nn.CrossEntropyLoss()
    elif config.LOSS.LOSS == 'MS_SSIM_L1':
        criterion = MS_SSIM_L1_Loss(config.LOSS.ALPHA, config.LOSS.DATA_RANGE)
    else:
        raise ValueError('Unkown loss {}'.format(config.LOSS.LOSS))

    return criterion
