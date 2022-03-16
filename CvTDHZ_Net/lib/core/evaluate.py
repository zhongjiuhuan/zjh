from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import cv2


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


# --------------------------------------------
# PSNR
# --------------------------------------------
@torch.no_grad()
def calculate_psnr(output, target, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not output.shape == target.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = output.shape[:2]
    output = output[border:h - border, border:w - border]
    target = target[border:h - border, border:w - border]

    output = output.astype(np.float64)
    target = target.astype(np.float64)
    mse = np.mean((output - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
@torch.no_grad()
def calculate_ssim(output, target, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not output.shape == target.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = output.shape[:2]
    output = output[border:h - border, border:w - border]
    target = target[border:h - border, border:w - border]

    if output.ndim == 2:
        return ssim(output, target)
    elif output.ndim == 3:
        if output.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(output[:, :, i], target[:, :, i]))
            return np.array(ssims).mean()
        elif output.shape[2] == 1:
            return ssim(np.squeeze(output), np.squeeze(target))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(output, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    output = output.astype(np.float64)
    target = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(output, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(target, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(output ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(target ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(output * target, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
