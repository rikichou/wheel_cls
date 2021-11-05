# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : batch_operators.py
#@time   : 2020-12-24 17:33:50
"""

import torch
import numpy as np
from .fmix import sample_mask


class MixUp(object):
    """ Mixup operator """

    def __init__(self, alpha=0.2):
        assert alpha > 0., 'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha

    def __call__(self, batched_inputs):
        images = batched_inputs["images"]
        targets = batched_inputs["targets"]

        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(images.size(0), device=images.device, dtype=torch.long)
        batched_inputs["images"] = lam * images + (1 - lam) * images[index]
        batched_inputs["targets_a"], batched_inputs["targets_b"] = targets, targets[index]
        batched_inputs["lam"] = lam
        return batched_inputs


class Cutmix(object):
    """ Cutmix operator """

    def __init__(self, alpha=0.2):
        assert alpha > 0., \
                'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha

    def _rand_bbox(self, size, lam):
        """ _rand_bbox """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batched_inputs):
        images = batched_inputs["images"]
        targets = batched_inputs["targets"]
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(images.size(0), device=images.device, dtype=torch.long)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        batched_inputs["images"] = images
        batched_inputs["targets_a"], batched_inputs["targets_b"] = targets, targets[index]
        batched_inputs["lam"] = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        return batched_inputs


class Fmix(object):
    """ Fmix operator """

    def __init__(self, alpha=1, decay_power=3, max_soft=0., reformulate=False):
        self.alpha = alpha
        self.decay_power = decay_power
        self.max_soft = max_soft
        self.reformulate = reformulate

    def __call__(self, batched_inputs):
        images = batched_inputs["images"]
        targets = batched_inputs["targets"]
        
        index = torch.randperm(images.size(0), device=images.device, dtype=torch.long)
        size = (images.size(2), images.size(3))
        lam, mask = sample_mask(self.alpha, self.decay_power, size, self.max_soft, self.reformulate)
        batched_inputs["images"] = mask * images + (1 - mask) * images[index]
        batched_inputs["targets_a"], batched_inputs["targets_b"] = targets, targets[index]
        batched_inputs["lam"] = lam
        return batched_inputs
