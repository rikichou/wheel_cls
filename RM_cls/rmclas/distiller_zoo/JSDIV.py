# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : JSDIV.py
#@time   : 2021-01-04 19:33:46
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .KD import DistillKL


class DistillJSDIV(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillJSDIV, self).__init__()
        self.kl = DistillKL(T)

    def forward(self, y_s, y_t):
        p_s = F.softmax(y_s, dim=1)
        p_t = F.softmax(y_t, dim=1)
        loss = (self.kl(p_s, p_t) + self.kl(p_t, p_s)) / 2
        return loss



