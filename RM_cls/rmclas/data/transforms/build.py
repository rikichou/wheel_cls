# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX
        augmix_prob = cfg.INPUT.AUGMIX_PROB

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        autoaug_prob = cfg.INPUT.AUTOAUG_PROB

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        do_vertical_flip = cfg.INPUT.DO_VERTICAL_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_value = cfg.INPUT.REA.VALUE

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        # random rotation, if rot_expand set to True, image need extra resize after rotation
        do_rot = cfg.INPUT.ROT.ENABLED
        rot_degs = cfg.INPUT.ROT.DEGREES
        rot_expand = cfg.INPUT.ROT.EXPAND

        ''''''
        res.append(CenterMask())
        ''''''
        res.append(T.Resize(size_train))
        
        if do_autoaug:
            res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))
            res.append(T.Resize(size_train))
        if do_rot:
            res.append(T.RandomRotation(rot_degs, expand=rot_expand))
            if rot_expand: 
                res.append(T.Resize(size_train))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_vertical_flip:
            res.append(T.RandomVerticalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode), T.RandomCrop(size_train)])
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_augmix:
            res.append(T.RandomApply([AugMix()], p=augmix_prob))
        

        res.append(ToTensor())
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test))
        res.append(ToTensor())
    return T.Compose(res)
