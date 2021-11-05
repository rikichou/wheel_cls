# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : meta_arch_test.py
#@time   : 2020-12-26 14:49:00
"""

import sys
sys.path.append('.')
from rmclas.config import cfg
from rmclas.modeling import build_model
import argparse

# cfg.DATALOADER.SAMPLER = 'triplet'
cfg.DATASETS.NAMES = ("Luggage",)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    cfg.MODEL.META_ARCHITECTURE = 'ClassLevelBaseline'
    arch1 = build_model(cfg)
    # arch2 = mix_class_level_baseline(cfg)
    # arch3 = pair_wise_baseline(cfg)
