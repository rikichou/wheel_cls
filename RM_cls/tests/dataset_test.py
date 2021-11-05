# !/user/bin/env python
# coding=utf-8
"""
@project : RM_clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : dataset_test.py
#@time   : 2020-12-23 11:29:47
"""

import sys
sys.path.append('.')
from rmclas.data import build_train_loader, build_test_loader
from rmclas.config import cfg
import argparse

# cfg.DATALOADER.SAMPLER = 'triplet'
cfg.DATASETS.NAMES = ("Luggage",)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    train_loader = build_train_loader(cfg)
    test_loader = build_test_loader(cfg, mode="test")


