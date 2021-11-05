# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : acnet2net.py
#@time   : 2021-01-08 09:52:01
"""

import sys
import torch
import os
sys.path.append('.')

from rmclas.config import get_cfg
from rmclas.engine import DefaultTrainer, default_argument_parser, default_setup
from deploy.acnet.acblock_fusion import convert_acblock_weights
from rmclas.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.BACKBONE.DEPLOY = True

    model = DefaultTrainer.build_model(cfg)
    model.eval()
    state_dict = convert_acblock_weights(model, cfg.MODEL.WEIGHTS)

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(get_missing_parameters_message(incompatible.missing_keys))

    if incompatible.unexpected_keys:
        print(get_unexpected_parameters_message(incompatible.unexpected_keys))

    data = {}
    data["model"] = model.state_dict()
    os.makedirs("./deploy/torch_model/", exist_ok=True)
    torch.save(data, "./deploy/torch_model/acnet_wheel.pth")


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)

