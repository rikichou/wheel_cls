# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : kd_train_net.py
#@time   : 2021-01-04 09:23:31
"""


import sys

sys.path.append('.')
from rmclas.config import get_cfg
from rmclas.engine import KDTrainer, default_argument_parser, default_setup, DefaultTrainer, launch
from rmclas.utils.checkpoint import Checkpointer
from rmclas.config import add_kd_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_kd_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
        res = DefaultTrainer.test(cfg, model)
        return res

    if args.kd: trainer = KDTrainer(cfg)
    else:       raise ValueError("please use train_net.py instead")

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--kd", action="store_true", help="kd training with teacher model guided")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
