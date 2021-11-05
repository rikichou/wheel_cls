# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : build.py
#@time   : 2020-12-23 19:46:30
"""

from ...utils.registry import Registry

HEADS_REGISTRY = Registry("HEADS")
HEADS_REGISTRY.__doc__ = """
Registry for heads 
"""


def build_heads(cfg):
    """
    Build Heads defined by `cfg.MODEL.HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return HEADS_REGISTRY.get(head)(cfg)