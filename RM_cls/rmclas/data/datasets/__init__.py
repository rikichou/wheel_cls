# !/user/bin/env python
# coding=utf-8
"""
@project : RM_clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : __init__.py
#@time   : 2020-12-22 20:32:28
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`ImageDataset`.
"""

# classifaction datasets
from .txt_reader import TxTReader
from .dir_reader import DirReader

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
