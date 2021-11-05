# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : __init__.py
#@time   : 2020-12-23 19:45:30
"""

from .build import HEADS_REGISTRY, build_heads

# import all the head, so they will be registered
from .cls_head import ClsHead
from .embedding_head import EmbeddingHead
