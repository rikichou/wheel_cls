# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : __init__.py
#@time   : 2020-12-23 17:54:21
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .class_level_baseline import ClassLevelBaseline
from .pair_wise_baseline import PairWiseBaseline
from .mix_class_level_baseline import MixClassLevelBaseline
from .distill_baseline import DistillBaseline


