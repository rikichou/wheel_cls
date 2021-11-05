# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : kd_config.py
#@time   : 2020-12-29 09:58:23
"""

from rmclas.config import CfgNode as CN


def update_model_teacher_config(cfg):
    cfg = cfg.clone()

    frozen = cfg.is_frozen()

    cfg.defrost()

    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.HEADS = CN()
    _C.MODEL_TEACHER.HEADS.NAME = "EmbeddingHead"

    # Pooling layer type
    _C.MODEL_TEACHER.HEADS.POOL_LAYER = "avgpool"
    _C.MODEL_TEACHER.HEADS.NECK_FEAT = "before"
    _C.MODEL_TEACHER.HEADS.CLS_LAYER = "linear"

    _C.MODEL_TEACHER.HEADS.NORM = 'BN'
    _C.MODEL_TEACHER.HEADS.SCALE = 64
    _C.MODEL_TEACHER.HEADS.MARGIN = 0.35

    _C.MODEL_TEACHER.HEADS.NUM_CLASSES = 2

    cfg.MODEL.META_ARCHITECTURE = cfg.MODEL_TEACHER.META_ARCHITECTURE
    # ---------------------------------------------------------------------------- #
    # teacher model Backbone options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.BACKBONE.NAME = cfg.MODEL_TEACHER.BACKBONE.NAME
    # Input feature dimension
    cfg.MODEL.BACKBONE.FEAT_DIM = cfg.MODEL_TEACHER.BACKBONE.FEAT_DIM
    cfg.MODEL.BACKBONE.PRETRAIN = False

    # Backbone specific
    # Used for mobilenetv2
    cfg.MODEL.BACKBONE.T = cfg.MODEL_TEACHER.BACKBONE.T  # expansion ratio
    cfg.MODEL.BACKBONE.WIDTH_MULT = cfg.MODEL_TEACHER.BACKBONE.WIDTH_MULT  # channels multiplier
    # Used for resnetV2
    cfg.MODEL.BACKBONE.BLOCK_NAME = cfg.MODEL_TEACHER.BACKBONE.BLOCK_NAME
    cfg.MODEL.BACKBONE.BLOCKS = cfg.MODEL_TEACHER.BACKBONE.BLOCKS
    # Used for ShuffleNetv2
    cfg.MODEL.BACKBONE.NETSIZE = cfg.MODEL_TEACHER.BACKBONE.NETSIZE  # choice [0.2, 0.3, 0.5, 1, 1.5, 2]
    # Used for vgg
    cfg.MODEL.BACKBONE.CONFIGURATION = cfg.MODEL_TEACHER.BACKBONE.CONFIGURATION  # choice ['A', 'B', 'D', 'E', 'S']
    cfg.MODEL.BACKBONE.BATCHNORM = cfg.MODEL_TEACHER.BACKBONE.BATCHNORM
    # Used for wrn
    cfg.MODEL.BACKBONE.DEPTH = cfg.MODEL_TEACHER.BACKBONE.DEPTH  # choice [16, 40]
    cfg.MODEL.BACKBONE.WIDEN_FACTOR = cfg.MODEL_TEACHER.BACKBONE.WIDEN_FACTOR  # channels multiplier

    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.HEADS.NAME = cfg.MODEL_TEACHER.HEADS.NAME

    # Pooling layer type
    cfg.MODEL.HEADS.POOL_LAYER = cfg.MODEL_TEACHER.HEADS.POOL_LAYER

    cfg.MODEL.HEADS.SCALE = cfg.MODEL_TEACHER.HEADS.SCALE
    cfg.MODEL.HEADS.MARGIN = cfg.MODEL_TEACHER.HEADS.MARGIN

    cfg.MODEL.HEADS.NUM_CLASSES = cfg.MODEL_TEACHER.HEADS.NUM_CLASSES

    if frozen: cfg.freeze()

    return cfg


def add_kd_config(cfg):
    _C = cfg

    _C.MODEL_TEACHER = CN()
    _C.MODEL_TEACHER.META_ARCHITECTURE = 'ClassLevelBaseline'

    # ---------------------------------------------------------------------------- #
    # teacher model Backbone options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.BACKBONE = CN()

    _C.MODEL_TEACHER.BACKBONE.NAME = "build_resnetv2_backbone"

    # Backbone generic
    # Backbone feature dimension
    _C.MODEL_TEACHER.BACKBONE.FEAT_DIM = 2048

    # Backbone specific
    # Used for mobilenetv2
    _C.MODEL_TEACHER.BACKBONE.T = 6  # expansion ratio
    _C.MODEL_TEACHER.BACKBONE.WIDTH_MULT = 1  # channels multiplier
    # Used for resnetV2
    _C.MODEL_TEACHER.BACKBONE.BLOCK_NAME = "Bottleneck"
    _C.MODEL_TEACHER.BACKBONE.BLOCKS = [3, 4, 6, 3]
    # Used for ShuffleNetv2
    _C.MODEL_TEACHER.BACKBONE.NETSIZE = 1  # choice [0.2, 0.3, 0.5, 1, 1.5, 2]
    # Used for vgg
    _C.MODEL_TEACHER.BACKBONE.CONFIGURATION = 'A'  # choice ['A', 'B', 'D', 'E', 'S']
    _C.MODEL_TEACHER.BACKBONE.BATCHNORM = True
    # Used for wrn
    _C.MODEL_TEACHER.BACKBONE.DEPTH = 16  # choice [16, 40]
    _C.MODEL_TEACHER.BACKBONE.WIDEN_FACTOR = 1  # channels multiplier

    # ---------------------------------------------------------------------------- #
    # teacher model HEADS options
    # ---------------------------------------------------------------------------- #
    _C.MODEL_TEACHER.HEADS = CN()
    _C.MODEL_TEACHER.HEADS.NAME = "EmbeddingHead"

    # Pooling layer type
    _C.MODEL_TEACHER.HEADS.POOL_LAYER = "avgpool"
    _C.MODEL_TEACHER.HEADS.NECK_FEAT = "before"
    _C.MODEL_TEACHER.HEADS.CLS_LAYER = "linear"

    _C.MODEL_TEACHER.HEADS.NORM = 'BN'
    _C.MODEL_TEACHER.HEADS.SCALE = 64
    _C.MODEL_TEACHER.HEADS.MARGIN = 0.35

    _C.MODEL_TEACHER.HEADS.NUM_CLASSES = 2

    # ---------------------------------------------------------------------------- #
    # distill options
    # ---------------------------------------------------------------------------- #
    # Used for distill
    _C.MODEL.STUDENT_WEIGHTS = ""
    _C.MODEL.TEACHER_WEIGHTS = ""
    _C.MODEL.MODEL_TEACHER = ""

    # distill parameter
    _C.DISTILL = CN()
    _C.DISTILL.METHOD = "kd"  # choices=['kd', 'hint', 'attention', 'similarity',
                                     # 'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                     # 'rkd', 'pkt', 'abound', 'factor', 'nst']
    _C.DISTILL.GAMMA = 0.1
    _C.DISTILL.ALPHA = 0.9
    _C.DISTILL.BETA  = 0
    # KL distillation
    _C.DISTILL.KD_T  = 4

    # NCE distillation
    _C.DISTILL.FEAT_DIM = 128

    # hint layer
    _C.DISTILL.HINT_LAYER = 2  # choices=[0, 1, 2, 3, 4]

    _C.DISTILL.INIT_EPOCHS = 30

    _C.DISTILL.KL_CHOICE = 'jsdiv'  # choices=['jsdiv', 'kl']
