# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : cls_head.py
#@time   : 2020-12-23 19:42:32
"""

from torch import nn
import torch.nn.functional as F

from rmclas.layers import *
from rmclas.utils.weight_init import weights_init_classifier
from .build import HEADS_REGISTRY


@HEADS_REGISTRY.register()
class ClsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")

        # Classification layer
        if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=True)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`Heads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]

        classifier_name = self.classifier.__class__.__name__
        # fmt: off
        if classifier_name == 'Linear':
            cls_outputs = self.classifier(global_feat)
            # pred_class_logits = F.linear(global_feat, self.classifier.weight)+self.classifier.bias
            pred_class_logits = self.classifier(global_feat)
        else:
            cls_outputs = self.classifier(global_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(global_feat),
                                                             F.normalize(self.classifier.weight))
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
        }

