# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : distill_baseline.py
#@time   : 2020-12-29 09:35:40
"""

import torch

from .class_level_baseline import ClassLevelBaseline
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class DistillBaseline(ClassLevelBaseline):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, batched_inputs, is_feat=False, preact=False):
        images = self.preprocess_image(batched_inputs)
        if is_feat:
            features, out = self.backbone(images, is_feat=is_feat, preact=preact)
        else:
            out = self.backbone(images, is_feat=is_feat, preact=preact)

        if self.training:
            assert "targets" in batched_inputs, "targets are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(out, targets)

        else:
            outputs = self.heads(out)
        if is_feat:
            return features, outputs
        return outputs

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images
