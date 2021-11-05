# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : mix_class_level_baseline.py
#@time   : 2020-12-24 21:01:15
"""

from .class_level_baseline import ClassLevelBaseline
from rmclas.modeling.losses import *
from rmclas.utils.events import get_event_storage


def log_accuracy(pred_class_logits, targets_a, targets_b, lam, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.size(0)
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct_a = pred_class.eq(targets_a.view(1, -1).expand_as(pred_class))
    correct_b = pred_class.eq(targets_b.view(1, -1).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = (lam * correct_a[:k].view(-1).float().sum(dim=0, keepdim=True)
                     + (1-lam) * correct_b[:k].view(-1).float().sum(dim=0, keepdim=True))
        ret.append(correct_k.mul_(1. / bsz))

    storage = get_event_storage()
    storage.put_scalar("cls_accuracy", ret[0])


class MixClassLevelBaseline(ClassLevelBaseline):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets_a" in batched_inputs and "targets_b" in batched_inputs, \
                "mix annotation are missing in training!"
            targets_a = batched_inputs["targets_a"].to(self.device)
            targets_b = batched_inputs["targets_b"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets_a.sum() < 0: targets_a.zero_()
            if targets_b.sum() < 0: targets_b.zero_()

            outputs_a = self.heads(features, targets_a)
            outputs_b = self.heads(features, targets_b)
            return {
                "outputs_a": outputs_a,
                "outputs_b": outputs_b,
                "targets_a": targets_a,
                "targets_b": targets_b,
                "lam"      : batched_inputs["lam"]
            }
        else:
            outputs = self.heads(features)
            return outputs

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs_a = outs["outputs_a"]
        outputs_b = outs["outputs_b"]
        targets_a = outs["targets_a"]
        targets_b = outs["targets_b"]
        lam = outs["lam"]
        # model predictions
        pred_class_logits = outputs_a['pred_class_logits'].detach()
        cls_outputs_a       = outputs_a['cls_outputs']
        cls_outputs_b       = outputs_b['cls_outputs']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, targets_a, targets_b, lam)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = (lam * cross_entropy_loss(
                cls_outputs_a,
                targets_a,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
                ) + (1-lam) * cross_entropy_loss(
                    cls_outputs_b,
                    targets_b,
                    self._cfg.MODEL.LOSSES.CE.EPSILON,
                    self._cfg.MODEL.LOSSES.CE.ALPHA,
                )) * self._cfg.MODEL.LOSSES.CE.SCALE

        return loss_dict
