# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : evaluation.py
#@time   : 2020-12-26 14:21:31
"""

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from termcolor import colored
from collections import Mapping
from .evaluator import DatasetEvaluator
from rmclas.utils import comm

logger = logging.getLogger(__name__)


def accuracy_numpy(pred, target, topk):
    res = []
    maxk = max(topk)
    num = pred.shape[0]
    pred_label = pred.argsort(axis=1)[:, -maxk:][:, ::-1]

    for k in topk:
        correct_k = np.logical_or.reduce(
            pred_label[:, :k] == target.reshape(-1, 1), axis=1)
        res.append(correct_k.sum() / num)
    return res


def accuracy_torch(pred, target, topk=1):
    res = []
    maxk = max(topk)
    num = pred.size(0)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1. / num))
    return res


def accuracy(pred, target, topk=1):
    """Calculate accuracy according to the prediction and target

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        res = accuracy_torch(pred, target, topk)
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        res = accuracy_numpy(pred, target, topk)
    else:
        raise TypeError('pred and target should both be'
                        'torch.Tensor or np.ndarray')

    return res[0] if return_single else res


def print_csv_format(results):
    """
    Print main metrics in a format similar to Detectron,
    so that they are easy to copypaste into a spreadsheet.
    Args:
        results (OrderedDict[dict]): task_name -> {metric -> score}
    """
    task = list(results.keys())
    metrics = ["metrics"] + task

    csv_results = []
    for task, res in results.items():
        csv_results.append((task, res))

    # tabulate it
    table = tabulate(
        csv_results,
        tablefmt="pipe",
        floatfmt=".2%",
        headers=metrics,
        numalign="left",
    )

    logger.info("Evaluation results in csv format: \n" + colored(table, "cyan"))


def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.
    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


class Evaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir

        self.features = []
        self.targets = []

    def reset(self):
        self.features = []
        self.targets = []

    def process(self, inputs, outputs):
        self.targets.extend(torch.reshape(inputs["targets"], (-1, 1)))
        # print(inputs["targets"].size())
        self.features.append(outputs["pred_class_logits"].cpu())
        # print(outputs["pred_class_logits"].size())

    def evaluate(self):
        topk = (1, )

        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            targets = comm.gather(self.targets)
            targets = sum(targets, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            features = self.features
            targets = self.targets

        results = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)
        
        acc = accuracy(results, targets, topk)
        self._results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
        return copy.deepcopy(self._results)
