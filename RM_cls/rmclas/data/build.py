# !/user/bin/env python
# coding=utf-8
"""
@project : RM_clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : build.py
#@time   : 2020-12-23 10:22:16
"""

import torch
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader

from rmclas.utils import comm
from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms


def build_train_loader(cfg, mapper=None):
    cfg = cfg.clone()

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        reader = d[0]
        dataset = DATASET_REGISTRY.get(reader)(cfg.DATASETS.ROOT, d[1])
        if comm.is_main_process():
            dataset.show_dataset()
        train_items.extend(dataset.train)
    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=True)

    train_set = CommDataset(train_items, cfg.DATASETS.CLASSES, transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if cfg.DATALOADER.PK_SAMPLER:
        data_sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size,
                                                     num_instance, train_set.num_classes)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def build_test_loader(cfg, mode):
    cfg = cfg.clone()

    test_items = list()

    assert mode in ['test', 'val', 'test_val'], "mode should be test, val or test_val"
    if mode == 'test':
        for d in cfg.DATASETS.TESTS:
            reader = d[0]
            dataset = DATASET_REGISTRY.get(reader)(cfg.DATASETS.ROOT, d[1])
            test_items.extend(dataset.test)
    elif mode == 'val':
        for d in cfg.DATASETS.TESTS:
            reader = d[0]
            dataset = DATASET_REGISTRY.get(reader)(cfg.DATASETS.ROOT, d[1])
            test_items.extend(dataset.val)
    else:
        for d in cfg.DATASETS.TESTS:
            reader = d[0]
            dataset = DATASET_REGISTRY.get(reader)(cfg.DATASETS.ROOT, d[1])
            if dataset.valeqtest:
                test_items.extend(dataset.test)
            else:
                test_items.extend(dataset.test + dataset.val)
    test_transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(test_items, cfg.DATASETS.CLASSES, test_transforms, relabel=True)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=2,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
