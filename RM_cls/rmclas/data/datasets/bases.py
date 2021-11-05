# !/user/bin/env python
# coding=utf-8
"""
@project : RM_clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : bases.py
#@time   : 2020-12-22 20:05:11
"""

import copy
import logging
import os
from tabulate import tabulate
from termcolor import colored

logger = logging.getLogger(__name__)


class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.
    Args:
        train (list): contains tuples of (img_path(s), label).
        val (list): contains tuples of (img_path(s), label).
        test (list): contains tuples of (img_path(s), label).
        transform: transform function.
        mode (str): 'train', 'val' or 'test'.
        combineall (bool): combines train, val and test in a
            dataset for training.
        verbose (bool): show information.
        valeqtest (bool): if val is equal to test
    """

    def __init__(self, train, val, test, transform=None, mode='train',
                 combineall=False, verbose=True, valeqtest=True, **kwargs):
        self.train = train
        self.val = val
        self.test = test
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose
        self.valeqtest = valeqtest

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'val':
            self.data = self.val
        elif self.mode == 'test':
            self.data = self.test
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | val | test]'.format(self.mode))

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def show_dataset(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, val and test in a dataset for training."""
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, label in data:
                combined.append((img_path, label))

        _combine_data(self.val)
        if not self.valeqtest:
            _combine_data(self.test)

        self.train = combined

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    """

    def __init__(self, train, val, test, **kwargs):
        super(ImageDataset, self).__init__(train, val, test, **kwargs)

    def show_dataset(self):
        headers = ['subset', 'images']
        csv_results = [[self.mode, len(self.data)]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.dataset_name} in csv format: \n" + colored(table, "cyan"))
