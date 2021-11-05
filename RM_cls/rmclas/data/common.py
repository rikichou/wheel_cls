# !/user/bin/env python
# coding=utf-8
"""
@project : RM_clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : common.py
#@time   : 2020-12-22 21:01:52
"""

from torch.utils.data import Dataset
from .data_utils import read_image
from tabulate import tabulate
from termcolor import colored
import logging

logger = logging.getLogger(__name__)


class CommDataset(Dataset):
    """Image classification Dataset"""

    def __init__(self, img_items, label_items, transform=None, relabel=True):
        self.img_items = img_items
        self.labels = label_items
        self.transform = transform
        self.relabel = relabel

        if relabel:
            self.label_dict = dict([(p, i) for i, p in enumerate(self.labels)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, label = self.img_items[index]
        img = read_image(img_path, "BGR")
        if self.transform is not None: img = self.transform(img)
        if self.relabel: label = self.label_dict[label]
        return {
            "images": img,
            "targets": label,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.labels)

    def show_label(self):
        headers = ['class_name', '# class_id']
        if self.relabel:
            csv_results = [[item[0], item[1]] for item in self.label_dict.items()]
        else:
            csv_results = [[item, item] for item in self.labels]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))
