# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : dir_reader.py
#@time   : 2021-01-06 11:02:13
"""


import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DirReader(ImageDataset):
    """DirReader.
    img_path:label

    Dataset statistics:
        - identities: ... (+1 for background).
        - images: ... (train) + ... (val) + ... (test).
    """

    def __init__(self, root='datasets', dataset_name='dataset_name', **kwargs):
        self.root = root
        self.dataset_name = dataset_name
        self.dataset_dir = osp.join(self.root, dataset_name)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.test_dir
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        test = self.process_dir(self.test_dir)
        if osp.exists(osp.join(self.dataset_dir, 'val')):
            val = self.process_dir(osp.join(self.dataset_dir, 'val'))
            valeqtest = False
        else:
            val = test
            valeqtest = True
        super(DirReader, self).__init__(train, test, val, valeqtest=valeqtest, **kwargs)

    def process_dir(self, dir_path):
        """process_dir

        :param dir_path: 文件夹路径
        img_path:label
        :return: data
        """
        data = []
        classes = [i for i in os.listdir(dir_path) if osp.isdir(osp.join(dir_path, i))]
        for label in classes:
            class_dir = osp.join(dir_path, label)
            for i in os.listdir(class_dir):
                if i.endswith('.jpg') or i.endswith('.png'):
                    img_name = osp.join(class_dir, i)
                    data.append((img_name, label))
        return data
