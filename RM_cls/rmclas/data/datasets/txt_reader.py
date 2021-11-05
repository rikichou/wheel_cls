# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : txt_reader.py
#@time   : 2021-01-05 16:27:52
"""

import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class TxTReader(ImageDataset):
    """TxT_Reader.
    img_path:label

    Dataset statistics:
        - identities: ... (+1 for background).
        - images: ... (train) + ... (val) + ... (test).
    """

    def __init__(self, root='datasets', dataset_name='dataset_name', **kwargs):
        self.root = root
        self.dataset_name = dataset_name
        self.dataset_dir = osp.join(self.root, dataset_name)
        self.train_txt = osp.join(self.dataset_dir, 'train.txt')
        self.test_txt = osp.join(self.dataset_dir, 'test.txt')

        required_files = [
            self.dataset_dir,
            self.train_txt,
            self.test_txt
        ]

        self.check_before_run(required_files)

        train = self.process_txt(self.train_txt)
        test = self.process_txt(self.test_txt)
        if osp.exists(osp.join(self.dataset_dir, 'val.txt')):
            val = self.process_txt(osp.join(self.dataset_dir, 'val.txt'))
            valeqtest = False
        else:
            val = test
            valeqtest = True
        super(TxTReader, self).__init__(train, test, val, valeqtest=valeqtest, **kwargs)

    def process_txt(self, txt_path):
        """process_txt

        :param txt_path: txt文件路径
        img_path:label
        :return: data
        """
        data = []

        with open(txt_path, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break
                img_name, img_label = [i for i in lines.strip().split(':')]

                img_name = osp.join(self.dataset_dir, img_name)
                # img_label = [i for i in img_label.strip().split(',')]

                data.append((img_name, img_label))
        return data
