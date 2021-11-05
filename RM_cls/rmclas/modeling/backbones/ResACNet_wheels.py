# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : ResACNet.py
#@time   : 2021-01-07 18:04:29
"""

import numpy as np
import torch
import torch.nn as nn

import logging
from rmclas.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from rmclas.layers import ACBConv2d

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, pad, deploy=False):
        super(BasicBlock, self).__init__()
        self.conv1 = ACBConv2d(inplanes, planes, kernel_size, stride, pad, deploy=deploy, with_bn=True)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class Bottleneck_a(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, pad, r, deploy=False):
        super(Bottleneck_a, self).__init__()
        self.block1 = WheelsMaskBlock(inplanes, planes, kernel_size, stride[0], pad, in_features = inplanes, deploy=False, dilate_ratio = 4)
        self.block2 = BasicBlock(planes, planes, kernel_size, stride[1], pad, deploy=deploy)
        self.block3 = BasicBlock(planes, planes, kernel_size, stride[1], pad, deploy=deploy)
        # self.se_pool = nn.AdaptiveAvgPool2d(1)
        # self.se_fc1 = nn.Linear(planes, r, bias=False)
        # self.se_fc1_relu = nn.ReLU(inplace=True)
        # self.se_fc2 = nn.Linear(r, planes, bias=False)
        # self.increase_prob = nn.Sigmoid()

    def forward(self, x):
        out = self.block1(x)
        temp = out
        out = self.block2(out)
        out = self.block3(out)
        # b, c, _, _ = out.size()
        # res = self.se_pool(out)
        # res = res.view(b, c)
        # res = self.se_fc1(res)
        # res = self.se_fc1_relu(res)
        # res = self.se_fc2(res)
        # res = self.increase_prob(res)
        # res = res.view(b, c, 1, 1)
        # res = out * res
        # res = res + temp
        res = out + temp
        return res


class Bottleneck_b(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, pad, r, deploy=False):
        super(Bottleneck_b, self).__init__()
        self.block1 = BasicBlock(inplanes, planes, kernel_size, stride, pad, deploy=deploy)
        self.block2 = BasicBlock(planes, planes, kernel_size, stride, pad, deploy=deploy)
        # self.se_pool = nn.AdaptiveAvgPool2d(1)
        # self.se_fc1 = nn.Linear(planes, r, bias=False)
        # self.se_fc1_relu = nn.ReLU(inplace=True)
        # self.se_fc2 = nn.Linear(r, planes, bias=False)
        # self.increase_prob = nn.Sigmoid()

    def forward(self, x):
        temp = x
        out = self.block1(x)
        out = self.block2(out)
        # b, c, _, _ = out.size()
        # res = self.se_pool(out)
        # res = res.view(b, c)
        # res = self.se_fc1(res)
        # res = self.se_fc1_relu(res)
        # res = self.se_fc2(res)
        # res = self.increase_prob(res)
        # res = res.view(b, c, 1, 1)
        # res = out * res
        # res = res + temp
        res = out + temp
        return res

class WheelsMaskBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, pad, in_features, deploy=False, dilate_ratio = 4):
        super(BasicBlock, self).__init__()
        self.in_features = in_features
        self.dilate_ratio = dilate_ratio
        self.conv1 = ACBConv2d(inplanes, planes, kernel_size, stride, pad, deploy=deploy, with_bn=True)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU(planes)

    def _create_tire_featuremap(in_features, dilate_ratio):
        feature_map = np.ones([in_features, in_features])
        center = np.floor(in_features/dilate_ratio).astype(np.int)
        stride = np.floor((in_features - center)/2).astype(np.int)

        feature_map[stride:(in_features-stride), stride:(in_features-stride)] = 0
        feature_tensor = torch.from_numpy(feature_map)

        return feature_tensor

    def _map_mask(self, x, in_features, dilate_ratio):
        feature_map = self._create_tire_featuremap(in_features, dilate_ratio)
        x.mul(feature_map)

        return x
    
    def forward(self, x):
        x = self._map_mask(x, self.in_features, self.dilate_ratio)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class resnet18(nn.Module):
    def __init__(self, deploy=False):
        super(resnet18, self).__init__()
        self.layer1 = Bottleneck_a(3, 16, 3, [2, 1], 1, 4, deploy=deploy)
        self.layer2 = Bottleneck_a(16, 32, 3, [2, 1], 1, 8, deploy=deploy)
        # self.layer3 = Bottleneck_b(32, 32, 3, 1, 1, 8)
        self.layer4 = Bottleneck_a(32, 64, 3, [2, 1], 1, 16, deploy=deploy)
        # self.layer5 = Bottleneck_b(64, 64, 3, 1, 1, 16)
        # self.layer6 = Bottleneck_b(64, 64, 3, 1, 1, 16)
        # self.layer7 = Bottleneck_b(64, 64, 3, 1, 1, 16)
        self.layer8 = Bottleneck_a(64, 64, 3, [2, 1], 1, 32, deploy=deploy)
        # self.fc1 = nn.Linear(64, 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, is_feat=False, preact=False):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer4 = self.layer4(layer2)
        layer8 = self.layer8(layer4)

        if is_feat:
            return [layer1, layer2, layer4, layer8], layer8
        else:
            return layer8


@BACKBONE_REGISTRY.register()
def build_ResAcNet_wheels_backbone(cfg):
    """
    Create a ResAcNet instance from config.
    Returns:
        ResAcNet: a :class:`ResAcNet` instance.
    """
    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    deploy        = cfg.MODEL.BACKBONE.DEPLOY
    # fmt: on

    model = resnet18(deploy)

    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                o_state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
            if 'state_dict' in o_state_dict:
                o_state_dict = o_state_dict['state_dict']
            state_dict = {}
            for k, v in o_state_dict.items():
                if 'module.' in k:
                    k = k.replace('module.', '')
                state_dict[k] = v
        else:
            raise ValueError("you should provide pretrain path, for we have no imagenet pretrain model to download")

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model
