# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : acblock_fusion.py
#@time   : 2021-01-08 10:08:01
"""

import numpy as np
import torch

SQUARE_KERNEL_KEYWORD = 'square_conv.weight'

def _fuse_kernel(kernel, gamma, std):
    b_gamma = np.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = np.tile(b_gamma, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    b_std = np.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = np.tile(b_std, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * b_gamma / b_std

def _add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                                        square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

def convert_acblock_weights(backbone, modelpath):
    eps = 1e-11
    with_acblock_dict = torch.load(modelpath,map_location=torch.device('cpu'))["model"]
    # print(with_acblock_dict.keys())
    # print(backbone.state_dict().keys())
    deploy_wo_acblock_dict = backbone.state_dict()
    # 读取非卷积之外的权重参数
    deploy_wo_acblock_dict.update(with_acblock_dict)

    # 将square_conv, ver_conv, hor_conv及其对应bn参数融合至fused_conv
    square_conv_var_names = [name for name in with_acblock_dict.keys() if SQUARE_KERNEL_KEYWORD in name]
    for square_name in square_conv_var_names:
        square_kernel = with_acblock_dict[square_name]
        square_mean = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_mean')]
        square_std = np.sqrt(with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_var')] + eps)
        square_gamma = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.weight')]
        square_beta = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.bias')]

        ver_kernel = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_conv.weight')]
        ver_mean = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_mean')]
        ver_std = np.sqrt(with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.running_var')] + eps)
        ver_gamma = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.weight')]
        ver_beta = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'ver_bn.bias')]

        hor_kernel = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_conv.weight')]
        hor_mean = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_mean')]
        hor_std = np.sqrt(with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.running_var')] + eps)
        hor_gamma = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.weight')]
        hor_beta = with_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'hor_bn.bias')]

        fused_bias = square_beta + ver_beta + hor_beta - square_mean * square_gamma / square_std \
                     - ver_mean * ver_gamma / ver_std - hor_mean * hor_gamma / hor_std
        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)
        _add_to_square_kernel(fused_kernel, _fuse_kernel(ver_kernel, ver_gamma, ver_std))
        _add_to_square_kernel(fused_kernel, _fuse_kernel(hor_kernel, hor_gamma, hor_std))

        deploy_wo_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_wo_acblock_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias
    return deploy_wo_acblock_dict
    # backbone.load_state_dict(deploy_wo_acblock_dict)