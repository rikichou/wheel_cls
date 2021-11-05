from __future__ import print_function
import sys

caffe_root = '/lmliu/lmliu/code/caffe-master/'
sys.path.insert(0, caffe_root + 'python')

from caffe import layers as L
from .layers import conv, conv_bn


def mobile_module(name, input, num_3x3, num_1x1, stride=1):
    dw_name = name + "/dw"
    model3 = L.Convolution(input, kernel_size=3, stride=stride, num_output=num_3x3, group=num_3x3, bias_term=False,
                           pad=1, engine=1, weight_filler=dict(type='xavier'))  # ,name = dw_name
    bn1 = L.BatchNorm(model3, use_global_stats=True, in_place=True)
    scale1 = L.Scale(bn1, bias_term=True, in_place=True)
    relu1 = L.ReLU(scale1, in_place=True)

    model1 = L.Convolution(relu1, kernel_size=1, stride=1, num_output=num_1x1, bias_term=False, pad=0,
                           weight_filler=dict(type='xavier'))  # ,name = name
    bn2 = L.BatchNorm(model1, use_global_stats=True, in_place=True)
    scale2 = L.Scale(bn2, bias_term=True, in_place=True)
    relu2 = L.ReLU(scale2, in_place=True)

    return relu2


def res_block(input, num_output=32, stride=1):
    conv1 = conv(input=input, output=num_output, kernel_size=3, stride=stride)
    conv2 = L.Convolution(conv1, kernel_size=3, stride=1, num_output=num_output, bias_term=False, pad=1, weight_filler=dict(type='xavier'))
    bn = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
    block1 = L.Scale(bn, bias_term=True, in_place=True)

    if stride == 1:
        block2 = input
    else:
        conv3 = L.Convolution(input, kernel_size=1, stride=stride, num_output=num_output, bias_term=False, pad=0, weight_filler=dict(type='xavier'))
        bn = L.BatchNorm(conv3, use_global_stats=False, in_place=True)
        block2 = L.Scale(bn, bias_term=True, in_place=True)

    residual_eltwise = L.Eltwise(block1, block2, eltwise_param=dict(operation=1))
    relu = L.ReLU(residual_eltwise, in_place=True)
    return relu


def BasicRFB(name, input, inchannels, outchannels, stride=1, scale=1.0, map_reduce=8, vision=1):
    inter_planes = inchannels // map_reduce

    branch0_0 = conv_bn(name + "_branch0_0", input, inter_planes, kernel_size=1, stride=1, pad=0, activation=False)
    branch0_1 = conv_bn(name + "_branch0_1", branch0_0, 2 * inter_planes, kernel_size=3, stride=stride, pad=1)
    branch0_2 = conv_bn(name + "_branch0_2", branch0_1, 2 * inter_planes, kernel_size=3, stride=1, pad=vision + 1, dilation=vision + 1, activation=False)

    branch1_0 = conv_bn(name + "_branch1_0", input, inter_planes, kernel_size=1, stride=1, pad=0, activation=False)
    branch1_1 = conv_bn(name + "_branch1_1", branch1_0, 2 * inter_planes, kernel_size=3, stride=stride, pad=1)
    branch1_2 = conv_bn(name + "_branch1_2", branch1_1, 2 * inter_planes, kernel_size=3, stride=1, pad=vision + 2, dilation=vision + 2, activation=False)

    branch2_0 = conv_bn(name + "_branch2_0", input, inter_planes, kernel_size=1, stride=1, pad=0, activation=False)
    branch2_1 = conv_bn(name + "_branch2_1", branch2_0, (inter_planes // 2) * 3, kernel_size=3, stride=1, pad=1)
    branch2_2 = conv_bn(name + "_branch2_2", branch2_1, 2 * inter_planes, kernel_size=3, stride=stride, pad=1)
    branch2_3 = conv_bn(name + "_branch2_3", branch2_2, 2 * inter_planes, kernel_size=3, stride=1, pad=vision + 4, dilation=vision + 4, activation=False)

    concat = L.Concat(branch0_2, branch1_2, branch2_3, axis=1)
    branch_linear = conv_bn(name + "_branch_linear", concat, outchannels, kernel_size=1, stride=1, pad=0, activation=False)

    branch_shortcut = conv_bn(name + "_branch_shortcut", input, outchannels, kernel_size=1, stride=stride, pad=0, activation=False)

    residual_eltwise = L.Eltwise(branch_linear, branch_shortcut, eltwise_param=dict(operation=1))

    relu = L.ReLU(residual_eltwise, in_place=True)
    return relu


def InvertedResidual(name, input, in_channels, out_channels, stride, expand_ratio):
    use_res_connect = stride == 1 and in_channels == out_channels
    hidden_dim = int(round(in_channels * expand_ratio))

    if expand_ratio != 1:
        conv1 = conv_bn(name+'conv1', input, hidden_dim, kernel_size=1, stride=1, pad=0, activation=True, dilation=1)

        conv2 = L.Convolution(conv1, kernel_size=3, stride=stride, num_output=hidden_dim, group=hidden_dim, bias_term=False,
                              pad=1, engine=1, weight_filler=dict(type='xavier'))

        bn1 = L.BatchNorm(conv2, use_global_stats=True, in_place=True)
        scale1 = L.Scale(bn1, bias_term=True, in_place=True)
        relu1 = L.ReLU(scale1, in_place=True)

        conv3 = L.Convolution(relu1, kernel_size=1, stride=1, num_output=out_channels, bias_term=False, pad=0,
                              weight_filler=dict(type='xavier'))
        bn2 = L.BatchNorm(conv3, use_global_stats=True, in_place=True)
        scale2 = L.Scale(bn2, bias_term=True, in_place=True)
    else:
        conv2 = L.Convolution(input, kernel_size=3, stride=stride, num_output=hidden_dim, group=hidden_dim, bias_term=False,
                               pad=1, engine=1, weight_filler=dict(type='xavier'))

        bn1 = L.BatchNorm(conv2, use_global_stats=True, in_place=True)
        scale1 = L.Scale(bn1, bias_term=True, in_place=True)
        relu1 = L.ReLU(scale1, in_place=True)

        conv3 = L.Convolution(relu1, kernel_size=1, stride=1, num_output=out_channels, bias_term=False, pad=0,
                              weight_filler=dict(type='xavier'))
        bn2 = L.BatchNorm(conv3, use_global_stats=True, in_place=True)
        scale2 = L.Scale(bn2, bias_term=True, in_place=True)
    if use_res_connect:
        residual_eltwise = L.Eltwise(input, scale2, eltwise_param=dict(operation=1))
        return residual_eltwise
    else:
        return scale2


def BasicBlock(name, input, planes, kernel_size, stride, pad):
    conv1 = conv_bn(name + 'conv1', input, planes, kernel_size=kernel_size, stride=stride, pad=pad, activation=False, dilation=1, bias_term=True)
    out = L.PReLU(conv1, in_place=True)
    return out


def Bottleneck_a(name, input, planes, kernel_size, stride, pad):
    block1 = BasicBlock(name+'block1', input, planes, kernel_size, stride[0], pad)
    block2 = BasicBlock(name+'block2', block1, planes, kernel_size, stride[1], pad)
    block3 = BasicBlock(name+'block3', block2, planes, kernel_size, stride[1], pad)
    return L.Eltwise(block3, block1, eltwise_param=dict(operation=1))


def Bottleneck_b(name, input, planes, kernel_size, stride, pad):
    block1 = BasicBlock(name+'block1', input, planes, kernel_size, stride, pad)
    block2 = BasicBlock(name+'block2', block1, planes, kernel_size, stride, pad)
    return L.Eltwise(block2, input, eltwise_param=dict(operation=1))

