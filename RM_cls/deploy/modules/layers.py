from __future__ import print_function
import sys

caffe_root = '/lmliu/lmliu/code/caffe-master/'
sys.path.insert(0, caffe_root + 'python')

from caffe import layers as L, params as P


def conv_bn(name, input, output, kernel_size=3, stride=1, pad=1, activation=True, dilation=1, bias_term=False):
    conv = L.Convolution(input, kernel_size=kernel_size, stride=stride, num_output=output, bias_term=bias_term, pad=pad, weight_filler=dict(type='xavier'), dilation=dilation)

    # in-place compute means your input and output has the same memory area,which will be more memory effienct
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)

    # scale = L.Scale(bn,filler=dict(value=1),bias_filler=dict(value=0),bias_term=True, in_place=True)
    out = L.Scale(bn, bias_term=True, in_place=True)

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out


def conv(name, input, output, kernel_size=3, stride=1, pad=1, activation=True):
    out = L.Convolution(input, kernel_size=kernel_size, stride=stride, num_output=output, bias_term=True, pad=pad, weight_filler=dict(type='xavier'))  # ,name = name

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out


def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(name, bottom, ks=2, stride=2, padding=0):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride, pad=padding, name=name)


def sigmoid(name, bottom):
    return L.Sigmoid(bottom)


def SeperableConv2d(name, input, num_3x3, num_1x1, stride=1):
    dw_name = name + "/dw"
    model3 = L.Convolution(input, kernel_size=3, stride=stride, num_output=num_3x3, group=num_3x3, bias_term=False,
                           pad=1, engine=1, weight_filler=dict(type='xavier'), name=dw_name)
    relu1 = L.ReLU(model3, in_place=True)
    model1 = L.Convolution(relu1, kernel_size=1, stride=1, num_output=num_1x1, bias_term=False, pad=0,
                           weight_filler=dict(type='xavier'), name=name)

    return model1


def deconv(name, input, output, kernel_size=3, stride=2, pad=1, activation=True):
    out = L.Deconvolution(input, convolution_param=dict(kernel_size=kernel_size, stride=stride, num_output=output, pad=pad), param=[dict(lr_mult=0)])

    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out


def eltwise(input, branch_shortcut):
    out = L.Eltwise(input, branch_shortcut, eltwise_param=dict(operation=1))
    return out


def deconv_bn(name, input, output, kernel_size=3, stride=2, pad=1, activation=True):
    deconv = L.Deconvolution(input, convolution_param=dict(kernel_size=kernel_size, stride=stride, num_output=output, pad=pad), param=[dict(lr_mult=0)])
    bn = L.BatchNorm(deconv, use_global_stats=True, in_place=True)
    out = L.Scale(bn, bias_term=True, in_place=True)
    if activation is True:
        out = L.ReLU(out, in_place=True)
    return out