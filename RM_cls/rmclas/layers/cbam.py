# encoding: utf-8
"""
@author:  lmliu
@contact: lmliu@streamax.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CbamBlock", "ConvBlock"]


def get_activation_layer(activation, out_channels):
    """
    Create activation layer from string/function.
    """
    assert (activation is not None)

    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "relu6":
        return nn.ReLU6(inplace=True)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "prelu":
        return nn.PReLU(out_channels)
    else:
        raise NotImplementedError()


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                dilation=1, groups=1, bias=False, use_bn=True, bn_eps=1e-5,
                activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation, out_channels)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class MeanConv(nn.Module):
    """自定义卷积核
        卷积核参数不更新，通过1x1实现mean操作，卷积核大小为1/C，最终featuremap=channel_sum(x)/channel_num=channel_mean(x)
    """
    def __init__(self, channels):
        super(MeanConv, self).__init__()
        self.weight = nn.Parameter(data=torch.ones(1, channels, 1, 1) / channels, requires_grad=False)

    def forward(self, x):
        out = F.conv2d(x, self.weight, stride=1, padding=0)
        return out


class MLP(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        self.fc1 = nn.Linear(in_features=channels, out_features=mid_channels)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=mid_channels, out_features=channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.mlp = MLP(channels=channels, reduction_ratio=reduction_ratio)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = F.sigmoid

    def forward(self, x):
        n,c,_,_ = x.size()
        att1 = self.avg_pool(x)
        att1 = self.mlp(att1)
        att2 = self.max_pool(x)
        att2 = self.mlp(att2)
        att = att1 + att2
        att = self.sigmoid(att)
        # att = att.unsqueeze(2).unsqueeze(3).expand_as(x)
        att = att.view(n,c,1,1)
        x = x * att
        return x


class SpatialGate(nn.Module):
    """自定义空间注意力网络
        caffe不支持tensor的mean和max操作，改用1x1卷积代替mean，取消max
    """
    def __init__(self,channels):
        super(SpatialGate, self).__init__()
        # self.max_fake = ConvBlock(in_channels=channels, out_channels=1, kernel_size=3,
        #                     stride=1, padding=1, bias=False, use_bn=False, activation=None)
        self.conv = ConvBlock(in_channels=1, out_channels=1, kernel_size=7,
                            stride=1, padding=3, bias=False, use_bn=True, activation=None)
        self.mean = MeanConv(channels)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = F.sigmoid

    def forward(self, x):
        # n,c,h,w = x.size()
        att1 = self.mean(x)
        # att2 = self.max_fake(x)
        # att = torch.cat((att1, att2), dim=1)
        att = self.conv(att1)
        att = self.sigmoid(att).expand_as(x)
        # att = att.expand(n,c,h,w)
        x = x * att
        # att1 = x.max(dim=1)[0].unsqueeze(1)
        # att2 = x.mean(dim=1).unsqueeze(1)
        # att = torch.cat((att1, att2), dim=1)
        # att = self.conv(att)
        # att = self.sigmoid(att).expand_as(x)
        # x = x * att
        return x


class CbamBlock(nn.Module):
    """受限于caffe约束，对于spatialattention仅支持mean方式构建cbam
    """
    def __init__(self, channels, reduction_ratio=16):
        super(CbamBlock, self).__init__()
        self.ch_gate = ChannelGate(channels=channels, reduction_ratio=reduction_ratio)
        self.sp_gate = SpatialGate(channels)

    def forward(self, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x