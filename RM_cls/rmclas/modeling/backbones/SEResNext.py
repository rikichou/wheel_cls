"""
    SE-ResNeXt for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['SEResNeXt', 'seresnext50_32x4d', 'seresnext101_32x4d', 'seresnext101_64x4d']


import os
import torch
import torch.nn as nn
import torch.nn.init as init
from inspect import isfunction
#from .common import conv1x1_block, SEBlock
from .resnet import ResInitBlock
from .resnext import ResNeXtBottleneck

import logging
from .build import BACKBONE_REGISTRY
from rmclas.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


logger = logging.getLogger(__name__)

def get_activation_layer(activation):
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "relu6":
            return nn.ReLU6(inplace=True)
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Module))
        return activation

def round_channels(channels,
                   divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)#get_activation_layer(activation)

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x

def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)

def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)

class SEBlock(nn.Module):
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation=(lambda: nn.ReLU(inplace=True)),
                 out_activation=(lambda: nn.Sigmoid())):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_conv:
            self.conv1 = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                bias=True)
        else:
            self.fc1 = nn.Linear(
                in_features=channels,
                out_features=mid_channels)
        self.activ = get_activation_layer(mid_activation)
        if use_conv:
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=channels,
                bias=True)
        else:
            self.fc2 = nn.Linear(
                in_features=mid_channels,
                out_features=channels)
        self.sigmoid = get_activation_layer(out_activation)

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.view(x.size(0), -1)
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x

class SEResNeXtUnit(nn.Module):
    """
    SE-ResNeXt unit.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width):
        super(SEResNeXtUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResNeXtBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width)
        self.se = SEBlock(channels=out_channels)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class SEResNeXt(nn.Module):
    """
    SE-ResNeXt model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 in_channels=3,
                 in_size=(112, 112),
                 num_classes=1000):
        super(SEResNeXt, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SEResNeXtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        '''x = x.view(x.size(0), -1)
        x = self.output(x)'''
        return x


def get_seresnext(blocks,
                  cardinality,
                  bottleneck_width,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create SE-ResNeXt model with specific parameters.
    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported SE-ResNeXt with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SEResNeXt(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        '''from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)'''

    return net


def seresnext50_32x4d(**kwargs):
    """
    SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="seresnext50_32x4d", **kwargs)


def seresnext101_32x4d(**kwargs):
    """
    SE-ResNeXt-101 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="seresnext101_32x4d", **kwargs)


def seresnext101_64x4d(**kwargs):
    """
    SE-ResNeXt-101 (64x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="seresnext101_64x4d", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        seresnext50_32x4d,
        seresnext101_32x4d,
        seresnext101_64x4d,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnext50_32x4d or weight_count == 27559896)
        assert (model != seresnext101_32x4d or weight_count == 48955416)
        assert (model != seresnext101_64x4d or weight_count == 88232984)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))

@BACKBONE_REGISTRY.register()
def build_SEResNext_backbone(cfg):
    """
    Create a SEResNext instance from config.
    Returns:
        SEResNext: a :class:`SEResNext` instance.
    """
    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    '''T             = cfg.MODEL.BACKBONE.T
    width_mult    = cfg.MODEL.BACKBONE.WIDTH_MULT'''
    # fmt: on

    model = seresnext50_32x4d()

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


if __name__ == "__main__":
    _test()