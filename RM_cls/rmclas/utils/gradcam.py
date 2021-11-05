"""
@project : Rm_Clas
@author  : xcli
@contact : xcli@streamax.com
#@file   : gradcam.py
#@time   : 2020-08-02 09:16:57
"""

from gradcam_utils import GradCAM
#from rmclas.engine import DefaultTrainer, default_argument_parser, default_setup, launch
import cv2
import torch
import torch.nn as nn
import numpy as np

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0 and self.cols_to_crop == 0:
            return input
        elif self.rows_to_crop > 0 and self.cols_to_crop == 0:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, :]
        elif self.rows_to_crop == 0 and self.cols_to_crop > 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


class ACBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False, use_affine=True, with_bn=True):
        super(ACBConv2d, self).__init__()
        self.deploy = deploy
        self.with_bn = with_bn
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (padding, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, padding)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            if self.with_bn:
                self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
                self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
                self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

    def forward(self, input):
        if self.deploy:
            result = self.fused_conv(input)
            return result
        else:
            square_outputs = self.square_conv(input)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            if self.with_bn:
                square_outputs = self.square_bn(square_outputs)
                vertical_outputs = self.ver_bn(vertical_outputs)
                horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, pad, deploy=False):
        super(BasicBlock, self).__init__()
        self.conv1 = ACBConv2d(inplanes, planes, kernel_size, stride, pad, deploy=deploy, with_bn=True)
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
        self.block1 = BasicBlock(inplanes, planes, kernel_size, stride[0], pad, deploy=deploy)
        self.block2 = BasicBlock(planes, planes, kernel_size, stride[1], pad, deploy=deploy)
        self.block3 = BasicBlock(planes, planes, kernel_size, stride[1], pad, deploy=deploy)


    def forward(self, x):
        out = self.block1(x)
        temp = out
        out = self.block2(out)
        out = self.block3(out)
        res = out + temp
        return res

class resnet18(nn.Module):
    def __init__(self, deploy=False):
        super(resnet18, self).__init__()
        self.layer1 = Bottleneck_a(3, 16, 3, [2, 1], 1, 4, deploy=deploy)
        self.layer2 = Bottleneck_a(16, 32, 3, [2, 1], 1, 8, deploy=deploy)
        self.layer4 = Bottleneck_a(32, 64, 3, [2, 1], 1, 16, deploy=deploy)
        self.layer8 = Bottleneck_a(64, 64, 3, [2, 1], 1, 32, deploy=deploy)

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


class ClsHead(nn.Module):
    def __init__(self):
        super(ClsHead, self).__init__()
        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        x = self.pool_layer(features)
        x = x[..., 0, 0]
        x = self.classifier(x)
        x = self.softmax(x)
        return x


class meta_arch(nn.Module):
    def __init__(self):
        super(meta_arch, self).__init__()
        self.backbone = resnet18(deploy=True)
        self.heads = ClsHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.heads(x)
        return x

if __name__ == '__main__':

    model_dic = torch.load('/xcli/weights/ClsWheel_ACNet.pth')
    model = meta_arch()
    model.load_state_dict(model_dic, strict=False)
    model.cuda()

    #model = GradCAM(model, target_layer_name = 'None')

    imgpath = '/xcli/wheels_cls/wheel_imgs/wheel_imgs_1/test/0000/_02_06_0000_200526054722_2005260547260167.jpg'
    img = cv2.imread(imgpath)
    img_re = cv2.resize(img, (112, 112))

    img_re = img_re - np.array([127.5, 127.5, 127.5])
    img_re = np.transpose(img_re, (2, 0, 1))
    img_re = img_re[np.newaxis, :]
    img_re = img_re.astype(np.float32)
    img_re = torch.from_numpy(img_re)
    img_re = img_re.cuda()
    results = model(img_re)

    print(results)

    #cv2.imshow(results)