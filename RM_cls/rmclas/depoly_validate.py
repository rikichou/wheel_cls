import numpy as np
import sys

sys.path.append('.')
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from config.config import get_cfg
from engine.defaults import DefaultTrainer

caffe_root = '/xcli/codes/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')

import caffe

net_name = 'res_acnet'

if net_name == 'res_acnet':
    net_file = '/xcli/wheels_cls/RM_cls/deploy/caffe_models/wheel_res_acnet/res_acnet.prototxt'
    caffe_model = '/xcli/wheels_cls/RM_cls/deploy/caffe_models/wheel_res_acnet/res_acnet.caffemodel'
    torch_model = '/xcli/wheels_cls/RM_cls/deploy/torch_model/model_final_vertical_flip.pth'
    config_file = '/xcli/wheels_cls/RM_cls/configs/wheel_acnet_ex6.yml'
else:
    raise ValueError('we are not support %s yet' % net_name)

size = (112, 112)

# net = caffe.Net(net_file, caffe_model, caffe.TEST)

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


class Bottleneck_b(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, pad, r, deploy=False):
        super(Bottleneck_b, self).__init__()
        self.block1 = BasicBlock(inplanes, planes, kernel_size, stride, pad, deploy=deploy)
        self.block2 = BasicBlock(planes, planes, kernel_size, stride, pad, deploy=deploy)


    def forward(self, x):
        temp = x
        out = self.block1(x)
        out = self.block2(out)
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


def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
    cfg.freeze()
    return cfg


def forward_pytorch(image, config):
    cfg = setup(config)
    net = meta_arch()#DefaultTrainer.build_model(cfg)
    state_dict = torch.load(torch_model, map_location=torch.device('cpu'))['model']
    state_dict["pixel_mean"] = torch.tensor([[[[0]], [[0]], [[0]]]])

    # print(state_dict.keys())
    # print(net.state_dict().keys())
    net.load_state_dict(state_dict, strict=False)
    net.cuda()
    net.eval()
    image = torch.from_numpy(image)

    image = Variable(image.cuda())

    t0 = time.time()
    out = net(image)
    t1 = time.time()
    print(out)
    return t1-t0, out['pred_class_logits'][0]


def forward_caffe(protofile, weightfile, image):

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(1, 3, size[1], size[0])
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params


if __name__ == '__main__':
    print('This is main ....')

    img = np.ones([1, 3, size[1], size[0]], dtype=np.float32)
    time_pytorch, out = forward_pytorch(img, config_file)
    #time_caffe, caffe_blobs, caffe_params = forward_caffe(net_file, caffe_model, img)

    print('pytorch forward time %d', time_pytorch)
    #print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    out_blob_name = "InnerProduct1"

    det_pytorch_data = out.data.cpu().numpy().flatten()

    #det_caffe_data = caffe_blobs[out_blob_name].data[0][...].flatten()
    #det_diff = abs(det_pytorch_data - det_caffe_data).sum()
    print(det_pytorch_data)
    #print(det_caffe_data)
