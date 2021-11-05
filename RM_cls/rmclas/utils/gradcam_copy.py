"""
@project : Rm_Clas
@author  : xcli
@contact : xcli@streamax.com
#@file   : gradcam.py
#@time   : 2020-08-02 09:16:57
reference: https://winycg.blog.csdn.net/article/details/100691786
"""

import numpy as np
from numpy.core.numeric import load
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import copy
import os.path as osp

import sys

from torch.nn import modules
sys.path.append('/xcli/wheels_cls/RM_cls/rmclas/utils/gradcam_model')


import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.cm as cm
from PIL import Image

'''import sys
sys.path.append(os.path.dirname(__file__) + os.sep + './')
from modeling.backbones.Repvgg import build_RepVGG_backbone'''

import faulthandler

MODELLOAD = '/xcli/wheels_cls/RM_cls/deploy/torch_model/RepVGG_wheelcls_deployed_0928.pth'

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        #self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

class ClsHead(nn.Module):
    def __init__(self):
        super(ClsHead, self).__init__()
        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, 2, bias=True)
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
        self.backbone =  create_RepVGG(deploy=True)
        self.heads = ClsHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.heads(x)
        return x

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.size()[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        #target_layer = 'backbone.stage4.0.nonlinearity'#self.model.backbone.stage4[0].nonlinearity
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def save_gradcam(filename, gcam, raw_image, image_size, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet(gcam)[..., :3] * 255.0
    cmap = cmap[:, :, ::-1]
    background = cv2.resize(raw_image, (112, 112))
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * background
    else:
        gcam = (cmap.astype(np.float) + background.astype(np.float)) / 2
    gcam = np.uint8(gcam)
    gcam = cv2.resize(gcam, image_size)
    
    im_raw = cv2.resize(raw_image, image_size)
    im = np.vstack((gcam, im_raw))
    cv2.imwrite(filename, im)
    '''
    im=Image.fromarray(im) 
    im.save(filename)
    '''

def model_load(dict_dir, cuda_flag=True):
    """
    load trained model from the given path
    Input:
    dict_dir: directory storing the trained model's weight dictionary
    cuda_flag: use cuda or not

    return:
    the loaded torch model
    """
    model_dict = torch.load(dict_dir)
    model = meta_arch()
    print(dict_dir)
    model.load_state_dict(model_dict)

    if cuda_flag:
        model.cuda()

    return model

def gradcam_lanuch(image_dir, dict_dir, output_dir, image_size, target_layer):

    model = model_load(dict_dir, cuda_flag=True)

    dirs = []
    for dir in os.listdir(image_dir):
        dirs.append(dir)
    dirs.sort()

    bp = BackPropagation(model)
    gcam = GradCAM(model)

    faulthandler.enable()

    global_id = 0
    for dir in dirs:
        cur_dir = os.path.join(image_dir, dir)
        for jpeg in os.listdir(cur_dir):
            try:
                img_re = cv2.imdecode(np.fromfile(os.path.join(cur_dir, jpeg), dtype = np.uint8), cv2.IMREAD_COLOR)
            except:
                print((jpeg,cur_dir))
            #img_re = cv2.imread(os.path.join(cur_dir, jpeg), cv2.IMREAD_COLOR)
            img_raw = img_re.copy()
            img_re = cv2.resize(img_re, (112, 112))

            img_re = img_re - np.array([127.5, 127.5, 127.5])
            img_re = np.transpose(img_re, (2, 0, 1))
            img_re = img_re[np.newaxis, :]
            img_re = img_re.astype(np.float32)
            img_re = torch.from_numpy(img_re)
            image = img_re.cuda()
            
            probs, ids = bp.forward(image)
            _ = gcam.forward(image)
            gcam.backward(ids=ids[:, [0]])

            regions = gcam.generate(target_layer=target_layer)
            
            save_gradcam(filename=osp.join(
                output_dir, str(global_id))+'.jpg',
                gcam=regions[0, 0],
                raw_image=img_raw,
                image_size = image_size
            )
            

            global_id += 1
    
if __name__ == "__main__":
    image_dir = '/xcli/wheels_cls/wheel_imgs/gradcamtest'
    dirs = []
    for dir in os.listdir(image_dir):
        dirs.append(dir)
    dirs.sort()


    model_dict = torch.load(MODELLOAD)
    if 'state_dict' in model_dict:
        model_dict = model_dict['state_dict']
    elif 'model' in model_dict:
        model_dict = model_dict['model']
    ckpt = {k.replace('backbone.', ''): v for k, v in model_dict.items()}  # strip the names
        #print(ckpt.keys())
    net1 = meta_arch()
    net1.load_state_dict(ckpt, strict=False)
    net1.eval()
    net1.cuda()

    """
    # load pretrained ResNet50 for test
    #net1 = models.__dict__['resnet50'](pretrained=True)
    #net1.cuda()
    #net1.eval()
    """


    bp = BackPropagation(model=net1)
    gcam = GradCAM(model=net1)

    faulthandler.enable()

    global_id = 0
    for dir in dirs:
        cur_dir = os.path.join(image_dir, dir)
        for jpeg in os.listdir(cur_dir):
            try:
                img_re = cv2.imdecode(np.fromfile(os.path.join(cur_dir, jpeg), dtype = np.uint8), cv2.IMREAD_COLOR)
            except:
                print((jpeg,cur_dir))
            #img_re = cv2.imread(os.path.join(cur_dir, jpeg), cv2.IMREAD_COLOR)
            img_raw = img_re.copy()
            img_re = cv2.resize(img_re, (112, 112))

            img_re = img_re - np.array([127.5, 127.5, 127.5])
            img_re = np.transpose(img_re, (2, 0, 1))
            img_re = img_re[np.newaxis, :]
            img_re = img_re.astype(np.float32)
            img_re = torch.from_numpy(img_re)
            image = img_re.cuda()
            
            probs, ids = bp.forward(image)
            _ = gcam.forward(image)
            gcam.backward(ids=ids[:, [0]])

            #regions = gcam.generate(target_layer='layer4')  #for pretrained resnet50
            regions = gcam.generate(target_layer='backbone.stage4.0.nonlinearity')
            
            save_gradcam(filename=osp.join(
                '/xcli/outputs/RepVGG_gardcam', str(global_id))+'.jpg',
                gcam=regions[0, 0],
                raw_image=img_raw,
                image_size = (330, 330),
            )
            

            global_id += 1
