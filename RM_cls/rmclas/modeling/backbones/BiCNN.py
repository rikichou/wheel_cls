import os

import torch
import torchvision

import logging
from .build import BACKBONE_REGISTRY
from rmclas.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
logger = logging.getLogger(__name__)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


__all__ = ['BCNN', 'BCNNManager']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-09'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-01-13'
__version__ = '1.2'


class BCNN(torch.nn.Module):
    """B-CNN for CUB200.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        #self.fc = torch.nn.Linear(512**2, 200)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        #assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        #assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 7**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (7**2)  # Bilinear
        #assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = X.view(-1, 1)
        X = X.squeeze()
        #X = self.fc(X)
        #assert X.size() == (N, 200)
        print(X.size())
        return X


@BACKBONE_REGISTRY.register()
def build_BiCNN_backbone(cfg):
    """
    Create a BiCNN instance from config.
    Returns:
        BiCNN: a :class:`BiCNN` instance.
    """
    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    '''T             = cfg.MODEL.BACKBONE.T
    width_mult    = cfg.MODEL.BACKBONE.WIDTH_MULT'''
    # fmt: on

    model = BCNN()

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