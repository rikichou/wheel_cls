# encoding: utf-8
"""
@author:  lmliu
@contact: lmliu@streamax.com
"""

from .build import build_backbone, BACKBONE_REGISTRY

from .resnetv2 import build_resnetV2_backbone
from .mobilenetv2 import build_mobilenetV2_backbone
from .ShuffleNetv1 import build_shufflenetV1_backbone
from .ShuffleNetv2 import build_shufflenetV2_backbone
from .vgg import build_vgg_backbone
from .wrn import build_wrn_backbone
from .ResACNet import build_ResAcNet_backbone
from .ResACNet_wheels import build_ResAcNet_wheels_backbone
from .RexNet import build_ReXNetV1_backbone
from .ResACNet_Silu import build_ResAcNet_Silu_backbone
from .Repvgg import *
from .SEResNext import build_SEResNext_backbone
from .SEResNextv2 import build_SEResNextV2_backbone
from .common import *
from .resnet import *
from .resnext import *
from .BiCNN import build_BiCNN_backbone