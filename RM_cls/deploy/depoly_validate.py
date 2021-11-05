import numpy as np
import sys

sys.path.append('.')
import time
import torch
from torch.autograd import Variable
from rmclas.config import get_cfg
from rmclas.engine import DefaultTrainer

caffe_root = '/xcli/codes/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')

import caffe

net_name = 'res_acnet'

if net_name == 'res_acnet':
    net_file = '/xcli/wheels_cls/RM_cls/deploy/caffe_models/wheel_res_acnet/res_acnet.prototxt'
    caffe_model = '/xcli/wheels_cls/RM_cls/deploy/caffe_models/wheel_res_acnet/res_acnet_display_0928.caffemodel'
    torch_model = '/xcli/wheels_cls/RM_cls/deploy/torch_model/acnet_display_0928_deployed.pth'
    config_file = '/xcli/wheels_cls/RM_cls/configs/wheel_acnet_ex6.yml'
else:
    raise ValueError('we are not support %s yet' % net_name)

size = (112, 112)

# net = caffe.Net(net_file, caffe_model, caffe.TEST)

def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.PIXEL_MEAN = [0, 0, 0]
    cfg.MODEL.BACKBONE.DEPLOY = True
    cfg.freeze()
    return cfg


def forward_pytorch(image, config):
    cfg = setup(config)
    net = DefaultTrainer.build_model(cfg)
    state_dict = torch.load(torch_model, map_location=torch.device('cpu'))['model']
    state_dict["pixel_mean"] = torch.tensor([[[[0]], [[0]], [[0]]]])

    # print(state_dict.keys())
    # print(net.state_dict().keys())
    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()
    image = torch.from_numpy(image)

    image = Variable(image.cuda())

    t0 = time.time()
    out = net(image)
    t1 = time.time()
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
    return t1-t0, net.blobs, net.params, output


if __name__ == '__main__':
    print('This is main ....')

    img = np.ones([1, 3, size[1], size[0]], dtype=np.float32)
    time_pytorch, out = forward_pytorch(img, config_file)
    time_caffe, caffe_blobs, caffe_params, caffe_out = forward_caffe(net_file, caffe_model, img)

    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)

    print('------------ Output Difference ------------')
    out_blob_name = "InnerProduct1"

    det_pytorch_data = out.data.cpu().numpy().flatten()

    det_caffe_data = caffe_blobs[out_blob_name].data[0][...].flatten()
    det_diff = abs(det_pytorch_data - det_caffe_data).sum()
    print(det_pytorch_data)
    print(det_caffe_data)
