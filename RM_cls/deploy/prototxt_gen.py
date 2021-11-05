from __future__ import print_function
import sys

caffe_root = '/xcli/codes/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')

import os
import caffe
from caffe import to_proto
from deploy.modules.modules import Bottleneck_a
from deploy.modules.layers import *


def caffe_resacnet(lmdb, batch_size=1):

    data, label = L.ImageData(source=lmdb, batch_size=batch_size, shuffle=True, ntop=2, transform_param=dict(crop_size=112, mirror=False, scale=0.0078125, mean_value=127.5), include=dict(phase=caffe.TRAIN))

    x = Bottleneck_a("bottleneck1", data, 16, 3, [2, 1], 1)
    x = Bottleneck_a("bottleneck2", x, 32, 3, [2, 1], 1)
    x = Bottleneck_a("bottleneck4", x, 64, 3, [2, 1], 1)
    x = Bottleneck_a("bottleneck8", x, 64, 3, [2, 1], 1)
    x = L.Pooling(x, pool=P.Pooling.AVE, kernel_size=7, name="pool")

    x = L.InnerProduct(x, num_output=2)
    x = L.Softmax(x)
    return to_proto(x)

def make_net(net_name):
    path = "/xcli/wheels_cls/RM_cls/deploy/caffe_models/"
    prototxt = os.path.join(path, net_name, "{}.prototxt".format(net_name))
    os.makedirs(os.path.join(path, net_name), exist_ok=True)
    if net_name == 'res_acnet':
        with open(prototxt, 'w') as f:
            print(caffe_resacnet('/path/to/caffe-train-lmdb'), file=f)
    else:
        raise ValueError('we are not support %s yet' % net_name)


if __name__ == '__main__':
    net_name = 'res_acnet'
    make_net(net_name)

