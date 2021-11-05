import torch
import sys

caffe_root = '/xcli/codes/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append('./')

import caffe
import numpy as np

net_name = 'res_acnet'

if net_name == 'res_acnet':
    protofile = '/xcli/wheels_cls/RM_cls/deploy/caffe_models/wheel_res_acnet/res_acnet.prototxt'
    caffemodel = '/xcli/wheels_cls/RM_cls/deploy/caffe_models/wheel_res_acnet/res_acnet_display_1008.caffemodel'
    pytorch_pth = '/xcli/wheels_cls/RM_cls/deploy/torch_model/model_final_display_1008_deployed.pth'
else:
    raise ValueError('we are not support %s yet' % net_name)


pytorch_net = torch.load(pytorch_pth, map_location=torch.device("cpu"))['model']
caffe_net = caffe.Net(protofile, caffe.TEST)

caffe_params = caffe_net.params


def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean
    bn_param[1].data[...] = running_var
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases
    scale_param[0].data[...] = weights


def res_acnet_torch2caffe():
    for (name, weights) in pytorch_net.items():
        weights = weights.cpu().data.numpy()

        # backbone.layer1.block1.conv1.fused_conv.weight
        # backbone.layer1.block1.conv1.fused_conv.bias
        # backbone.layer1.block1.bn1.weight
        # backbone.layer1.block1.bn1.bias
        # backbone.layer1.block1.bn1.running_mean
        # backbone.layer1.block1.bn1.running_var
        # backbone.layer1.block1.bn1.num_batches_tracked
        # backbone.layer1.block1.relu.weight
        if name == 'backbone.layer1.block1.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution1']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer1.block1.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution1']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer1.block1.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer1.block1.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer1.block1.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer1.block1.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm1']
            scale_params = caffe_params['Scale1']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer1.block1.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU1']
            prelu_params[0].data[...] = weights

        # backbone.layer1.block2.conv1.fused_conv.weight
        # backbone.layer1.block2.conv1.fused_conv.bias
        # backbone.layer1.block2.bn1.weight
        # backbone.layer1.block2.bn1.bias
        # backbone.layer1.block2.bn1.running_mean
        # backbone.layer1.block2.bn1.running_var
        # backbone.layer1.block2.bn1.num_batches_tracked
        # backbone.layer1.block2.relu.weight
        if name == 'backbone.layer1.block2.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution2']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer1.block2.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution2']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer1.block2.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer1.block2.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer1.block2.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer1.block2.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm2']
            scale_params = caffe_params['Scale2']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer1.block2.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU2']
            prelu_params[0].data[...] = weights

        # backbone.layer1.block3.conv1.fused_conv.weight
        # backbone.layer1.block3.conv1.fused_conv.bias
        # backbone.layer1.block3.bn1.weight
        # backbone.layer1.block3.bn1.bias
        # backbone.layer1.block3.bn1.running_mean
        # backbone.layer1.block3.bn1.running_var
        # backbone.layer1.block3.bn1.num_batches_tracked
        # backbone.layer1.block3.relu.weight
        if name == 'backbone.layer1.block3.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution3']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer1.block3.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution3']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer1.block3.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer1.block3.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer1.block3.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer1.block3.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm3']
            scale_params = caffe_params['Scale3']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer1.block3.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU3']
            prelu_params[0].data[...] = weights

        # backbone.layer2.block1.conv1.fused_conv.weight
        # backbone.layer2.block1.conv1.fused_conv.bias
        # backbone.layer2.block1.bn1.weight
        # backbone.layer2.block1.bn1.bias
        # backbone.layer2.block1.bn1.running_mean
        # backbone.layer2.block1.bn1.running_var
        # backbone.layer2.block1.bn1.num_batches_tracked
        # backbone.layer2.block1.relu.weight
        if name == 'backbone.layer2.block1.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution4']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer2.block1.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution4']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer2.block1.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer2.block1.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer2.block1.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer2.block1.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm4']
            scale_params = caffe_params['Scale4']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer2.block1.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU4']
            prelu_params[0].data[...] = weights

        # backbone.layer2.block2.conv1.fused_conv.weight
        # backbone.layer2.block2.conv1.fused_conv.bias
        # backbone.layer2.block2.bn1.weight
        # backbone.layer2.block2.bn1.bias
        # backbone.layer2.block2.bn1.running_mean
        # backbone.layer2.block2.bn1.running_var
        # backbone.layer2.block2.bn1.num_batches_tracked
        # backbone.layer2.block2.relu.weight
        if name == 'backbone.layer2.block2.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution5']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer2.block2.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution5']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer2.block2.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer2.block2.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer2.block2.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer2.block2.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm5']
            scale_params = caffe_params['Scale5']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer2.block2.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU5']
            prelu_params[0].data[...] = weights

        # backbone.layer2.block3.conv1.fused_conv.weight
        # backbone.layer2.block3.conv1.fused_conv.bias
        # backbone.layer2.block3.bn1.weight
        # backbone.layer2.block3.bn1.bias
        # backbone.layer2.block3.bn1.running_mean
        # backbone.layer2.block3.bn1.running_var
        # backbone.layer2.block3.bn1.num_batches_tracked
        # backbone.layer2.block3.relu.weight
        if name == 'backbone.layer2.block3.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution6']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer2.block3.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution6']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer2.block3.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer2.block3.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer2.block3.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer2.block3.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm6']
            scale_params = caffe_params['Scale6']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer2.block3.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU6']
            prelu_params[0].data[...] = weights

        # backbone.layer4.block1.conv1.fused_conv.weight
        # backbone.layer4.block1.conv1.fused_conv.bias
        # backbone.layer4.block1.bn1.weight
        # backbone.layer4.block1.bn1.bias
        # backbone.layer4.block1.bn1.running_mean
        # backbone.layer4.block1.bn1.running_var
        # backbone.layer4.block1.bn1.num_batches_tracked
        # backbone.layer4.block1.relu.weight
        if name == 'backbone.layer4.block1.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution7']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer4.block1.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution7']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer4.block1.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer4.block1.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer4.block1.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer4.block1.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm7']
            scale_params = caffe_params['Scale7']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer4.block1.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU7']
            prelu_params[0].data[...] = weights

        # backbone.layer4.block2.conv1.fused_conv.weight
        # backbone.layer4.block2.conv1.fused_conv.bias
        # backbone.layer4.block2.bn1.weight
        # backbone.layer4.block2.bn1.bias
        # backbone.layer4.block2.bn1.running_mean
        # backbone.layer4.block2.bn1.running_var
        # backbone.layer4.block2.bn1.num_batches_tracked
        # backbone.layer4.block2.relu.weight
        if name == 'backbone.layer4.block2.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution8']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer4.block2.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution8']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer4.block2.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer4.block2.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer4.block2.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer4.block2.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm8']
            scale_params = caffe_params['Scale8']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer4.block2.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU8']
            prelu_params[0].data[...] = weights

        # backbone.layer4.block3.conv1.fused_conv.weight
        # backbone.layer4.block3.conv1.fused_conv.bias
        # backbone.layer4.block3.bn1.weight
        # backbone.layer4.block3.bn1.bias
        # backbone.layer4.block3.bn1.running_mean
        # backbone.layer4.block3.bn1.running_var
        # backbone.layer4.block3.bn1.num_batches_tracked
        # backbone.layer4.block3.relu.weight
        if name == 'backbone.layer4.block3.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution9']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer4.block3.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution9']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer4.block3.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer4.block3.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer4.block3.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer4.block3.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm9']
            scale_params = caffe_params['Scale9']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer4.block3.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU9']
            prelu_params[0].data[...] = weights

        # backbone.layer8.block1.conv1.fused_conv.weight
        # backbone.layer8.block1.conv1.fused_conv.bias
        # backbone.layer8.block1.bn1.weight
        # backbone.layer8.block1.bn1.bias
        # backbone.layer8.block1.bn1.running_mean
        # backbone.layer8.block1.bn1.running_var
        # backbone.layer8.block1.bn1.num_batches_tracked
        # backbone.layer8.block1.relu.weight
        if name == 'backbone.layer8.block1.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution10']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer8.block1.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution10']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer8.block1.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer8.block1.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer8.block1.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer8.block1.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm10']
            scale_params = caffe_params['Scale10']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer8.block1.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU10']
            prelu_params[0].data[...] = weights

        # backbone.layer8.block2.conv1.fused_conv.weight
        # backbone.layer8.block2.conv1.fused_conv.bias
        # backbone.layer8.block2.bn1.weight
        # backbone.layer8.block2.bn1.bias
        # backbone.layer8.block2.bn1.running_mean
        # backbone.layer8.block2.bn1.running_var
        # backbone.layer8.block2.bn1.num_batches_tracked
        # backbone.layer8.block2.relu.weight
        if name == 'backbone.layer8.block2.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution11']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer8.block2.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution11']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer8.block2.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer8.block2.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer8.block2.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer8.block2.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm11']
            scale_params = caffe_params['Scale11']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer8.block2.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU11']
            prelu_params[0].data[...] = weights

        # backbone.layer8.block3.conv1.fused_conv.weight
        # backbone.layer8.block3.conv1.fused_conv.bias
        # backbone.layer8.block3.bn1.weight
        # backbone.layer8.block3.bn1.bias
        # backbone.layer8.block3.bn1.running_mean
        # backbone.layer8.block3.bn1.running_var
        # backbone.layer8.block3.bn1.num_batches_tracked
        # backbone.layer8.block3.relu.weight
        if name == 'backbone.layer8.block3.conv1.fused_conv.weight':
            print(name)
            conv_param = caffe_params['Convolution12']
            conv_param[0].data[...] = weights
        if name == 'backbone.layer8.block3.conv1.fused_conv.bias':
            print(name)
            conv_param = caffe_params['Convolution12']
            conv_param[1].data[...] = weights
        if name == 'backbone.layer8.block3.bn1.weight':
            scale_weights = weights
            print(name)
        if name == 'backbone.layer8.block3.bn1.bias':
            scale_biases = weights
            print(name)
        if name == 'backbone.layer8.block3.bn1.running_mean':
            running_mean = weights
            print(name)
        if name == 'backbone.layer8.block3.bn1.running_var':
            running_var = weights
            print(name)

            bn_params = caffe_params['BatchNorm12']
            scale_params = caffe_params['Scale12']

            save_bn2caffe(running_mean, running_var, bn_params)
            save_scale2caffe(scale_weights, scale_biases, scale_params)
        if name == 'backbone.layer8.block3.relu.weight':
            print(name)
            prelu_params = caffe_params['PReLU12']
            prelu_params[0].data[...] = weights

        # heads.classifier.weight
        # heads.classifier.bias
        if name == 'heads.classifier.weight':
            print(name)
            InnerProduct_param = caffe_params['InnerProduct1']
            InnerProduct_param[0].data[...] = weights
        if name == 'heads.classifier.bias':
            print(name)
            InnerProduct_param = caffe_params['InnerProduct1']
            InnerProduct_param[1].data[...] = weights

    print('save caffemodel to %s' % caffemodel)
    caffe_net.save(caffemodel)


def checkpoint_travel():
    for (name, weights) in pytorch_net.items():
        print(name)


if __name__ == '__main__':
    print('This is main ...')
    # checkpoint_travel()
    if net_name == 'res_acnet':
        res_acnet_torch2caffe()
        # pass
    else:
        raise ValueError('we are not support %s yet' % net_name)
