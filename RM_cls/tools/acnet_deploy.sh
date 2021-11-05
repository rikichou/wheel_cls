#!/usr/bin/env bash

CONFIG=$1
WEIGHT=$2

cd /xcli/wheels_cls/RM_cls

# torch model deploy
python deploy/acnet2net.py --config-file $CONFIG MODEL.WEIGHTS $WEIGHT

# caffe model deploy
# python deploy/prototxt_gen.py
# python deploy/prototxt_2_caffe.py
