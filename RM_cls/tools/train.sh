#!/usr/bin/env bash

CONFIG=$1

cd /xcli/wheels_cls/RM_cls/


python tools/train_net.py --config-file $CONFIG
