# Rm_Clas

## Getting Start

### Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.6
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`
- gdown (for automatically downloading pre-train model)
- sklearn
- termcolor
- tabulate
- [faiss](https://github.com/facebookresearch/faiss) `pip install faiss-cpu`

### prepare dataset
* txt读取方式
数据集准备如下：  
root  
│         
│
└───Dataset1  
│   │train.txt      
│   │test.txt  
│   │val.txt(optional)      
│   
└───Dataset2  
    │train.txt      
    │test.txt  
    │val.txt(optional)   

txt格式：  
imgs/1.jpg:猫  注：imgs/1.jpg是Dataset下的路径  
imgs/2.jpg:狗 
* dir读取方式
数据集准备如下：  
root  
│         
│
└───Dataset1  
│   │      
│   └───train  
│   │   └───猫     
│   │   └───狗      
│   │      
│   └───test  
│   │      
│   └───val(optional)   
│   
└───Dataset2  
    │      
    └───train   
    │   └───猫     
    │   └───鸭   
    │      
    └───test  
    │      
    └───val(optional)   
### prepare your config file
* 参考./configs/baseline.yml和./configs/kd_baseline.yml设计自己的config文件

### train starting
```bash
python tools/train_net.py --config-file configs/baseline.yml  
python tools/train_net.py --config-file configs/baseline.yml --num-gpus 4  
```
### eval only
* 修改config文件中MODEL.WEIGHTS和MODEL.DEVICE
* 运行
```bash
python tools/train_net.py --config-file configs/baseline.yml --eval-only
```
or  
```bash
python tools/train_net.py --config-file configs/baseline.yml --eval-only \
MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"
```
### distill
```bash
python tools/kd_train_net.py --config-file configs/kd_baseline.yml
python tools/kd_train_net.py --config-file configs/kd_baseline.yml --num-gpus 4 
```

## Features
* 多数据集整合训练
* cutmix, mixup, fmix, autoaugment, random erasing
* triplet loss, cosface loss, circle loss, curriculum loss(已经替换arcface loss)
* 12 state-of-the-art knowledge distillation methods + jsdiv(paddle clas)
* 分布式sampler,包含类别平衡sampler(PK_SAMPLER)
* 优化器集成adam, ranger, sgd, lamb, swa
* 模块划分严格，可扩展性强
* 易用性强，常规训练只需要修改config文件即可

## Structure
### rmclas::config

### rmclas::data::datasets
#### rmclas::data::samplers
1. distributed triplet sampler
2. distributed train sampler

#### rmclas::data::transforms
1. autoaugment
2. batch operators: cutmix, mixup, fmix
3. other transform

### rmclas::engine
1. build train loop
2. creat hooks

### rmclas::evaluation
1. evaluation code

### rmclas::modeling
1. build backbone
2. build head
3. build loss

### rmclas::solver
1. build solver

## TODO
1. FP16 bug修复