from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = 'ClassLevelBaseline'
_C.MODEL.FREEZE_LAYERS = ['']

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = ""

# Backbone generic
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = False
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048

# Backbone specific
# Used for mobilenetv2
_C.MODEL.BACKBONE.T = 6  # expansion ratio
_C.MODEL.BACKBONE.WIDTH_MULT = 1  # channels multiplier
# Used for resnetV2
_C.MODEL.BACKBONE.BLOCK_NAME = "Bottleneck"
_C.MODEL.BACKBONE.BLOCKS = [3, 4, 6, 3]
# Used for ShuffleNetv2
_C.MODEL.BACKBONE.NETSIZE = 1  # choice [0.2, 0.3, 0.5, 1, 1.5, 2]
# Used for vgg
_C.MODEL.BACKBONE.CONFIGURATION = 'A'  # choice ['A', 'B', 'D', 'E', 'S']
_C.MODEL.BACKBONE.BATCHNORM = True
# Used for wrn
_C.MODEL.BACKBONE.DEPTH = 16  # choice [16, 40]
_C.MODEL.BACKBONE.WIDEN_FACTOR = 1  # channels multiplier
# Used for ResAcNet
_C.MODEL.BACKBONE.DEPLOY = False  # switch of deployment

# ---------------------------------------------------------------------------- #
# HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "ClsHead"
# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 2
# Embedding dimension in head
_C.MODEL.HEADS.EMBEDDING_DIM = 0
# If use BNneck in embedding
_C.MODEL.HEADS.WITH_BNNECK = True
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128

# Used for distill
_C.MODEL.STUDENT_WEIGHTS = ""
_C.MODEL.TEACHER_WEIGHTS = ""
_C.MODEL.MODEL_TEACHER = ""

# ---------------------------------------------------------------------------- #
# LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = True
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.GAMMA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Cosface Loss options
_C.MODEL.LOSSES.COSFACE = CN()
_C.MODEL.LOSSES.COSFACE.MARGIN = 0.25
_C.MODEL.LOSSES.COSFACE.GAMMA = 128
_C.MODEL.LOSSES.COSFACE.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.DO_VERTICAL_FLIP = False
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10

# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
_C.INPUT.AUTOAUG_PROB = 0.0

# Augmix augmentation
_C.INPUT.DO_AUGMIX = False
_C.INPUT.AUGMIX_PROB = 0.0

_C.INPUT.CUTMIX = False
_C.INPUT.MIXUP = False
_C.INPUT.FMIX = False
_C.INPUT.MIX_ALPHA = 0.2

#Random rotation 
_C.INPUT.ROT = CN()
_C.INPUT.ROT.ENABLED = False
_C.INPUT.ROT.DEGREES = 90
_C.INPUT.ROT.EXPAND = False

# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.485*255, 0.456*255, 0.406*255]
# Random Patch
_C.INPUT.RPT = CN()
_C.INPUT.RPT.ENABLED = False
_C.INPUT.RPT.PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root of Datasets
_C.DATASETS.ROOT = "/lmliu/lmliu/dataset"
# List of the dataset names for training
_C.DATASETS.NAMES = (("DirReader", "Alive_detect"),)   # choice ["DirReader", TxTReader]
# List of the dataset names for testing
_C.DATASETS.TESTS = (("DirReader", "Alive_detect"),)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False
# List of the dataset Class
_C.DATASETS.CLASSES = ['positive', 'negative']

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 32
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.FP16_ENABLED = False

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_EPOCH = 120

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# Multi-step learning rate options
_C.SOLVER.SCHED = "MultiStepLR"

_C.SOLVER.DELAY_EPOCHS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.FREEZE_ITERS = 0

# SWA options
# _C.SOLVER.SWA = CN()
# _C.SOLVER.SWA.ENABLED = False
# _C.SOLVER.SWA.ITER = 10
# _C.SOLVER.SWA.PERIOD = 2
# _C.SOLVER.SWA.LR_FACTOR = 10.
# _C.SOLVER.SWA.ETA_MIN_LR = 3.5e-6
# _C.SOLVER.SWA.LR_SCHED = False

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

_C.TEST.EVAL_PERIOD = 20

# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 64
# _C.TEST.METRIC = "cosine"
# _C.TEST.ROC_ENABLED = False

# Precise batchnorm
_C.TEST.PRECISE_BN = CN()
_C.TEST.PRECISE_BN.ENABLED = False
_C.TEST.PRECISE_BN.DATASET = 'luggage'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False

#Use Gardcam to check the distribution of weights during the model training
_C.GC = CN()

_C.GC.IMAGE_DIR = '/xcli/wheels_cls/wheel_imgs/wheel_imgs_1/test'
_C.GC.DICT_DIR = '/xcli/weights/ClsWheel_ACNet.pth'
_C.GC.OUTPUT_DIR = '/xcli/outputs/gradcam'
_C.GC.IMAGE_SIZE = (300, 300)
_C.GC.TARGET_LAYER = 'backbone.layer8.block3.relu'