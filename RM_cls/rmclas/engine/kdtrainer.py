# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : kdtrainer.py
#@time   : 2020-12-29 09:55:00
"""

import logging
import time

import torch
from torch import nn
import numpy as np

from .defaults import DefaultTrainer
from rmclas.utils.file_io import PathManager
from rmclas.modeling.meta_arch import build_model
from rmclas.utils.checkpoint import Checkpointer
from rmclas.config import update_model_teacher_config
from rmclas.distiller_zoo import *
from rmclas.layers import ConvReg, LinearEmbed, Connector, Translator, Paraphraser
from rmclas.utils.pretrain import init
from rmclas.utils.logger import setup_logger
from rmclas.utils import comm
from rmclas.solver.build import build_kd_optimizer
from rmclas.engine.train_loop import SimpleTrainer
from rmclas.utils.events import get_event_storage
from rmclas.modeling.losses import cross_entropy_loss


class KDTrainer(DefaultTrainer):
    """
    A knowledge distillation trainer for rmclas task.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self._hooks = []
        logger = logging.getLogger('rmclas.' + __name__)

        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes)
        model_s = self.build_model(cfg)
        model_t = self.build_model_teacher(cfg)
        for param in model_t.parameters():
            param.requires_grad = False

        # Load pre-trained teacher model
        logger.info("Loading teacher model ...")
        Checkpointer(model_t).load(cfg.MODEL.TEACHER_WEIGHTS)

        if PathManager.exists(cfg.MODEL.STUDENT_WEIGHTS):
            logger.info("Loading student model ...")
            Checkpointer(model_s).load(cfg.MODEL.STUDENT_WEIGHTS)
        else:
            logger.info("No student model checkpoints")

        # change student model and teacher model to eval
        data = torch.randn(2, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
        model_t.eval()
        model_s.eval()
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

        # build criterion
        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        # criterion_cls = nn.CrossEntropyLoss()
        if cfg.DISTILL.KL_CHOICE == 'kl':
            criterion_div = DistillKL(cfg.DISTILL.KD_T)
        elif cfg.DISTILL.KL_CHOICE == 'jsdiv':
            criterion_div = DistillJSDIV(cfg.DISTILL.KD_T)
        else:
            raise ValueError("we are not support {} yet".format(cfg.DISTILL.KL_CHOICE))
        if cfg.DISTILL.METHOD == 'kd':
            criterion_kd = DistillKL(cfg.DISTILL.KD_T)
        elif cfg.DISTILL.METHOD == 'hint':
            criterion_kd = HintLoss()
            regress_s = ConvReg(feat_s[cfg.DISTILL.HINT_LAYER].shape, feat_t[cfg.DISTILL.HINT_LAYER].shape)
            module_list.append(regress_s)
            trainable_list.append(regress_s)
        elif cfg.DISTILL.METHOD == 'attention':
            criterion_kd = Attention()
        elif cfg.DISTILL.METHOD == 'nst':
            criterion_kd = NSTLoss()
        elif cfg.DISTILL.METHOD == 'similarity':
            criterion_kd = Similarity()
        elif cfg.DISTILL.METHOD == 'rkd':
            criterion_kd = RKDLoss()
        elif cfg.DISTILL.METHOD == 'pkt':
            criterion_kd = PKT()
        elif cfg.DISTILL.METHOD == 'kdsvd':
            criterion_kd = KDSVD()
        elif cfg.DISTILL.METHOD == 'correlation':
            criterion_kd = Correlation()
            embed_s = LinearEmbed(feat_s[-1].shape[1], cfg.DISTILL.FEAT_DIM)
            embed_t = LinearEmbed(feat_t[-1].shape[1], cfg.DISTILL.FEAT_DIM)
            module_list.append(embed_s)
            module_list.append(embed_t)
            trainable_list.append(embed_s)
            trainable_list.append(embed_t)
        elif cfg.DISTILL.METHOD == 'vid':
            s_n = [f.shape[1] for f in feat_s[1:-1]]
            t_n = [f.shape[1] for f in feat_t[1:-1]]
            criterion_kd = nn.ModuleList(
                [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
            )
            # add this as some parameters in VIDLoss need to be updated
            trainable_list.append(criterion_kd)
        elif cfg.DISTILL.METHOD == 'abound':
            s_shapes = [f.shape for f in feat_s[1:-1]]
            t_shapes = [f.shape for f in feat_t[1:-1]]
            connector = Connector(s_shapes, t_shapes)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(connector)
            init_trainable_list.append(model_s.get_feat_modules())
            criterion_kd = ABLoss(len(feat_s[1:-1]))
            init(model_s, model_t, init_trainable_list, criterion_kd, data_loader, logger, cfg)
            # classification
            module_list.append(connector)
        elif cfg.DISTILL.METHOD == 'factor':
            s_shape = feat_s[-2].shape
            t_shape = feat_t[-2].shape
            paraphraser = Paraphraser(t_shape)
            translator = Translator(s_shape, t_shape)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(paraphraser)
            criterion_init = nn.MSELoss()
            init(model_s, model_t, init_trainable_list, criterion_init, data_loader, logger, cfg)
            # classification
            criterion_kd = FactorTransfer()
            module_list.append(translator)
            module_list.append(paraphraser)
            trainable_list.append(translator)
        elif cfg.DISTILL.METHOD == 'fsp':
            s_shapes = [s.shape for s in feat_s[:-1]]
            t_shapes = [t.shape for t in feat_t[:-1]]
            criterion_kd = FSP(s_shapes, t_shapes)
            # init stage training
            init_trainable_list = nn.ModuleList([])
            init_trainable_list.append(model_s.get_feat_modules())
            init(model_s, model_t, init_trainable_list, criterion_kd, data_loader, logger, cfg)
            # classification training
            pass
        else:
            raise NotImplementedError(cfg.DISTILL.METHOD)

        criterion_list = nn.ModuleList([])
        # criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss

        # optimizer
        optimizer = build_kd_optimizer(cfg, trainable_list.named_parameters())

        # append teacher after optimizer to avoid weight_decay
        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()

        optimizer_ckpt = dict(optimizer=optimizer)

        self._trainer = SimpleTrainer(
            model_s, data_loader, optimizer, cfg
        )

        self.iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
        self.scheduler = self.build_lr_scheduler(cfg, optimizer, self.iters_per_epoch)

        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = Checkpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model_s,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process(),
            **optimizer_ckpt,
            **self.scheduler,
        )
        self.start_epoch = 0

        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        self.max_iter = self.max_epoch * self.iters_per_epoch
        self.warmup_iters = cfg.SOLVER.WARMUP_ITERS
        self.delay_epochs = cfg.SOLVER.DELAY_EPOCHS
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

        #
        self.model_s = model_s
        self.model_t = model_t
        self.module_list = module_list
        self.criterion_list = criterion_list

        self._data_loader_iter = iter(data_loader)

        for module in self.module_list:
            module.train()
        # set teacher as eval()
        self.module_list[-1].eval()

    def run_step(self):

        assert self.model_s.training, "[KDTrainer] base model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        if self.cfg.DISTILL.METHOD == 'abound':
            self.module_list[1].eval()
        elif self.cfg.DISTILL.METHOD == 'factor':
            self.module_list[2].eval()

        # criterion_cls = self.criterion_list[0]
        criterion_div = self.criterion_list[0]
        criterion_kd = self.criterion_list[1]

        model_s = self.module_list[0]
        model_t = self.module_list[-1]

        target = data["targets"]

        if torch.cuda.is_available():
            target = target.cuda()
        # ===================forward=====================
        preact = False
        if self.cfg.DISTILL.METHOD in ['abound']:
            preact = True
        feat_s, logit_s = model_s(data, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(data, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div + other
        # loss_cls = criterion_cls(logit_s["cls_outputs"], target)
        loss_cls = cross_entropy_loss(
            logit_s["cls_outputs"],
            target,
            self.cfg.MODEL.LOSSES.CE.EPSILON,
            self.cfg.MODEL.LOSSES.CE.ALPHA,
        )
        loss_div = criterion_div(logit_s["pred_class_logits"], logit_t["pred_class_logits"])

        # other kd beyond KL divergence
        if self.cfg.DISTILL.METHOD == 'kd':
            loss_kd = 0
        elif self.cfg.DISTILL.METHOD == 'hint':
            f_s = self.module_list[1](feat_s[self.cfg.DISTILL.HINT_LAYER])
            f_t = feat_t[self.cfg.DISTILL.HINT_LAYER]
            loss_kd = criterion_kd(f_s, f_t)
        elif self.cfg.DISTILL.METHOD == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.cfg.DISTILL.METHOD == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.cfg.DISTILL.METHOD == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.cfg.DISTILL.METHOD == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif self.cfg.DISTILL.METHOD == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif self.cfg.DISTILL.METHOD == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif self.cfg.DISTILL.METHOD == 'correlation':
            f_s = self.module_list[1](feat_s[-1])
            f_t = self.module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif self.cfg.DISTILL.METHOD == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif self.cfg.DISTILL.METHOD == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif self.cfg.DISTILL.METHOD == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif self.cfg.DISTILL.METHOD == 'factor':
            factor_s = self.module_list[1](feat_s[-2])
            factor_t = self.module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(self.cfg.DISTILL)

        loss_cls = self.cfg.DISTILL.GAMMA*loss_cls
        loss_div = self.cfg.DISTILL.ALPHA*loss_div
        loss_kd = self.cfg.DISTILL.BETA*loss_kd

        loss_dict = {"loss_cls": loss_cls,
                     "loss_div": loss_div,
                     "loss_kd":  loss_kd}

        losses = sum(loss_dict.values())

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        with torch.cuda.stream(torch.cuda.Stream()):
            self._trainer._write_metrics(loss_dict, data_time)
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    @classmethod
    def build_model_teacher(cls, cfg) -> nn.Module:
        cfg_t = update_model_teacher_config(cfg)
        model_t = build_model(cfg_t)
        return model_t



