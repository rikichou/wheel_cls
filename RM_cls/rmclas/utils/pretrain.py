# !/user/bin/env python
# coding=utf-8
"""
@project : Rm_Clas
@author  : lmliu
@contact : lmliu@streamax.com
#@file   : pretrain.py
#@time   : 2020-12-30 10:47:20
"""

from __future__ import print_function, division

import time
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.SOLVER.BASE_LR
    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=opt.SOLVER.MOMENTUM,
                          weight_decay=opt.SOLVER.WEIGHT_DECAY_BIAS)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, opt.DISTILL.INIT_EPOCHS + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):

            data_time.update(time.time() - end)

            # ============= forward ==============
            preact = (opt.DISTILL.METHOD == 'abound')
            feat_s, _ = model_s(data, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(data, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.DISTILL.METHOD == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.DISTILL.METHOD == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.DISTILL.METHOD == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.DISTILL.METHOD))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        logger.log_value('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, opt.DISTILL.INIT_EPOCHS, batch_time=batch_time, losses=losses))
        sys.stdout.flush()
