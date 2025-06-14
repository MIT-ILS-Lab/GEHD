# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
from bisect import bisect_right
import torch.optim.lr_scheduler as LR

def multi_step(optimizer, config):
    return LR.MultiStepLR(optimizer, config["milestones"], config["gamma"])

def cos(optimizer, config):
    return LR.CosineAnnealingLR(
        optimizer, config["max_epoch"], eta_min=config["lr_min"]
    )

def poly(optimizer, config):
    lr_lambda = lambda epoch: (1 - epoch / config["max_epoch"]) ** config["lr_power"]
    return LR.LambdaLR(optimizer, lr_lambda)

def constant(optimizer, config):
    lr_lambda = lambda epoch: 1
    return LR.LambdaLR(optimizer, lr_lambda)

def cos_warmup(optimizer, config):
    def lr_lambda(epoch):
        warmup = config["warmup_epoch"]
        warmup_init = config["warmup_init"]
        if epoch <= warmup:
            return (1 - warmup_init) * epoch / warmup + warmup_init
        else:
            lr_min = config["lr_min"]
            ratio = (epoch - warmup) / (config["max_epoch"] - warmup)
            return lr_min + 0.5 * (1.0 - lr_min) * (1 + math.cos(math.pi * ratio))

    return LR.LambdaLR(optimizer, lr_lambda)

def poly_warmup(optimizer, config):
    def lr_lambda(epoch):
        warmup = config["warmup_epoch"]
        warmup_init = config["warmup_init"]
        if epoch <= warmup:
            return (1 - warmup_init) * epoch / warmup + warmup_init
        else:
            ratio = (epoch - warmup) / (config["max_epoch"] - warmup)
            return (1 - ratio) ** config["lr_power"]

    return LR.LambdaLR(optimizer, lr_lambda)

def step_warmup(optimizer, config):
    def lr_lambda(epoch):
        warmup = config["warmup_epoch"]
        warmup_init = config["warmup_init"]
        if epoch <= warmup:
            return (1 - warmup_init) * epoch / warmup + warmup_init
        else:
            milestones = sorted(config["milestones"])
            return config["gamma"] ** bisect_right(milestones, epoch)

    return LR.LambdaLR(optimizer, lr_lambda)

def knee_schedule(optimizer, config):
    warmup_steps = config["warmup_steps"]
    num_training_steps = config["max_epoch"]
    constant_pct = 0.50

    constant_steps = int(constant_pct * num_training_steps) - warmup_steps
    anneal_steps = num_training_steps - warmup_steps - constant_steps

    warmup_scheduler = LR.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
    constant_scheduler = LR.ConstantLR(optimizer, factor=1.0, total_iters=constant_steps)
    anneal_scheduler = LR.LinearLR(optimizer, start_factor=0.999, end_factor=0.0, total_iters=anneal_steps)

    return LR.SequentialLR(optimizer,
                           schedulers=[warmup_scheduler, constant_scheduler, anneal_scheduler],
                           milestones=[warmup_steps, warmup_steps + constant_steps])

def get_lr_scheduler(optimizer, config):
    lr_dict = {
        "cos": cos,
        "step": multi_step,
        "poly": poly,
        "constant": constant,
        "cos_warmup": cos_warmup,
        "poly_warmup": poly_warmup,
        "step_warmup": step_warmup,
        "knee": knee_schedule
    }
    lr_func = lr_dict[config["lr_type"]]
    return lr_func(optimizer, config)
