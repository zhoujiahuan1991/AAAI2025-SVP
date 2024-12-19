# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

# from losses import DistillationLoss
import utils.utils as utils
import time


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30
        

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True) # [batch_size, 3, 224, 224]
        targets = targets.to(device, non_blocking=True) # [batch_size,]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            outputs, hidden_states = model(samples, if_random_token_rank=args.if_random_token_rank)
            loss = criterion(outputs, targets)

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)
                
        optimizer.zero_grad()


        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()


    for images, target in metric_logger.log_every(data_loader, 10, header):
        start = time.time()
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output, _ = model(images)
            loss = criterion(output, target)



        try:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()


        except:
            acc1, acc2 = accuracy(output, target, topk=(1, 2))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
