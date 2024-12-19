# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import os.path as op
import argparse
import datetime
import numpy as np
import time
import torch
import random

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import model.model_zoo
from dataset.datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RASampler
from contextlib import suppress

import utils

from utils.logger import setup_logger
from utils import utils
from utils.options import get_args_parser
from utils.iotools import save_train_configs

torch.cuda.empty_cache()
def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def main(args):
    utils.init_distributed_mode(args)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    generator = set_seed(args.seed+utils.get_rank())

    model_name = args.model
    device = torch.device(args.device) # 'cuda:3'
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.transfer_type == "prompt":
        args.output_dir = op.join(args.output_dir, args.transfer_type, args.prompt_type, args.data_set, f'{cur_time}_type_{args.prompt_type}_depth{args.prompt_depth}_model_{args.model}_lr_{args.lr}')
    else:
        args.output_dir = op.join(args.output_dir, args.transfer_type, args.data_set, f'{cur_time}_{model_name}_lr_{args.lr}')
    logger = setup_logger('SVP', save_dir=args.output_dir, if_train=not args.eval, distributed_rank=utils.get_rank(), if_no_logging=args.if_no_logging)

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    if utils.get_rank() == 0:
        save_train_configs(args.output_dir, args)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=generator)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        transfer_type=args.transfer_type,
        prompt_type=args.prompt_type,
        prompt_add_gen=args.prompt_add_gen,
        prompt_shared=args.prompt_shared,
        prompt_dropout=args.prompt_dropout,
        shared_layers=args.shared_layers,
        dataset=args.data_set,
        input_relu=args.input_relu,
    )
    logger.info(f"Created model: {args.model}")


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'head.0.weight', 'head.0.bias', 'head.1.weight', 'head.1.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        logger.info(f"Loaded pre-trained checkpoint from {args.finetune}")




    model.to(device)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    total_param_names = []
    grad_param_names = []
    for name, param in model.named_parameters():
        total_param_names.append(name)
        if param.requires_grad:
            grad_param_names.append(name)

    logger.info("grad_param_names:"+'\n'+'\n'.join(grad_param_names)+'\n')

    # lr缩放
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0 # linear_scaled_lr = 6.25e-5
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    
    # amp about
    amp_autocast = suppress
    # amp_autocast函数使用了suppress装饰器，目的是在运行时忽略可能产生的警告或异常，确保函数执行不因这些异常而中断。
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()   # 🌟🌟
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        


    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint and args.if_amp: # change loss_scaler if not amp
                loss_scaler.load_state_dict(checkpoint['scaler'])
            elif 'scaler' in checkpoint and not args.if_amp:
                loss_scaler = 'none'
        lr_scheduler.step(args.start_epoch)
        
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return


    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args,
        )


        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                    'args': args,
                }, checkpoint_path)
             

        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                        'args': args,
                    }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        logger.info(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info(f'Max accuracy: {max_accuracy:.2f}%')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('Vim training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
