import torch
import argparse
import math
import os
import numpy as np
import random
import time
from contextlib import nullcontext

from model import CLIP, build_optimizer, get_lr
from dataset import build_dataloader
from loss import ClipLoss

from config import get_config
import utils


def parse_args():
    parser = argparse.ArgumentParser('Simple CLIP training script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    # easy config modification
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', type=str, help='root of output folder, ''the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    # distributed training
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    # data dir
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--caption_file', default='', type=str)
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()
    return args



def train(cfg, args):
    master_process = args.rank == 0
    model = CLIP(cfg)
    model = model.to(args.device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    raw_model = model.module if args.distributed else model
    if master_process:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"total params: {n_parameters:.2f}M, {n_parameters / 1e3:.2f}B")
        
    # build optimizer
    optimizer = build_optimizer(cfg.train, model)
    
    # setup for half precision training
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True   
    if torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
        dtype = 'bfloat16'
    else:
        ptdtype = torch.float16
        dtype = 'float16'
    ctx = nullcontext() if not torch.cuda.is_available() else torch.cuda.amp.autocast(True, dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # dataloader
    if args.distributed:
        dataloader = build_dataloader(cfg, local_rank=args.local_rank, world_size=args.world_size)
    else:
        dataloader = build_dataloader(cfg, local_rank=0, world_size=1)
        
    
    # training
    start = time.time()
    steps = 0
    for epoch in range(0, cfg.train.epochs):
        model.train()
        loss = ClipLoss(rank=args.rank, world_size=args.world_size)

        loss_m = utils.AverageMeter()
        end = time.time()
        
        for i, (images, captions) in enumerate(dataloader):
            """
            images: [B, C, H, W]
            captions is a list of tokens and masks.
            captions[0]: [B, 1, L]
            captions[1]: [B, 1, L]
            """
            images = images.to(device=args.device, non_blocking=True)
            texts = captions[0].squeeze(1)
            texts = texts.to(device=args.device, non_blocking=True)
            
            # update learning rate
            lr = get_lr(steps, cfg)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # forward
            with ctx:
                image_features, text_features, logit_scale = model(images, texts, normalize=True)
                total_loss = loss(image_features, text_features, logit_scale)

            # backward
            scaler.scale(total_loss).backward()
            
            # clip gradients
            if cfg.train.grad_clip_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm, norm_type=2.0)

            # optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # clamp to 4.6052 = ln(100), as in the original CLIP paper.
            with torch.no_grad():
                raw_model.logit_scale.clamp_(0, math.log(100))

            steps += 1
            
            # log
            if master_process and i % 100 == 0:
                batch_size = len(images)
                loss_m.update(total_loss.item(), batch_size)
                end = time.time()
                print(
                    f"Steps: {steps}\t"
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g})\t"
                    f"LR: {optimizer.param_groups[0]['lr']:5f}\t"
                    f"Time: {end - start:.3f}")

            # save checkpoint
            if master_process and steps % cfg.checkpoint.save_freq_steps == 0:
                checkpoint_dict = {
                    "steps": steps,
                    "state_dict": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),}
                if scaler is not None:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                torch.save(checkpoint_dict, os.path.join('./trained_model', "epoch_latest.pt"))
                print("Saved checkpoint at step", steps)
            

def main():
    # setup configs
    args = parse_args()
    cfg = get_config(args)

    # setup for distributed training
    utils.init_distributed_mode(args)

    # seed
    if args.distributed:
        seed = cfg.seed + utils.get_rank()
    else:
        seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # train clip
    train(cfg, args)
    

if __name__ == '__main__':
    main()