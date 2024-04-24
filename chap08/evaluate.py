import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

import utils
from model import CLIP, load_checkpoint
from dataset import ImageNet1K
from tokenizer import tokenize
from config import get_config
from imagenet_classes import imagenet_classnames, openai_imagenet_template

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluate(model, dataloader, ctx, cfg, args):
    top1 = 0.0
    top5 = 0.0
    if not utils.is_main_process():
        return top1, top5
    
    model.eval()
    top1, top5 = zero_shot_eval(model, dataloader, ctx, cfg, args)
    return top1, top5

def zero_shot_eval(model, dataloader, ctx, cfg, args):
    print('Starting zero-shot evaluation...')
    
    classifier = zero_shot_classifier(model, ctx, args)
    
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=cfg.data.batch_size):
            images = images.to(args.device)
            target = target.to(args.device)

            with ctx:
                image_features = model.encode_image(images, normalize=True)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)

    print('Finished zero-shot imagenet.')
    print(f"top1: {top1:.4f}, top5: {top5:.4f}")
    return top1, top5


def zero_shot_classifier(model, ctx, args):
    tokenizer = tokenize
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(imagenet_classnames):
            texts = [template(classname) for template in openai_imagenet_template]  # format with class
            texts, _ = tokenizer(texts)
            texts = texts.to(args.device)
            with ctx:
                texts = texts.to(device=args.device)
                class_embeddings = model.encode_text(texts, normalize=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights



if __name__ == '__main__':
    
    def parse_args():
        parser = argparse.ArgumentParser('Simple CLIP evaluation script')
        parser.add_argument('--cfg', type=str, required=True, help='path to config file')
        parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
        parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
        parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
        parser.add_argument('--local_rank', default=0, type=int, help='local rank for DistributedDataParallel')
        parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
        parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
        parser.add_argument('--resume', help='resume from checkpoint')

        args = parser.parse_args()
        return args
    
    
    # setup configs
    args = parse_args()
    cfg = get_config(args)
    
    # setup for distributed training
    utils.init_distributed_mode(args)

    # seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True   
    if torch.cuda.is_bf16_supported():
        ptdtype = torch.bfloat16
        dtype = 'bfloat16'
    else:
        ptdtype = torch.float16
        dtype = 'float16'
        
    ctx = nullcontext() if not torch.cuda.is_available() else torch.cuda.amp.autocast(True, dtype=ptdtype)
    
    # model
    model = CLIP(cfg)
    model = model.to(args.device)
    steps, epoch = load_checkpoint(args, model)
    
    # datasets
    val_dataset = ImageNet1K(cfg)
    print(f"val dataset size: {len(val_dataset)}")
    eval_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, 
                                            shuffle=False, num_workers=cfg.data.num_workers, 
                                            pin_memory=True, sampler=None)
    
    # evaluate
    evaluate(model, eval_dataloader, ctx, cfg, args)