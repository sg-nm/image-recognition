import math
import numpy as np

import torch
import torch.nn as nn
from torch import optim as optim
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformer import VisionTransformer, TextTransformer

class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        
        vision_heads = config.img_encoder.width // config.img_encoder.head_width
        self.visual = VisionTransformer(
            image_size=config.img_encoder.image_size,
            patch_size=config.img_encoder.patch_size,
            width=config.img_encoder.width,
            layers=config.img_encoder.layers,
            heads=vision_heads,
            mlp_ratio=config.img_encoder.mlp_ratio,
            ls_init_value=config.img_encoder.ls_init_value,
            output_dim=config.embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            gap=config.img_encoder.gap,)
        
        self.text = TextTransformer(
            context_length=config.text_encoder.context_length,
            vocab_size=config.text_encoder.vocab_size,
            width=config.text_encoder.width,
            heads=config.text_encoder.heads,
            layers=config.text_encoder.layers,
            ls_init_value=config.text_encoder.ls_init_value,
            output_dim=config.embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
       
    def encode_image(self, image, normalize=False, projector=True):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text, normalize=True):
        image_features = self.encode_image(image, normalize=normalize)
        text_features  = self.encode_text(text, normalize=normalize)
        return image_features, text_features, self.logit_scale.exp()



def build_optimizer(config, model):
    """Build optimizer, set weight decay of normalization to 0 by default."""
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if config.optimizer.name == 'adamw':
        optimizer = optim.AdamW([
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": config.weight_decay},],
                lr=config.base_lr, betas=config.optimizer.betas, eps=config.optimizer.eps)
    elif config.optimizer.name == 'adamw_zero':
        optimizer = ZeroRedundancyOptimizer([
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": config.weight_decay},],
                    optimizer_class=torch.optim.AdamW,
                    eps=config.optimizer.eps,
                    betas=config.optimizer.betas, lr=config.base_lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer.name}')
    return optimizer


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, cfg):
    # warmup
    if it < cfg.train.warmup_steps:
        return cfg.train.base_lr * it / cfg.train.warmup_steps
    # if > lr_decay_iters, return min learning rate
    if it > cfg.train.lr_decay_iters:
        return cfg.train.min_lr
    # cosine decay down to min learning rate
    decay_ratio = (it - cfg.train.warmup_steps) / (cfg.train.lr_decay_iters - cfg.train.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.train.min_lr + coeff * (cfg.train.base_lr - cfg.train.min_lr)


def load_checkpoint(args, model, optimizer=None, scaler=None):
    print(f'====> Resuming from {args.resume}..........')
    state_dict = torch.load(args.resume, map_location='cpu')
    msg = model.load_state_dict(state_dict['state_dict'], strict=False)
    print(msg)
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(state_dict['scaler'])
    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
        steps = None
    elif 'steps' in state_dict:
        steps = state_dict['steps']
        epoch = None
    torch.cuda.empty_cache()
    return steps, epoch
