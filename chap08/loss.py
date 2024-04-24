import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn

# gather image and text features from all gpus
def gather_features(image_features, text_features, rank=0, world_size=1):
    gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
    gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)
    gathered_image_features[rank] = image_features
    gathered_text_features[rank] = text_features
    
    all_image_features = torch.cat(gathered_image_features, dim=0)
    all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(self, rank=0, world_size=1):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(image_features, text_features, self.rank, self.world_size)
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss