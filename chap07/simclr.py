import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl

class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder f()
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)
        
        # Projector g()
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    # set up the optimizer
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]

    # InfoNCE loss
    def info_nce_loss(self, batch, mode='train'):
        """
        batch[0] is a list of 2 augmented images.
        i.e., batch[0] = [img1, img2] and img1(img2) = [B, C, H, W]
        """
        imgs, _ = batch
        # # save the original image for visualization
        # torchvision.utils.save_image(imgs[0], './original1.png')
        # torchvision.utils.save_image(imgs[1], './original2.png')
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out the cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        
        # positive samples are located at (batch_size) away from the main diagonal
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = loss.mean()

        # Logging loss
        self.log(mode+'_loss', loss)
        comb_sim = torch.cat([cos_sim[pos_mask][:,None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
        self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
        return loss

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode='val')