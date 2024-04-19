"""
Reference:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
"""

import os
import torch
import torch.utils.data as data
from torchvision.datasets import STL10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data_aug import ContrastiveTransformations
from simclr import SimCLR

data_path = "./data"
save_path = "./saved_models"
os.makedirs(save_path, exist_ok=True)

# for reproducibility
seed = 0
pl.seed_everything(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Define the transformations for the contrastive learning
contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([
                                          transforms.ColorJitter(brightness=0.5,
                                                                 contrast=0.5,
                                                                 saturation=0.5,
                                                                 hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])

# Load the STL-10 dataset
unlabeled_data = STL10(root=data_path, split='unlabeled', download=True,
                       transform=ContrastiveTransformations(contrast_transforms, n_views=2))
train_data_contrast = STL10(root=data_path, split='train', download=True, 
                            transform=ContrastiveTransformations(contrast_transforms, n_views=2))

# train SimCLR model
max_epochs = 20
batch_size = 512
hidden_dim = 128
lr = 5e-4
weight_decay = 1e-4
temperature = 0.07
NUM_WORKERS = 4
trainer = pl.Trainer(default_root_dir=os.path.join(save_path, 'SimCLR'),
                        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                        devices=1, max_epochs=max_epochs,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                LearningRateMonitor('epoch')])

trainer.logger._default_hp_metric = None
# Check whether pretrained model exists. If yes, load it and skip training
pretrained_filename = os.path.join(save_path, 'SimCLR.ckpt')
if os.path.isfile(pretrained_filename):
    print(f'Load pretrained model at {pretrained_filename}')
    model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
else:
    train_loader = data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = data.DataLoader(train_data_contrast, batch_size=batch_size, shuffle=False,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    model = SimCLR(max_epochs=max_epochs, hidden_dim=hidden_dim, lr=lr, temperature=temperature, weight_decay=weight_decay)
    trainer.fit(model, train_loader, val_loader)
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training