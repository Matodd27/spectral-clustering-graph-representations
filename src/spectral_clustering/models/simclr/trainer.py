import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.utils import make_grid, save_image

from .simclr import ContrastiveTransformations, contrast_transforms, save_augmentation_preview, get_model, WarmupCosineScheduler, SimCLR, ntxent_loss

DATASET_PATH = os.path.expanduser('~/scratch/data')
root = os.path.expanduser('~/scratch/simclr_fashionmnist')
results_dir = os.path.join(root, 'SimCLR_results')
os.makedirs(results_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


batch_size = 512
num_workers = 8
num_epochs = 100
tau = 0.2
warmup_epochs = 10
losses_train = []

FMNIST_contrast = datasets.FashionMNIST(
    root=DATASET_PATH,
    train=True,
    download=True,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
)

train_loader = DataLoader(
    FMNIST_contrast,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
)


def main():
    global losses_train

    save_augmentation_preview(
        FMNIST_contrast,
        os.path.join(results_dir, 'aug_preview.png'),
    )

    model = get_model().to(device)

    base_lr = 0.3 * batch_size / 256
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    dataloaders = {'train': train_loader}

    simclrobj = SimCLR(
        model,
        optimizer,
        dataloaders,
        ntxent_loss,
        scheduler,
        scaler,
        results_dir,
    )
    losses_train = simclrobj.train(epochs=num_epochs)


if __name__ == '__main__':
    main()
