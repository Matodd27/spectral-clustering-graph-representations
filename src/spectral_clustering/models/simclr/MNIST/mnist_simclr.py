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


DATASET_PATH = os.path.expanduser('~/scratch/data')
root = os.path.expanduser('~/scratch/simclr_mnist')
results_dir = os.path.join(root, 'SimCLR_results')
os.makedirs(results_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


batch_size = 512
num_workers = 8
num_epochs = 1000
tau = 0.2
warmup_epochs = 10
losses_train = []


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


contrast_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.90, 1.05),
        fill=0,
    ),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.15, contrast=0.15)],
        p=0.5,
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


class SimCLRNet(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        backbone = resnet18(weights=None)
        if input_channels == 1:
            backbone.conv1 = nn.Conv2d(
                1,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        backbone.maxpool = nn.Identity()
        backbone_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def encode(self, x):
        return self.backbone(x)

    def forward(self, x):
        h = self.encode(x)
        z = self.projector(h)
        return z


def get_model(input_channels=1):
    return SimCLRNet(input_channels=input_channels)


def ntxent_loss(a, b, tau=0.2):
    z = torch.cat([a, b], dim=0)
    z = F.normalize(z, dim=1)

    logits = torch.matmul(z, z.T) / tau
    logits.fill_diagonal_(float('-inf'))

    batch_size_local = a.size(0)
    targets = torch.arange(2 * batch_size_local, device=z.device)
    targets = (targets + batch_size_local) % (2 * batch_size_local)

    return F.cross_entropy(logits, targets)

def encode_dataset(model_path):  
    model = get_model(input_channels=1).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    latents, labels = [], []

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root="/Users/matthewtodd/Uni work/Year 4/Dissertation/git-repo/spectral-clustering/notebooks/data",
        train=True,
        download=False,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=torch.cuda.is_available())
            h = model.encode(x)
            latents.append(h.cpu())
            labels.append(y.cpu())

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return latents, labels

class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs, warmup_epochs):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        self.last_epoch = -1

    def get_lr_scale(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(max(1, self.warmup_epochs))

        cosine_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = float(epoch - self.warmup_epochs) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def step(self, epoch):
        self.last_epoch = epoch
        scale = self.get_lr_scale(epoch)
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group['lr'] = base_lr * scale

    def state_dict(self):
        return {
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.total_epochs = state_dict.get('total_epochs', self.total_epochs)
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.last_epoch = state_dict.get('last_epoch', self.last_epoch)


def save_augmentation_preview(dataset, preview_path, num_samples=6):
    preview_batches = []
    max_samples = min(num_samples, len(dataset))
    for idx in range(max_samples):
        views, _ = dataset[idx]
        preview_batches.extend([views[0], views[1]])

    if not preview_batches:
        return

    grid = make_grid(
        torch.stack(preview_batches),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, preview_path)


class SimCLR:
    def __init__(self, model, optimiser, dataloaders, loss_fn, scheduler, scaler, results_dir):
        self.model = model
        self.optimiser = optimiser
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.scaler = scaler
        self.results_dir = results_dir
        self.use_amp = torch.cuda.is_available()
        self.losses_train = []
        self.start_epoch = 0

    def save_checkpoint(self, epoch, best=False):
        np.savez(
            os.path.join(self.results_dir, 'losses_train.npz'),
            losses_train=np.array(self.losses_train, dtype=np.float32),
        )
        torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'model.pth'))
        torch.save(self.optimiser.state_dict(), os.path.join(self.results_dir, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(self.results_dir, 'scheduler.pth'))

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "losses_train": [float(x) for x in self.losses_train],
        }

        if self.use_amp:
            scaler_state = self.scaler.state_dict()
            torch.save(scaler_state, os.path.join(self.results_dir, 'scaler.pth'))
            checkpoint['scaler_state_dict'] = scaler_state

        torch.save(checkpoint, os.path.join(self.results_dir, 'checkpoint_last.pth'))
        if best:
            torch.save(checkpoint, os.path.join(self.results_dir, 'checkpoint_best.pth'))
        if epoch%100 == 0:
            torch.save(checkpoint, os.path.join(self.results_dir, f'checkpoint_{epoch}.pth'))

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.results_dir, 'checkpoint_last.pth')
        if not os.path.isfile(checkpoint_path):
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        scaler_state = checkpoint.get('scaler_state_dict')
        if scaler_state is not None and self.use_amp:
            self.scaler.load_state_dict(scaler_state)
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.losses_train = checkpoint.get("losses_train", [])

        print(f"Resuming from epoch {self.start_epoch}.")

    def train(self, epochs=100):
        self.load_checkpoint()
        if self.start_epoch >= epochs:
            print(f"Training already completed through epoch {self.start_epoch - 1}.")
            return self.losses_train

        best_loss = min(self.losses_train) if self.losses_train else float('inf')

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            epoch_losses_train = []
            self.scheduler.step(epoch)

            for imgs, _ in self.dataloaders['train']:
                x1 = imgs[0].to(device, non_blocking=torch.cuda.is_available())
                x2 = imgs[1].to(device, non_blocking=torch.cuda.is_available())

                self.optimiser.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    z1 = self.model(x1)
                    z2 = self.model(x2)
                    loss = self.loss_fn(z1, z2, tau=tau)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

                epoch_losses_train.append(loss.detach().item())

            epoch_loss = float(np.mean(epoch_losses_train))
            self.losses_train.append(epoch_loss)

            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss

            self.save_checkpoint(epoch, best=is_best)
            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"train_loss: {epoch_loss:.4f} | "
                f"lr: {self.optimiser.param_groups[0]['lr']:.6f}"
            )

        return self.losses_train


MNIST_contrast = datasets.MNIST(
    root=DATASET_PATH,
    train=True,
    download=True,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
)

train_loader = DataLoader(
    MNIST_contrast,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
)


def main():
    global losses_train

    save_augmentation_preview(
        MNIST_contrast,
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
    print(device)
    losses_train = simclrobj.train(epochs=num_epochs)


if __name__ == '__main__':
    main()
