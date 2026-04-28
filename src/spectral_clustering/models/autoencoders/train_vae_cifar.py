from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .vae_cifar import (
    CIFARVAETrainer,
    ConvCIFARVAE,
    DEFAULT_DATASET_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_ROOT,
    get_fixed_preview_batch,
    make_dataloader_kwargs,
    resolve_device,
)


DATASET_PATH = Path(os.path.expanduser(str(DEFAULT_DATASET_PATH)))
ROOT = Path(os.path.expanduser(str(DEFAULT_ROOT)))
RESULTS_DIR = Path(os.path.expanduser(str(DEFAULT_RESULTS_DIR)))

BATCH_SIZE = 256
NUM_WORKERS = 1
NUM_EPOCHS = 300
LEARNING_RATE = 3e-4
LATENT_DIM = 128
BETA = 0.25
WARMUP_EPOCHS = 50
CHECKPOINT_INTERVAL = 25
PREVIEW_INTERVAL = 50
LATENT_EXPORT_SPLIT = "train"
PREVIEW_BATCH_SIZE = 8
LOG_INTERVAL = 50
RECON_LOSS_TYPE = "l1"


def build_dataloaders(
    dataset_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], datasets.CIFAR10]:
    """Create train, train-eval, and test dataloaders for CIFAR10."""
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=str(dataset_path),
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = datasets.CIFAR10(
        root=str(dataset_path),
        train=False,
        transform=transform,
        download=False,
    )

    loader_kwargs = make_dataloader_kwargs(
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    dataloaders = {
        "train": train_loader,
        "train_eval": train_eval_loader,
        "test": test_loader,
    }
    return dataloaders, train_dataset


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    dataloaders, train_dataset = build_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )

    preview_batch = get_fixed_preview_batch(
        train_dataset,
        batch_size=PREVIEW_BATCH_SIZE,
    )

    model = ConvCIFARVAE(latent_dim=LATENT_DIM)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None

    trainer = CIFARVAETrainer(
        model=model,
        optimiser=optimiser,
        dataloaders=dataloaders,
        results_dir=RESULTS_DIR,
        num_epochs=NUM_EPOCHS,
        beta=BETA,
        warmup_epochs=WARMUP_EPOCHS,
        recon_loss_type=RECON_LOSS_TYPE,
        scaler=scaler,
        scheduler=None,
        device=device,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        preview_interval=PREVIEW_INTERVAL,
        log_interval=LOG_INTERVAL,
        latent_export_split=LATENT_EXPORT_SPLIT,
        preview_batch=preview_batch,
        use_amp=device.type == "cuda",
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Results directory: {RESULTS_DIR}")

    trainer.train()


if __name__ == "__main__":
    main()
