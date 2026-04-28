from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# Note the CIFAR-10 vae was trained using the Computational Shared Facility at the University of Manchester, dataset paths and results directories are set to ~/scratch by default, and the training script is in train_vae_cifar.py
DEFAULT_DATASET_PATH = Path("~/scratch/data").expanduser()
DEFAULT_ROOT = Path("~/scratch/vae_cifar").expanduser()
DEFAULT_RESULTS_DIR = DEFAULT_ROOT / "VAE_cifar"

SplitName = Literal["train", "test", "both"]


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return the target device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloader_kwargs(
    num_workers: int,
    pin_memory: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def normalise_history(history: list[dict[str, Any]] | None) -> list[dict[str, float]]:
    if not history:
        return []

    normalised: list[dict[str, float]] = []
    for entry in history:
        normalised.append(
            {
                "epoch": float(entry["epoch"]),
                "total_loss": float(entry["total_loss"]),
                "recon_loss": float(entry["recon_loss"]),
                "kld_loss": float(entry["kld_loss"]),
                "beta": float(entry["beta"]),
                "learning_rate": float(entry["learning_rate"]),
                "mean_abs_mu": float(entry.get("mean_abs_mu", 0.0)),
                "mean_logvar": float(entry.get("mean_logvar", 0.0)),
                "std_mu": float(entry.get("std_mu", 0.0)),
            }
        )
    return normalised


def loss_history_from_legacy(losses_train: list[float] | np.ndarray | None) -> list[dict[str, float]]:
    if losses_train is None:
        return []

    values = [float(v) for v in np.asarray(losses_train, dtype=np.float32).tolist()]
    history: list[dict[str, float]] = []
    for index, value in enumerate(values, start=1):
        history.append(
            {
                "epoch": float(index),
                "total_loss": value,
                "recon_loss": value,
                "kld_loss": 0.0,
                "beta": 1.0,
                "learning_rate": 0.0,
                "mean_abs_mu": 0.0,
                "mean_logvar": 0.0,
                "std_mu": 0.0,
            }
        )
    return history


def extract_model_state_dict(
    checkpoint_or_state: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    if "model_state_dict" in checkpoint_or_state:
        state_dict = checkpoint_or_state["model_state_dict"]
        return state_dict, checkpoint_or_state
    return checkpoint_or_state, None


def infer_model_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    first_conv = state_dict["encoder.0.weight"]
    hidden_weight = state_dict["encoder_to_hidden.weight"]
    mu_weight = state_dict["fc_mu.weight"]
    return {
        "input_channels": int(first_conv.shape[1]),
        "base_channels": int(first_conv.shape[0]),
        "hidden_dim": int(hidden_weight.shape[0]),
        "latent_dim": int(mu_weight.shape[0]),
    }


class ConvCIFARVAE(nn.Module):

    def __init__(
        self,
        latent_dim: int = 128,
        input_channels: int = 3,
        base_channels: int = 64,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.feature_channels = base_channels * 8
        self.feature_spatial_size = 2
        self.feature_dim = self.feature_channels * self.feature_spatial_size * self.feature_spatial_size

        self.encoder_to_hidden = nn.Linear(self.feature_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_from_latent = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_decoder = nn.Linear(hidden_dim, self.feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.feature_channels,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels,
                input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        hidden = F.relu(self.encoder_to_hidden(features), inplace=True)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.decoder_from_latent(z), inplace=True)
        features = F.relu(self.hidden_to_decoder(hidden), inplace=True)
        features = features.view(
            z.size(0),
            self.feature_channels,
            self.feature_spatial_size,
            self.feature_spatial_size,
        )
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def beta_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    recon_loss_type: str = "l1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses averaged per sample."""
    batch_size = x.size(0)
    if recon_loss_type == "l1":
        recon_loss = F.l1_loss(recon_x, x, reduction="sum") / batch_size
    elif recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    else:
        raise ValueError("recon_loss_type must be either 'l1' or 'mse'.")
    kld_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss


@torch.no_grad()
def encode_dataloader_mu(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    target_device = resolve_device(device)
    amp_enabled = bool(use_amp) and target_device.type == "cuda"
    autocast_device = "cuda" if target_device.type == "cuda" else "cpu"

    model = model.to(target_device)
    model.eval()

    latents: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    non_blocking = target_device.type == "cuda"

    for images, batch_labels in dataloader:
        images = images.to(target_device, non_blocking=non_blocking)
        with torch.amp.autocast(autocast_device, enabled=amp_enabled):
            mu, _ = model.encode(images)

        latents.append(mu.float().cpu())
        labels.append(batch_labels.cpu())

    latents_array = torch.cat(latents, dim=0).numpy().astype(np.float32, copy=False)
    labels_array = torch.cat(labels, dim=0).numpy()
    return latents_array, labels_array


def get_fixed_preview_batch(
    dataset: Dataset[Any],
    batch_size: int = 8,
) -> torch.Tensor:
    """Collect a deterministic small batch from the start of a dataset."""
    images: list[torch.Tensor] = []
    max_items = min(batch_size, len(dataset))
    for index in range(max_items):
        sample = dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        images.append(image)

    if not images:
        raise ValueError("Cannot build a preview batch from an empty dataset.")

    return torch.stack(images, dim=0)


def save_reconstruction_preview(
    model: nn.Module,
    preview_batch: torch.Tensor,
    output_path: str | Path,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
) -> None:
    """Save a side-by-side grid of originals and reconstructions."""
    target_device = resolve_device(device)
    amp_enabled = torch.cuda.is_available() if use_amp is None else bool(use_amp)
    amp_enabled = amp_enabled and target_device.type == "cuda"
    autocast_device = "cuda" if target_device.type == "cuda" else "cpu"

    model = model.to(target_device)
    model.eval()

    preview_batch = preview_batch.to(target_device, non_blocking=target_device.type == "cuda")
    with torch.no_grad():
        with torch.amp.autocast(autocast_device, enabled=amp_enabled):
            mu, _ = model.encode(preview_batch)
            reconstructions = model.decode(mu)

    originals = preview_batch.detach().cpu()
    reconstructions = reconstructions.detach().cpu().clamp_(0.0, 1.0)
    comparison = torch.cat([originals, reconstructions], dim=0)
    grid = make_grid(comparison, nrow=originals.size(0), padding=2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)


def save_training_history(
    results_dir: str | Path,
    history: list[dict[str, float]],
) -> None:
    """Persist epoch-wise loss history in CSV, NPZ, and plot form."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    epochs = np.asarray([entry["epoch"] for entry in history], dtype=np.int32)
    total_loss = np.asarray([entry["total_loss"] for entry in history], dtype=np.float32)
    recon_loss = np.asarray([entry["recon_loss"] for entry in history], dtype=np.float32)
    kld_loss = np.asarray([entry["kld_loss"] for entry in history], dtype=np.float32)
    beta = np.asarray([entry["beta"] for entry in history], dtype=np.float32)
    learning_rate = np.asarray([entry["learning_rate"] for entry in history], dtype=np.float32)
    mean_abs_mu = np.asarray([entry["mean_abs_mu"] for entry in history], dtype=np.float32)
    mean_logvar = np.asarray([entry["mean_logvar"] for entry in history], dtype=np.float32)
    std_mu = np.asarray([entry["std_mu"] for entry in history], dtype=np.float32)

    np.savez(results_path / "losses_train.npz", losses_train=total_loss)
    np.savez(
        results_path / "history.npz",
        epoch=epochs,
        total_loss=total_loss,
        recon_loss=recon_loss,
        kld_loss=kld_loss,
        beta=beta,
        learning_rate=learning_rate,
        mean_abs_mu=mean_abs_mu,
        mean_logvar=mean_logvar,
        std_mu=std_mu,
    )

    with (results_path / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "total_loss",
                "recon_loss",
                "kld_loss",
                "beta",
                "learning_rate",
                "mean_abs_mu",
                "mean_logvar",
                "std_mu",
            ]
        )
        for entry in history:
            writer.writerow(
                [
                    int(entry["epoch"]),
                    float(entry["total_loss"]),
                    float(entry["recon_loss"]),
                    float(entry["kld_loss"]),
                    float(entry["beta"]),
                    float(entry["learning_rate"]),
                    float(entry["mean_abs_mu"]),
                    float(entry["mean_logvar"]),
                    float(entry["std_mu"]),
                ]
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, total_loss, label="Total loss", linewidth=2.0)
    ax.plot(epochs, recon_loss, label="Reconstruction loss", linewidth=1.5)
    ax.plot(epochs, kld_loss, label="KL loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CIFAR10 VAE Training Losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_path / "losses_plot.png", dpi=200, bbox_inches="tight")
    fig.savefig(results_path / "losses_plot.pdf", bbox_inches="tight")
    plt.close(fig)


class CIFARVAETrainer:

    def __init__(
        self,
        model: ConvCIFARVAE,
        optimiser: Optimizer,
        dataloaders: dict[str, DataLoader],
        results_dir: str | Path,
        num_epochs: int = 300,
        beta: float = 0.25,
        warmup_epochs: int = 50,
        recon_loss_type: str = "l1",
        scaler: torch.amp.GradScaler | None = None,
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
        checkpoint_interval: int = 25,
        preview_interval: int = 50,
        log_interval: int = 50,
        latent_export_split: SplitName = "train",
        preview_batch: torch.Tensor | None = None,
        use_amp: bool | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.optimiser = optimiser
        self.dataloaders = dataloaders
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.target_beta = float(beta)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.recon_loss_type = recon_loss_type
        self.scheduler = scheduler
        self.scaler = scaler
        self.checkpoint_interval = checkpoint_interval
        self.preview_interval = preview_interval
        self.log_interval = log_interval
        self.latent_export_split = latent_export_split
        self.preview_batch = preview_batch.cpu() if preview_batch is not None else None
        self.use_amp = (self.device.type == "cuda") if use_amp is None else bool(use_amp)
        self.use_amp = self.use_amp and self.device.type == "cuda"
        self.autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.non_blocking = self.device.type == "cuda"
        self.history: list[dict[str, float]] = []
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.model_config = {
            "latent_dim": model.latent_dim,
            "input_channels": model.input_channels,
            "base_channels": model.base_channels,
            "hidden_dim": model.hidden_dim,
        }

        if self.use_amp and self.scaler is None:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)

        if latent_export_split not in {"train", "test", "both"}:
            raise ValueError("latent_export_split must be one of 'train', 'test', or 'both'.")

        if "train" not in self.dataloaders:
            raise KeyError("dataloaders must include a 'train' dataloader.")

    def beta_at_epoch(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.target_beta
        progress = min(1.0, max(0.0, float(epoch + 1) / float(self.warmup_epochs)))
        return self.target_beta * progress

    def current_learning_rate(self) -> float:
        return float(self.optimiser.param_groups[0]["lr"])

    def _build_checkpoint(self, epoch: int) -> dict[str, Any]:
        checkpoint: dict[str, Any] = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "history": self.history,
            "losses_train": [float(entry["total_loss"]) for entry in self.history],
            "best_loss": float(self.best_loss),
            "model_config": self.model_config,
            "num_epochs": int(self.num_epochs),
            "target_beta": float(self.target_beta),
            "warmup_epochs": int(self.warmup_epochs),
            "recon_loss_type": self.recon_loss_type,
            "checkpoint_interval": int(self.checkpoint_interval),
            "preview_interval": int(self.preview_interval),
            "latent_export_split": self.latent_export_split,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None and self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        return checkpoint

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        save_training_history(self.results_dir, self.history)
        checkpoint = self._build_checkpoint(epoch)

        last_path = self.results_dir / "checkpoint_last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            torch.save(checkpoint, self.results_dir / "checkpoint_best.pth")

        epoch_number = epoch + 1
        if self.checkpoint_interval > 0 and epoch_number % self.checkpoint_interval == 0:
            torch.save(checkpoint, self.results_dir / f"checkpoint_{epoch_number:03d}.pth")

    def load_checkpoint(self) -> None:
        checkpoint_path = self.results_dir / "checkpoint_last.pth"
        if not checkpoint_path.is_file():
            return

        loaded = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict, checkpoint = extract_model_state_dict(loaded)
        self.model.load_state_dict(state_dict)

        if checkpoint is None:
            self.start_epoch = 0
            self.history = []
            self.best_loss = float("inf")
            print("Loaded raw model weights from checkpoint_last.pth. Optimiser state was not available.")
            return

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimiser.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None and self.scaler is not None and self.use_amp:
            self.scaler.load_state_dict(scaler_state)

        self.recon_loss_type = checkpoint.get("recon_loss_type", self.recon_loss_type)

        history = checkpoint.get("history")
        if history is not None:
            self.history = normalise_history(history)
        else:
            self.history = loss_history_from_legacy(checkpoint.get("losses_train"))

        self.best_loss = float(
            checkpoint.get(
                "best_loss",
                min((entry["total_loss"] for entry in self.history), default=float("inf")),
            )
        )
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"Resuming VAE training from epoch {self.start_epoch}.")

    def _maybe_step_scheduler(self, epoch: int) -> None:
        if self.scheduler is None:
            return
        try:
            self.scheduler.step()
        except TypeError:
            self.scheduler.step(epoch)

    def export_latents(
        self,
        split: SplitName | None = None,
        epoch_number: int | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Encode and optionally save latent means for the requested split(s)."""
        export_split = split or self.latent_export_split
        if export_split not in {"train", "test", "both"}:
            raise ValueError("split must be one of 'train', 'test', or 'both'.")

        if export_split == "both":
            split_names = ["train", "test"]
        else:
            split_names = [export_split]

        outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split_name in split_names:
            if split_name == "train":
                loader = self.dataloaders.get("train_eval", self.dataloaders["train"])
            else:
                loader = self.dataloaders.get("test")

            if loader is None:
                continue

            latents, labels = encode_dataloader_mu(
                self.model,
                loader,
                device=self.device,
                use_amp=self.use_amp,
            )
            outputs[split_name] = (latents, labels)

            if epoch_number is not None:
                np.savez(
                    self.results_dir / f"latents_{split_name}_epoch_{epoch_number:03d}.npz",
                    latents=latents,
                )
                np.savez(
                    self.results_dir / f"labels_{split_name}_epoch_{epoch_number:03d}.npz",
                    labels=labels,
                )

        return outputs

    def save_reconstruction_preview(self, epoch_number: int) -> None:
        if self.preview_batch is None:
            return

        save_reconstruction_preview(
            self.model,
            self.preview_batch,
            self.results_dir / f"recon_preview_epoch_{epoch_number:03d}.png",
            device=self.device,
            use_amp=self.use_amp,
        )

    def train(self) -> list[dict[str, float]]:
        self.load_checkpoint()
        if self.start_epoch >= self.num_epochs:
            print(
                f"Training already completed at epoch {self.start_epoch}. "
                f"No further work is needed."
            )
            return self.history

        total_batches = len(self.dataloaders["train"])

        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            beta_t = self.beta_at_epoch(epoch)

            total_loss_sum = 0.0
            recon_loss_sum = 0.0
            kld_loss_sum = 0.0
            total_samples = 0
            mu_abs_sum = 0.0
            mu_sum = 0.0
            mu_sq_sum = 0.0
            logvar_sum = 0.0
            total_latent_values = 0

            for step, (images, _) in enumerate(self.dataloaders["train"], start=1):
                images = images.to(self.device, non_blocking=self.non_blocking)
                batch_size = images.size(0)

                self.optimiser.zero_grad(set_to_none=True)

                with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
                    reconstructions, mu, logvar = self.model(images)
                    total_loss, recon_loss, kld_loss = beta_vae_loss(
                        reconstructions,
                        images,
                        mu,
                        logvar,
                        beta=beta_t,
                        recon_loss_type=self.recon_loss_type,
                    )

                if self.scaler is not None and self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    self.optimiser.step()

                total_loss_sum += float(total_loss.detach().item()) * batch_size
                recon_loss_sum += float(recon_loss.detach().item()) * batch_size
                kld_loss_sum += float(kld_loss.detach().item()) * batch_size
                total_samples += batch_size
                mu_abs_sum += float(mu.detach().abs().sum().item())
                mu_sum += float(mu.detach().sum().item())
                mu_sq_sum += float(mu.detach().pow(2).sum().item())
                logvar_sum += float(logvar.detach().sum().item())
                total_latent_values += mu.numel()

                if (
                    self.log_interval > 0
                    and (step == 1 or step % self.log_interval == 0 or step == total_batches)
                ):
                    print(
                        f"Epoch {epoch + 1:03d}/{self.num_epochs:03d} | "
                        f"Step {step:04d}/{total_batches:04d} | "
                        f"loss: {total_loss.detach().item():.4f} | "
                        f"recon: {recon_loss.detach().item():.4f} | "
                        f"kld: {kld_loss.detach().item():.4f} | "
                        f"beta: {beta_t:.4f}"
                    )

            self._maybe_step_scheduler(epoch)

            epoch_total_loss = total_loss_sum / max(1, total_samples)
            epoch_recon_loss = recon_loss_sum / max(1, total_samples)
            epoch_kld_loss = kld_loss_sum / max(1, total_samples)
            learning_rate = self.current_learning_rate()
            mean_abs_mu = mu_abs_sum / max(1, total_latent_values)
            mean_logvar = logvar_sum / max(1, total_latent_values)
            mean_mu = mu_sum / max(1, total_latent_values)
            std_mu = max(0.0, (mu_sq_sum / max(1, total_latent_values)) - (mean_mu * mean_mu)) ** 0.5

            self.history.append(
                {
                    "epoch": float(epoch + 1),
                    "total_loss": float(epoch_total_loss),
                    "recon_loss": float(epoch_recon_loss),
                    "kld_loss": float(epoch_kld_loss),
                    "beta": float(beta_t),
                    "learning_rate": float(learning_rate),
                    "mean_abs_mu": float(mean_abs_mu),
                    "mean_logvar": float(mean_logvar),
                    "std_mu": float(std_mu),
                }
            )

            is_best = epoch_total_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_total_loss

            self.save_checkpoint(epoch, is_best=is_best)

            epoch_number = epoch + 1
            if self.checkpoint_interval > 0 and epoch_number % self.checkpoint_interval == 0:
                self.export_latents(epoch_number=epoch_number)

            if self.preview_interval > 0 and epoch_number % self.preview_interval == 0:
                self.save_reconstruction_preview(epoch_number)

            print(
                f"Epoch {epoch_number:03d}/{self.num_epochs:03d} | "
                f"train_total: {epoch_total_loss:.4f} | "
                f"train_recon: {epoch_recon_loss:.4f} | "
                f"train_kld: {epoch_kld_loss:.4f} | "
                f"beta: {beta_t:.4f} | "
                f"lr: {learning_rate:.6f} | "
                f"mean_abs_mu: {mean_abs_mu:.4f} | "
                f"mean_logvar: {mean_logvar:.4f} | "
                f"std_mu: {std_mu:.4f}"
            )

        return self.history


def load_vae_model(
    checkpoint_path: str | Path | dict[str, Any],
    device: str | torch.device | None = None,
) -> tuple[ConvCIFARVAE, dict[str, Any] | None]:
    """Load a VAE model from either a full checkpoint or a raw state dict."""
    target_device = resolve_device(device)
    if isinstance(checkpoint_path, dict):
        loaded = checkpoint_path
    else:
        loaded = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    state_dict, checkpoint = extract_model_state_dict(loaded)

    if checkpoint is not None and "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    else:
        model_config = infer_model_config_from_state_dict(state_dict)

    model = ConvCIFARVAE(**model_config)
    model.load_state_dict(state_dict)
    model = model.to(target_device)
    model.eval()
    return model, checkpoint


def encode_cifar10_from_checkpoint(
    checkpoint_path: str | Path | dict[str, Any],
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    split: SplitName = "train",
    batch_size: int = 256,
    num_workers: int = 1,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    download: bool = False,
    save_dir: str | Path | None = None,
    epoch_tag: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load a checkpoint or raw state dict, encode CIFAR10 with latent means, and return numpy arrays.

    When `split` is "train" or "test", this returns `(latents, labels)`.
    When `split` is "both", this returns a dict containing both splits.
    """
    target_device = resolve_device(device)
    model, _ = load_vae_model(checkpoint_path, device=target_device)

    transform = transforms.ToTensor()
    dataset_root = Path(dataset_path).expanduser()
    pin_memory = target_device.type == "cuda"
    loader_kwargs = make_dataloader_kwargs(num_workers=num_workers, pin_memory=pin_memory)

    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    split_names = ["train", "test"] if split == "both" else [split]

    for split_name in split_names:
        is_train = split_name == "train"
        dataset = datasets.CIFAR10(
            root=str(dataset_root),
            train=is_train,
            transform=transform,
            download=download,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        latents, labels = encode_dataloader_mu(
            model,
            dataloader,
            device=target_device,
            use_amp=use_amp,
        )
        outputs[split_name] = (latents, labels)

        if save_dir is not None:
            destination = Path(save_dir).expanduser()
            destination.mkdir(parents=True, exist_ok=True)
            if epoch_tag is None:
                np.savez(destination / f"latents_{split_name}.npz", latents=latents)
                np.savez(destination / f"labels_{split_name}.npz", labels=labels)
            else:
                np.savez(
                    destination / f"latents_{split_name}_epoch_{epoch_tag:03d}.npz",
                    latents=latents,
                )
                np.savez(
                    destination / f"labels_{split_name}_epoch_{epoch_tag:03d}.npz",
                    labels=labels,
                )

    if split == "both":
        return outputs
    return outputs[split]
