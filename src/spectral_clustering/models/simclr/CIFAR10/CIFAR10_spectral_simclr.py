import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torchvision.utils import make_grid, save_image

from spectral_clustering.graphs.constructors import knn_graph
from spectral_clustering.metrics.clustering import clustering_accuracy
from spectral_clustering.models.spectral.base import BaseSpectralClustering


DATASET_PATH = os.path.expanduser('~/scratch/data')
root = os.path.expanduser('~/scratch/simclr_cifar10')
results_dir = os.path.join(root, 'SimCLR_spectral_results')
os.makedirs(results_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


batch_size = 512
num_workers = 8
num_epochs = 500
tau = 0.5
warmup_epochs = 10
n_clusters = 10
spectral_enabled = True
spectral_start_epoch = 10
spectral_refresh_interval = 5
spectral_subset_size = 12000
spectral_subset_seed = 0
spectral_k = 15
spectral_lambda_max = 0.15
spectral_lambda_ramp_epochs = 10
spectral_margin = 0.2
spectral_use_cosine_features = True
spectral_eval_batch_size = 1024
losses_train = []


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


contrast_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return sample, label, index


class SimCLRNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            input_channels,
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

    def forward_with_features(self, x):
        h = self.encode(x)
        z = self.projector(h)
        return h, z

    def forward(self, x):
        _, z = self.forward_with_features(x)
        return z


def get_model(input_channels=3):
    return SimCLRNet(input_channels=input_channels)


def ntxent_loss(a, b, tau=0.5):
    z = torch.cat([a, b], dim=0)
    z = F.normalize(z, dim=1)

    logits = torch.matmul(z, z.T) / tau
    logits.fill_diagonal_(float('-inf'))

    batch_size_local = a.size(0)
    targets = torch.arange(2 * batch_size_local, device=z.device)
    targets = (targets + batch_size_local) % (2 * batch_size_local)

    return F.cross_entropy(logits, targets)


def spectral_consistency_loss(h1, h2, centroids, margin=0.2, centroid_mask=None):
    if centroids is None:
        return h1.new_zeros(())

    if centroid_mask is not None:
        centroids = centroids[centroid_mask]

    if centroids.numel() == 0:
        return h1.new_zeros(())

    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    h_mean = F.normalize((h1 + h2) * 0.5, dim=1)
    centroids = F.normalize(centroids, dim=1)

    centroid_sim = torch.matmul(h_mean, centroids.T)
    pseudo_labels = torch.argmax(centroid_sim, dim=1)

    sim = torch.matmul(h_mean, h_mean.T)
    same_cluster = pseudo_labels[:, None] == pseudo_labels[None, :]
    diagonal = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)

    pos_mask = same_cluster & ~diagonal
    neg_mask = ~same_cluster & ~diagonal

    pos_loss = h1.new_zeros(())
    neg_loss = h1.new_zeros(())

    if pos_mask.any():
        pos_loss = (1.0 - sim[pos_mask]).mean()
    if neg_mask.any():
        neg_loss = F.relu(sim[neg_mask] - margin).mean()

    return pos_loss + neg_loss

def encode_dataset(model_path):  
    model = get_model(input_channels=3).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    latents, labels = [], []

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(
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


def load_or_create_spectral_subset_indices(dataset_size, subset_size, seed, results_dir):
    subset_path = os.path.join(results_dir, 'spectral_subset_indices.npy')
    target_size = min(subset_size, dataset_size)

    if os.path.isfile(subset_path):
        subset_indices = np.load(subset_path)
        subset_indices = np.asarray(subset_indices, dtype=np.int64)
        is_valid = (
            subset_indices.ndim == 1
            and subset_indices.size == target_size
            and np.all(subset_indices >= 0)
            and np.all(subset_indices < dataset_size)
        )
        if is_valid:
            return np.sort(subset_indices)
        print("Existing spectral subset indices were invalid; regenerating them.")

    rng = np.random.default_rng(seed)
    subset_indices = np.sort(
        rng.choice(dataset_size, size=target_size, replace=False).astype(np.int64)
    )
    np.save(subset_path, subset_indices)
    return subset_indices


def save_augmentation_preview(dataset, preview_path, num_samples=6):
    preview_batches = []
    max_samples = min(num_samples, len(dataset))
    for idx in range(max_samples):
        sample = dataset[idx]
        if len(sample) == 3:
            views, _, _ = sample
        else:
            views, _ = sample
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
    def __init__(self, model, optimiser, dataloaders, loss_fn, scheduler, scaler, results_dir, spectral_subset_indices):
        self.model = model
        self.optimiser = optimiser
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.scaler = scaler
        self.results_dir = results_dir
        self.use_amp = torch.cuda.is_available()
        self.autocast_device = 'cuda' if self.use_amp else 'cpu'
        self.losses_train = []
        self.simclr_losses_train = []
        self.spectral_losses_train = []
        self.spectral_lambda_history = []
        self.start_epoch = 0

        self.spectral_subset_indices = np.asarray(spectral_subset_indices, dtype=np.int64)
        self.spectral_refresh_epochs = []
        self.spectral_accuracy_history = []
        self.spectral_true_labels_subset = None
        self.spectral_pred_labels_subset = None
        self.spectral_centroids = None
        self.spectral_centroid_mask = None
        self.spectral_targets_valid = False

        self.best_loss = float('inf')
        self.best_spectral_acc = None

    def latest_spectral_accuracy(self):
        if not self.spectral_accuracy_history:
            return None
        return float(self.spectral_accuracy_history[-1])

    def current_spectral_lambda(self, epoch):
        if not spectral_enabled or epoch < spectral_start_epoch:
            return 0.0
        if spectral_lambda_ramp_epochs <= 0:
            return float(spectral_lambda_max)

        progress = float(epoch - spectral_start_epoch) / float(spectral_lambda_ramp_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return float(spectral_lambda_max * progress)

    def should_refresh_spectral(self, epoch):
        if not spectral_enabled or epoch < spectral_start_epoch:
            return False
        if epoch == spectral_start_epoch:
            return True
        if spectral_refresh_interval <= 0:
            return False
        return (epoch - spectral_start_epoch) % spectral_refresh_interval == 0

    def save_history_files(self):
        np.savez(
            os.path.join(self.results_dir, 'losses_train.npz'),
            losses_train=np.array(self.losses_train, dtype=np.float32),
        )
        np.savez(
            os.path.join(self.results_dir, 'spectral_history.npz'),
            train_losses=np.array(self.losses_train, dtype=np.float32),
            simclr_losses=np.array(self.simclr_losses_train, dtype=np.float32),
            spectral_losses=np.array(self.spectral_losses_train, dtype=np.float32),
            spectral_lambdas=np.array(self.spectral_lambda_history, dtype=np.float32),
            spectral_refresh_epochs=np.array(self.spectral_refresh_epochs, dtype=np.int64),
            spectral_accuracy_values=np.array(self.spectral_accuracy_history, dtype=np.float32),
        )
        np.save(
            os.path.join(self.results_dir, 'spectral_subset_indices.npy'),
            self.spectral_subset_indices,
        )

    def update_spectral_centroids(self, features, assignments):
        previous_centroids = None
        previous_mask = None
        if self.spectral_centroids is not None and self.spectral_centroid_mask is not None:
            previous_centroids = self.spectral_centroids.detach().cpu().numpy()
            previous_mask = self.spectral_centroid_mask.detach().cpu().numpy().astype(bool)

        feature_dim = features.shape[1]
        centroids = np.zeros((n_clusters, feature_dim), dtype=np.float32)
        centroid_mask = np.zeros(n_clusters, dtype=bool)

        if previous_centroids is not None and previous_centroids.shape == centroids.shape:
            centroids[:] = previous_centroids.astype(np.float32)
            centroid_mask[:] = previous_mask

        for cluster_idx in range(n_clusters):
            cluster_members = assignments == cluster_idx
            if cluster_members.any():
                centroid = features[cluster_members].mean(axis=0).astype(np.float32)
                if spectral_use_cosine_features:
                    norm = float(np.linalg.norm(centroid))
                    if norm > 1e-12:
                        centroid = centroid / norm
                centroids[cluster_idx] = centroid
                centroid_mask[cluster_idx] = True

        if not centroid_mask.any():
            self.spectral_centroids = None
            self.spectral_centroid_mask = None
            self.spectral_targets_valid = False
            return

        if spectral_use_cosine_features:
            norms = np.linalg.norm(centroids[centroid_mask], axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            centroids[centroid_mask] = centroids[centroid_mask] / norms

        self.spectral_centroids = torch.from_numpy(centroids).to(device=device, dtype=torch.float32)
        self.spectral_centroid_mask = torch.from_numpy(centroid_mask).to(device=device)
        self.spectral_targets_valid = bool(centroid_mask.any())

    def refresh_spectral_targets(self, epoch):
        was_training = self.model.training
        self.model.eval()

        feature_batches = []
        label_batches = []
        index_batches = []

        with torch.no_grad():
            for x, y, indices in self.dataloaders['spectral_eval']:
                x = x.to(device, non_blocking=torch.cuda.is_available())
                h = self.model.encode(x)
                feature_batches.append(h.cpu())
                label_batches.append(y.cpu())
                index_batches.append(indices.cpu())

        if was_training:
            self.model.train()

        if not feature_batches:
            self.spectral_targets_valid = False
            return None

        subset_features = torch.cat(feature_batches, dim=0).numpy().astype(np.float32)
        subset_labels = torch.cat(label_batches, dim=0).numpy().astype(np.int64)
        subset_indices = torch.cat(index_batches, dim=0).numpy().astype(np.int64)

        order = np.argsort(subset_indices)
        subset_features = subset_features[order]
        subset_labels = subset_labels[order]
        subset_indices = subset_indices[order]

        if not np.array_equal(subset_indices, self.spectral_subset_indices):
            raise ValueError("Spectral subset loader indices do not match the fixed saved subset.")

        features_for_graph = subset_features.copy()
        if spectral_use_cosine_features:
            norms = np.linalg.norm(features_for_graph, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            features_for_graph = features_for_graph / norms

        W = knn_graph(features_for_graph, k=spectral_k, kernel='gaussian', symmetrise=True)
        spectral = BaseSpectralClustering(n_clusters=n_clusters, kind='symmetric')
        spectral_pred_labels_subset = spectral.fit_predict(W, kind='symmetric')
        spectral_true_labels_subset = subset_labels
        spectral_acc_subset = float(
            clustering_accuracy(spectral_true_labels_subset, spectral_pred_labels_subset)
        )

        self.spectral_pred_labels_subset = np.asarray(spectral_pred_labels_subset, dtype=np.int64)
        self.spectral_true_labels_subset = np.asarray(spectral_true_labels_subset, dtype=np.int64)
        self.spectral_refresh_epochs.append(int(epoch))
        self.spectral_accuracy_history.append(spectral_acc_subset)
        self.update_spectral_centroids(features_for_graph, self.spectral_pred_labels_subset)
        self.save_history_files()

        print(
            f"Spectral refresh @ epoch {epoch + 1:03d} | "
            f"subset_acc: {spectral_acc_subset:.2f}"
        )
        return spectral_acc_subset

    def is_new_best(self, epoch_loss):
        latest_spectral_acc = self.latest_spectral_accuracy()
        eps = 1e-8

        if latest_spectral_acc is None:
            if epoch_loss + eps < self.best_loss:
                self.best_loss = epoch_loss
                return True, 'loss'
            return False, None

        better_acc = (
            self.best_spectral_acc is None
            or latest_spectral_acc > self.best_spectral_acc + eps
        )
        same_acc_better_loss = (
            self.best_spectral_acc is not None
            and abs(latest_spectral_acc - self.best_spectral_acc) <= eps
            and epoch_loss + eps < self.best_loss
        )

        if better_acc or same_acc_better_loss:
            self.best_spectral_acc = latest_spectral_acc
            self.best_loss = epoch_loss
            return True, 'spectral'

        return False, None

    def save_checkpoint(self, epoch, best=False):
        self.save_history_files()
        torch.save(self.model.state_dict(), os.path.join(self.results_dir, 'model.pth'))
        torch.save(self.optimiser.state_dict(), os.path.join(self.results_dir, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(self.results_dir, 'scheduler.pth'))

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "losses_train": [float(x) for x in self.losses_train],
            "simclr_losses_train": [float(x) for x in self.simclr_losses_train],
            "spectral_losses_train": [float(x) for x in self.spectral_losses_train],
            "spectral_lambda_history": [float(x) for x in self.spectral_lambda_history],
            "spectral_refresh_epochs": [int(x) for x in self.spectral_refresh_epochs],
            "spectral_accuracy_history": [float(x) for x in self.spectral_accuracy_history],
            "spectral_subset_indices": self.spectral_subset_indices.copy(),
            "spectral_targets_valid": bool(self.spectral_targets_valid),
            "spectral_centroids": (
                self.spectral_centroids.detach().cpu()
                if self.spectral_centroids is not None else None
            ),
            "spectral_centroid_mask": (
                self.spectral_centroid_mask.detach().cpu()
                if self.spectral_centroid_mask is not None else None
            ),
            "best_loss": float(self.best_loss),
            "best_spectral_acc": (
                None if self.best_spectral_acc is None else float(self.best_spectral_acc)
            ),
        }

        if self.use_amp:
            scaler_state = self.scaler.state_dict()
            torch.save(scaler_state, os.path.join(self.results_dir, 'scaler.pth'))
            checkpoint['scaler_state_dict'] = scaler_state

        torch.save(checkpoint, os.path.join(self.results_dir, 'checkpoint_last.pth'))
        if best:
            torch.save(checkpoint, os.path.join(self.results_dir, 'checkpoint_best.pth'))
        if epoch % 100 == 0:
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

        checkpoint_subset_indices = checkpoint.get('spectral_subset_indices')
        if checkpoint_subset_indices is not None:
            checkpoint_subset_indices = np.asarray(checkpoint_subset_indices, dtype=np.int64)
            if not np.array_equal(checkpoint_subset_indices, self.spectral_subset_indices):
                raise ValueError(
                    "Checkpoint spectral subset indices differ from spectral_subset_indices.npy."
                )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        scaler_state = checkpoint.get('scaler_state_dict')
        if scaler_state is not None and self.use_amp:
            self.scaler.load_state_dict(scaler_state)

        self.start_epoch = checkpoint["epoch"] + 1
        self.losses_train = checkpoint.get("losses_train", [])
        self.simclr_losses_train = checkpoint.get("simclr_losses_train", [])
        self.spectral_losses_train = checkpoint.get("spectral_losses_train", [])
        self.spectral_lambda_history = checkpoint.get("spectral_lambda_history", [])
        self.spectral_refresh_epochs = checkpoint.get("spectral_refresh_epochs", [])
        self.spectral_accuracy_history = checkpoint.get("spectral_accuracy_history", [])
        self.spectral_targets_valid = bool(checkpoint.get("spectral_targets_valid", False))
        self.best_loss = float(
            checkpoint.get(
                "best_loss",
                min(self.losses_train) if self.losses_train else float('inf'),
            )
        )
        self.best_spectral_acc = checkpoint.get("best_spectral_acc")

        spectral_centroids = checkpoint.get("spectral_centroids")
        spectral_centroid_mask = checkpoint.get("spectral_centroid_mask")
        if spectral_centroids is not None and spectral_centroid_mask is not None:
            self.spectral_centroids = spectral_centroids.to(device=device, dtype=torch.float32)
            self.spectral_centroid_mask = spectral_centroid_mask.to(device=device, dtype=torch.bool)
            self.spectral_targets_valid = bool(self.spectral_centroid_mask.any().item())
        else:
            self.spectral_centroids = None
            self.spectral_centroid_mask = None
            self.spectral_targets_valid = False

        print(f"Resuming from epoch {self.start_epoch}.")

    def train(self, epochs=100):
        self.load_checkpoint()
        if self.start_epoch >= epochs:
            print(f"Training already completed through epoch {self.start_epoch - 1}.")
            return self.losses_train

        if self.losses_train and self.best_loss == float('inf'):
            self.best_loss = min(self.losses_train)

        for epoch in range(self.start_epoch, epochs):
            refreshed_this_epoch = False
            if self.should_refresh_spectral(epoch):
                self.refresh_spectral_targets(epoch)
                refreshed_this_epoch = True

            self.model.train()
            epoch_losses_train = []
            epoch_simclr_losses = []
            epoch_spectral_losses = []
            self.scheduler.step(epoch)
            lambda_spec = self.current_spectral_lambda(epoch)

            for imgs, _, _ in self.dataloaders['train']:
                x1 = imgs[0].to(device, non_blocking=torch.cuda.is_available())
                x2 = imgs[1].to(device, non_blocking=torch.cuda.is_available())

                self.optimiser.zero_grad(set_to_none=True)

                with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
                    h1, z1 = self.model.forward_with_features(x1)
                    h2, z2 = self.model.forward_with_features(x2)
                    simclr_loss = self.loss_fn(z1, z2, tau=tau)

                    if spectral_enabled and self.spectral_targets_valid and lambda_spec > 0.0:
                        spectral_loss = spectral_consistency_loss(
                            h1,
                            h2,
                            self.spectral_centroids,
                            margin=spectral_margin,
                            centroid_mask=self.spectral_centroid_mask,
                        )
                    else:
                        spectral_loss = simclr_loss.new_zeros(())

                    loss = simclr_loss + lambda_spec * spectral_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()

                epoch_losses_train.append(loss.detach().item())
                epoch_simclr_losses.append(simclr_loss.detach().item())
                epoch_spectral_losses.append(spectral_loss.detach().item())

            epoch_loss = float(np.mean(epoch_losses_train))
            epoch_simclr_loss = float(np.mean(epoch_simclr_losses))
            epoch_spectral_loss = float(np.mean(epoch_spectral_losses))

            self.losses_train.append(epoch_loss)
            self.simclr_losses_train.append(epoch_simclr_loss)
            self.spectral_losses_train.append(epoch_spectral_loss)
            self.spectral_lambda_history.append(float(lambda_spec))

            is_best, best_reason = self.is_new_best(epoch_loss)
            self.save_checkpoint(epoch, best=is_best)

            latest_spectral_acc = self.latest_spectral_accuracy()
            spectral_acc_str = "n/a" if latest_spectral_acc is None else f"{latest_spectral_acc:.2f}"
            print(
                f"Epoch {epoch + 1:03d}/{epochs:03d} | "
                f"lr: {self.optimiser.param_groups[0]['lr']:.6f} | "
                f"loss: {epoch_loss:.4f} | "
                f"simclr: {epoch_simclr_loss:.4f} | "
                f"spectral: {epoch_spectral_loss:.4f} | "
                f"lambda: {lambda_spec:.4f} | "
                f"refresh: {'yes' if refreshed_this_epoch else 'no'} | "
                f"subset_acc: {spectral_acc_str}"
            )

            if is_best and best_reason == 'spectral' and latest_spectral_acc is not None:
                print(
                    f"New best spectral checkpoint @ epoch {epoch + 1:03d} | "
                    f"subset_acc: {latest_spectral_acc:.2f} | "
                    f"train_loss: {epoch_loss:.4f}"
                )

        return self.losses_train


CIFAR10_contrast = IndexedDataset(
    datasets.CIFAR10(
        root=DATASET_PATH,
        train=True,
        download=False,
        transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    )
)

CIFAR10_eval = IndexedDataset(
    datasets.CIFAR10(
        root=DATASET_PATH,
        train=True,
        download=False,
        transform=eval_transforms,
    )
)

spectral_subset_indices = load_or_create_spectral_subset_indices(
    dataset_size=len(CIFAR10_eval),
    subset_size=spectral_subset_size,
    seed=spectral_subset_seed,
    results_dir=results_dir,
)

train_loader = DataLoader(
    CIFAR10_contrast,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
)

eval_loader = DataLoader(
    CIFAR10_eval,
    batch_size=spectral_eval_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
)

spectral_eval_loader = DataLoader(
    Subset(CIFAR10_eval, spectral_subset_indices.tolist()),
    batch_size=spectral_eval_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=1,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
)


def main():
    global losses_train

    save_augmentation_preview(
        CIFAR10_contrast,
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

    dataloaders = {
        'train': train_loader,
        'eval': eval_loader,
        'spectral_eval': spectral_eval_loader,
    }

    simclrobj = SimCLR(
        model,
        optimizer,
        dataloaders,
        ntxent_loss,
        scheduler,
        scaler,
        results_dir,
        spectral_subset_indices,
    )
    losses_train = simclrobj.train(epochs=num_epochs)


if __name__ == '__main__':
    main()
