from typing import Callable, Optional, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import numpy as np


class AETrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable[[torch.Tensor, Any], torch.Tensor]] = None,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.loss_fn = loss_fn or (lambda x, out: nn.functional.mse_loss(out, x))
        self.lr = lr

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int = 20,
        optimiser: Optional[torch.optim.Optimizer] = None,
    ):
        self.model.to(self.device)
        self.model.train()

        if optimiser is None:
            optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_samples = 0

            for batch_idx, (x, *_) in enumerate(train_loader):
                x = x.to(self.device)
                optimiser.zero_grad()
                outputs = self.model(x)                 
                loss = self.loss_fn(x, outputs)    
                loss.backward()
                optimiser.step()

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}")

        return self.model

    @torch.no_grad()
    def encode_dataset(
        self,
        data,
        batch_size: int = 256,
    ):
        self.model.to(self.device)
        self.model.eval()

        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        latents = []

        for x, *rest in loader:
            x = x.to(self.device)

            out = self.model.encode(x)
            if isinstance(out, tuple):
                z = out[0]
            else:
                z = out

            latents.append(z.cpu())


        return np.array(torch.cat(latents, dim=0).cpu())


    @torch.no_grad()
    def decode_latent(
        self,
        data,
        batch_size: int=256
    ):
        return self.model.decode(torch.Tensor(data))