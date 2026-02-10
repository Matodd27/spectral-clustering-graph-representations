import torch
import torch.nn as nn

def ae_mse_loss(x: torch.Tensor, out) -> torch.Tensor:
    x_hat = out
    x = torch.flatten(x, start_dim=1)

    return nn.functional.mse_loss(x_hat, x, reduction="mean")

def beta_vae_loss(beta: float = 1.0):

    def loss_fn(x: torch.Tensor, out) -> torch.Tensor:
        x_hat, mu, logvar = out
        x = x.view(x.size(0), -1)

        recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon + beta * kld

    return loss_fn
