# spectral_clustering/models/autoencoders/conv_vae.py

from .base import BaseAE
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(BaseAE):
    def __init__(
        self,
        latent_dim: int,
        input_channels: int = 1,
        input_height: int = 28,
        input_width: int = 28,
    ):
        input_dim = input_channels * input_height * input_width
        super().__init__(latent_dim=latent_dim, input_dim=input_dim)

        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # encoder
        self.enc_conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_fc = nn.Linear(64 * 7 * 7, 256)

        self.mean_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)

        # decoder
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 64 * 7 * 7)

        self.dec_deconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.dec_deconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=input_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ) 

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def _ensure_image_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(
                x.size(0),
                self.input_channels,
                self.input_height,
                self.input_width,
            )
        return x

    def encode(self, x: torch.Tensor):
        x = self._ensure_image_shape(x)
        h = self.activation(self.enc_conv1(x))
        h = self.activation(self.enc_conv2(h))
        h = h.view(h.size(0), -1)  # flatten to (B, 64*7*7)
        h = self.activation(self.enc_fc(h))

        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar

    def reparametrisation(self, mean: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z: torch.Tensor):
        h = self.activation(self.dec_fc1(z))
        h = self.activation(self.dec_fc2(h))
        h = h.view(h.size(0), 64, 7, 7)
        h = self.activation(self.dec_deconv1(h))
        x_hat = torch.sigmoid(self.dec_deconv2(h))
        return x_hat

    def forward(self, x: torch.Tensor):
        x = self._ensure_image_shape(x)
        mean, logvar = self.encode(x)
        z = self.reparametrisation(mean, logvar)
        x_hat_img = self.decode(z)  # (B, 1, 28, 28)
        x_hat = x_hat_img.view(x_hat_img.size(0), -1)  # (B, 784)
        return x_hat, mean, logvar
