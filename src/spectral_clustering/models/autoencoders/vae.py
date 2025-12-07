from .base import BaseAE
import torch.nn as nn
import torch

class VAE(BaseAE):
    def __init__(self, latent_dim, input_dim=784, hidden_layer=400):
        super().__init__(latent_dim=latent_dim, input_dim=input_dim)

        # encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_layer),
            nn.LeakyReLU(0.2)
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(hidden_layer, latent_dim)
        self.logvar_layer = nn.Linear(hidden_layer, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layer),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_layer, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def reparametrisation(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrisation(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar