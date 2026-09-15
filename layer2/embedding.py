import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, in_dim=21, latent_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, fk21):
        """fk21: (B,21) -> z: (B,latent_dim)。"""
        return self.proj(fk21)


class MLPEmbedding(nn.Module):
    def __init__(self, in_dim=21, latent_dim=64, hidden=128):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, fk21):
        return self.net(fk21)