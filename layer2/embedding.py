"""
layer2/embedding.py — Layer 2: Embedding W (19维 FK 特征 -> 64维 latent)
========================================================================
设计 (已确认):
  - 纯线性 nn.Linear(19, 64) + bias, 轻量 (~1280 参数/机器人)
  - 体现架构主张: Layer 1 已用物理先验扛掉大部分, W 只学残差对齐
  - 每台机器人一个 W (Stage 3 新机器人建自己的 W_new)

之后若线性表达力不足, 可换 MLPEmbedding (附在下方, 默认不用)。
"""
import torch
import torch.nn as nn


class Embedding(nn.Module):
    """纯线性 embedding: 19 -> latent_dim。每台机器人专属。"""
    def __init__(self, in_dim=21, latent_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, fk21):
        """fk21: (B,21) -> z: (B,latent_dim)。"""
        return self.proj(fk21)


class MLPEmbedding(nn.Module):
    """备案: 线性不够时换这个 (2 层 MLP)。默认不用。"""
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