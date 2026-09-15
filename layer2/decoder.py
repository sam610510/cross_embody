import torch
import torch.nn as nn


class Decoder(nn.Module):
    """z(latent) + m(3) [+ 可选 q_prev] -> q(arm_dof)。"""
    def __init__(self, latent_dim=64, m_dim=3, out_dim=7, hidden=128, depth=3, use_q_prev=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.m_dim = m_dim
        self.out_dim = out_dim
        self.use_q_prev = use_q_prev
        in_dim = latent_dim + m_dim + (out_dim if use_q_prev else 0)
        # depth = 隐藏层数量 (depth=2 为原本 2 层, depth=3 加深)
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, m, q_prev=None):
        """z:(B,latent) m:(B,3) q_prev:(B,out_dim) 可选。回传 q:(B,out_dim)。"""
        parts = [z, m]
        if self.use_q_prev:
            if q_prev is None:
                raise ValueError("use_q_prev=True 但未提供 q_prev")
            parts.append(q_prev)
        return self.net(torch.cat(parts, dim=1))