"""
layer2/decoder.py — Layer 2: Robot-specific Decoder (latent -> 关节角)
======================================================================
角色 (架构核心): decoder 只做「latent <-> 关节角」的翻译, 不预测未来。
  - Stage 1: z_t + m -> q_t   (重建, autoencoder)
  - Stage 2: z_{t+1} + m -> q_{t+1}  (transformer 给出 z_{t+1}, decoder 翻译)
  预测未来是 transformer 的事; decoder 对「现在/未来」一视同仁 = decision/control 解耦。

设计 (已确认):
  - 输入 z(latent) + m(morphology), 输出 q(arm_dof)
  - 2 层 MLP
  - Stage 1 不含 q_t 输入 (A 方案); 之后要「加前一步避免暴冲」时, 再扩输入 (预留 q_prev 参数)
"""
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