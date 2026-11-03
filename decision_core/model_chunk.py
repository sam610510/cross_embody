"""
Decision core with ACTION CHUNKING (separate from model.py, which keeps the original causal
single-step decision core unchanged).

  history encoder : the same TrunkBlock stack as model.py over the 8-step latent history
                    (+ gripper opening per step); attention over the history is BIDIRECTIONAL by
                    default (ACT's observation encoder) -- every history step is in the past, so no
                    causal mask is needed. causal=True reproduces the original trunk (ablation).
  chunk decoder   : K query tokens = MLP(goal) + learned position of the step in the chunk;
                    pre-LN transformer decoder layers (self-attention among the K queries +
                    cross-attention to the history tokens). The goal is the query, so it cannot be
                    bypassed (GoalQueryPooling with K queries instead of one).
  outputs         : z_hat (B,K,128) = latents of the next K states, decoded by each arm as before;
                    gripper 'closed' logits (B,K)

The chunk is the decision core's INTENT for the next K steps: each arm's control layer may use the
whole future TCP path (e.g. turn the elbow away from an obstacle before it gets there).

Checkpoints: decision_core_chunk_K{K}_{bi|causal}_l{L}_h{H}_s{seed}{tag}.pt  (train_pp_chunk.py)
Load any pick-and-place checkpoint (chunk or original) with load_core().
"""
import numpy as np
import torch
import torch.nn as nn

from decision_core.model import (TrunkBlock, build_causal_mask, load_pp_core, D_MODEL, N_LAYERS, N_HEADS,
                                 FFN_MULT, T_HISTORY, GOAL_DIM_PP, FINGER_SCALE)


class HistoryEncoder(nn.Module):
    """TrunkBlock stack over the history tokens, bidirectional or causal."""

    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, t_history=T_HISTORY,
                 dropout=0.1, causal=False):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, t_history, d_model) * 0.02)
        self.blocks = nn.ModuleList([TrunkBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        mask = build_causal_mask(t_history) if causal else torch.zeros(t_history, t_history)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        x = x + self.pos_embed
        mask = self.attn_mask.to(dtype=x.dtype)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)


class DecisionCoreChunk(nn.Module):
    def __init__(self, t_history=T_HISTORY, n_layers=N_LAYERS, n_heads=N_HEADS, dropout=0.1,
                 goal_dim=GOAL_DIM_PP, chunk=10, dec_layers=2, causal=False):
        super().__init__()
        self.goal_dim, self.chunk = goal_dim, chunk
        self.encoder = HistoryEncoder(D_MODEL, n_layers, n_heads, t_history, dropout, causal)
        self.grip_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, D_MODEL))
        self.query_proj = nn.Sequential(nn.Linear(goal_dim, 64), nn.GELU(), nn.Linear(64, D_MODEL))
        self.chunk_pos = nn.Parameter(torch.randn(1, chunk, D_MODEL) * 0.02)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(D_MODEL, n_heads, D_MODEL * FFN_MULT, dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(dec_layers)])
        self.out_ln = nn.LayerNorm(D_MODEL)
        self.z_head = nn.Linear(D_MODEL, D_MODEL)
        self.grip_head = nn.Sequential(nn.Linear(D_MODEL + goal_dim, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1))

    def forward(self, z_hist, finger_hist, goal_input):
        """z_hist (B,T,128), finger_hist (B,T) metres, goal_input (B,goal_dim)
        -> z_hat (B,K,128), gripper logits (B,K)"""
        mem = self.encoder(z_hist + self.grip_embed((finger_hist * FINGER_SCALE).unsqueeze(-1)))
        h = self.query_proj(goal_input).unsqueeze(1) + self.chunk_pos
        for layer in self.decoder:
            h = layer(h, mem)
        h = self.out_ln(h)
        g = goal_input.unsqueeze(1).expand(-1, self.chunk, -1)
        return self.z_head(h), self.grip_head(torch.cat([h, g], -1)).squeeze(-1)


def load_core(path, device="cpu"):
    """Any pick-and-place decision core: chunk checkpoints -> DecisionCoreChunk, else model.load_pp_core."""
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["config"]
    if c.get("chunk", 1) <= 1:
        return load_pp_core(path, device)
    m = DecisionCoreChunk(c["t_history"], c.get("n_layers", N_LAYERS), c.get("n_heads", N_HEADS),
                          goal_dim=c.get("goal_dim", GOAL_DIM_PP), chunk=c["chunk"],
                          dec_layers=c.get("dec_layers", 2), causal=c.get("causal", False)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, c


# =============================================================================================
# BASELINE: the same decision core on the TCP instead of the latent (end-effector policy + each
# arm's own IK). Same history encoder, chunk decoder, goal input, gripper head and data; only the
# per-step input (TCP position + rot6d, 9-d) and the output (next K TCP poses) differ.
# =============================================================================================
TCP_DIM = 9
POS_IN_SCALE = 2.0            # metres -> roughly unit range for the input projection
POS_OUT_SCALE = 0.05          # one output unit = 5 cm of TCP motion from the current TCP
ROT_OUT_SCALE = 0.2
TCP_PAD_NOISE = (0.005, 0.02)  # std of the noise on padded history rows: position (m), rot6d


def tcp_features(enc, q):
    """(B,7) or (B,6) joints of ANY arm -> (B,9) TCP [p, rot6d] in the shared world frame"""
    return enc.forward(q)[:, :TCP_DIM]


def gram_schmidt6(r6):
    a = r6[..., :3] / r6[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = r6[..., 3:6] - (a * r6[..., 3:6]).sum(-1, keepdim=True) * a
    return torch.cat([a, b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)], -1)


def pad_tcp(x0, n):
    """n noisy copies of the first TCP feature (same role as PADDING_NOISE_STD for latents)"""
    s = torch.tensor([TCP_PAD_NOISE[0]] * 3 + [TCP_PAD_NOISE[1]] * 6, device=x0.device)
    return [x0 + torch.randn_like(x0) * s for _ in range(n)]


class DecisionCoreTCP(nn.Module):
    def __init__(self, t_history=T_HISTORY, n_layers=N_LAYERS, n_heads=N_HEADS, dropout=0.1,
                 goal_dim=GOAL_DIM_PP, chunk=1, dec_layers=2, causal=True, hist_len=T_HISTORY):
        super().__init__()
        self.goal_dim, self.chunk = goal_dim, chunk
        self.hist_len = hist_len          # < T: the TCP of older steps is hidden (no velocity to copy)
        self.in_proj = nn.Sequential(nn.Linear(TCP_DIM, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.encoder = HistoryEncoder(D_MODEL, n_layers, n_heads, t_history, dropout, causal)
        self.grip_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, D_MODEL))
        self.query_proj = nn.Sequential(nn.Linear(goal_dim, 64), nn.GELU(), nn.Linear(64, D_MODEL))
        self.chunk_pos = nn.Parameter(torch.randn(1, chunk, D_MODEL) * 0.02)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(D_MODEL, n_heads, D_MODEL * FFN_MULT, dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(dec_layers)])
        self.out_ln = nn.LayerNorm(D_MODEL)
        self.tcp_head = nn.Linear(D_MODEL, TCP_DIM)
        self.grip_head = nn.Sequential(nn.Linear(D_MODEL + goal_dim, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1))

    def forward(self, tcp_hist, finger_hist, goal_input):
        """tcp_hist (B,T,9), finger_hist (B,T), goal_input (B,goal_dim) -> TCP (B,K,9), gripper logits (B,K)"""
        if self.hist_len < tcp_hist.shape[1]:          # replace older rows by the oldest kept one
            keep = tcp_hist[:, -self.hist_len:]
            tcp_hist = torch.cat([keep[:, :1].expand(-1, tcp_hist.shape[1] - self.hist_len, -1), keep], 1)
        x = torch.cat([tcp_hist[..., :3] * POS_IN_SCALE, tcp_hist[..., 3:]], -1)
        mem = self.encoder(self.in_proj(x) + self.grip_embed((finger_hist * FINGER_SCALE).unsqueeze(-1)))
        h = self.query_proj(goal_input).unsqueeze(1) + self.chunk_pos
        for layer in self.decoder:
            h = layer(h, mem)
        h = self.out_ln(h)
        o = self.tcp_head(h)
        last = tcp_hist[:, -1:]                                   # residual on the current TCP
        p = last[..., :3] + POS_OUT_SCALE * o[..., :3]
        r6 = gram_schmidt6(last[..., 3:] + ROT_OUT_SCALE * o[..., 3:])
        g = goal_input.unsqueeze(1).expand(-1, self.chunk, -1)
        return torch.cat([p, r6], -1), self.grip_head(torch.cat([h, g], -1)).squeeze(-1)


def load_any_core(path, device="cpu"):
    """latent single-step (model.py), latent chunk, or TCP baseline -> (model, config).
    config["input"] is "tcp" for the baseline, else "latent"; config["chunk"] >= 1."""
    ck = torch.load(path, map_location=device, weights_only=False)
    c = dict(ck["config"]); c.setdefault("input", "latent"); c.setdefault("chunk", 1)
    if c["input"] == "tcp":
        m = DecisionCoreTCP(c["t_history"], c.get("n_layers", N_LAYERS), c.get("n_heads", N_HEADS),
                            goal_dim=c.get("goal_dim", GOAL_DIM_PP), chunk=c["chunk"],
                            dec_layers=c.get("dec_layers", 2), causal=c.get("causal", True),
                            hist_len=c.get("hist_len", T_HISTORY)).to(device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        return m, c
    m, _ = load_core(path, device)
    return m, c


class ChunkRunner:
    """Turns per-step decision-core calls into the output for THIS step, for any model:
    single-step models (K=1) just pass through; chunk models use temporal ensembling (ACT) over all
    chunks that cover this step (or the latest chunk only), with a new chunk every `every` steps.
    predict() -> (out (K,D), p_close (K,))."""

    def __init__(self, K, every=1, ensemble=True, m=0.1):
        self.K, self.every, self.ensemble, self.m = K, every, ensemble, m
        self.buf = []

    def reset(self):
        self.buf = []

    def __call__(self, step, predict):
        self.buf = [c for c in self.buf if step - c[0] < self.K]
        if not self.buf or step % self.every == 0:
            o, p = predict()
            self.buf.append((step, o, p))
        use = self.buf if self.ensemble else self.buf[-1:]           # oldest first
        w = torch.tensor([float(np.exp(-self.m * i)) for i in range(len(use))], device=use[0][1].device)
        w = w / w.sum()
        out = sum(wi * c[1][step - c[0]] for wi, c in zip(w, use))
        p = sum(wi * c[2][step - c[0]] for wi, c in zip(w, use))
        return out, p


def call_core(core, cfg, hist, f_hist, goal_in):
    """one forward pass of any core -> (out (K,D), p_close (K,)) for batch element 0"""
    o, l = core(hist, f_hist, goal_in)
    if o.dim() == 2:                                             # single-step latent model (B,128)
        o, l = o[:, None], l[:, None]
    return o[0], torch.sigmoid(l[0])