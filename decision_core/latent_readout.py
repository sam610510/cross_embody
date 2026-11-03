"""
Frozen READOUT of the shared latent: z (128) -> TCP position (3) + 6D orientation (6) of the FR3.

Only used to MEASURE / train in task space (mm): the decision core still outputs the shared latent, and
deployment still decodes it with inverse_decode + tcp_ik. The latent is an MLP embedding of FK features,
so "128-d latent error" and "mm of TCP error" are not the same thing -- this readout gives the second.

Fit once (about a minute on the GPU):
  python decision_core/latent_readout.py --fit
-> checkpoints/latent_readout_fr3.pt   (prints the held-out position error; should be well below 1 mm)
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn

R = os.path.expanduser("~/cross_embody")
DEFAULT = os.path.join(R, "checkpoints", "latent_readout_fr3.pt")
FR3_LO = [-2.9, -1.8, -2.9, -3.07, -2.87, 0.44, -3.05]
FR3_HI = [2.9, 1.8, 2.9, -0.12, 2.87, 4.62, 3.05]


class LatentReadout(nn.Module):
    def __init__(self, latent=128, hidden=512, out=9):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))

    def forward(self, z):
        return self.net(z)                      # [..., :3] TCP position (m), [..., 3:9] 6D rotation


def load_readout(path=DEFAULT, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    r = LatentReadout(ck["latent"], ck["hidden"]).to(device)
    r.load_state_dict(ck["state_dict"]); r.eval()
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def fit(args):
    sys.path[:0] = [R, os.path.join(R, "fr3_pick_gen", "scripts")]
    import pandas as pd
    from decision_core.model import load_frozen_stage1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fk, emb, _, _ = load_frozen_stage1(device=dev)
    lo, hi = torch.tensor(FR3_LO, device=dev), torch.tensor(FR3_HI, device=dev)
    qs = []                                     # real workspace poses from the datasets (+ random ones below)
    for d in args.datasets:
        for f in sorted(glob.glob(os.path.join(R, "fr3_pick_gen", "datasets", d, "data", "chunk-000", "episode_*.parquet"))):
            qs.append(np.stack(pd.read_parquet(f, columns=["observation.state"])["observation.state"].values)[:, :7])
    Qd = torch.tensor(np.concatenate(qs), dtype=torch.float32, device=dev) if qs else torch.zeros(0, 7, device=dev)
    print(f"[readout] {len(Qd)} dataset poses + random poses, device {dev}")
    net = LatentReadout(128, args.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    def batch(n):
        nd = n // 2 if len(Qd) else 0
        q = torch.cat([Qd[torch.randint(len(Qd), (nd,), device=dev)] + 0.05 * torch.randn(nd, 7, device=dev),
                       lo + (hi - lo) * torch.rand(n - nd, 7, device=dev)]).clamp(lo, hi) if nd else \
            lo + (hi - lo) * torch.rand(n, 7, device=dev)
        with torch.no_grad():
            f = fk.forward(q)
            return emb(f[:, :18]), f[:, :9]

    for it in range(args.steps):
        z, y = batch(args.batch)
        p = net(z)
        loss = ((p[:, :3] - y[:, :3]) * 100).pow(2).sum(-1).mean() + (p[:, 3:] - y[:, 3:]).pow(2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 1000 == 0 or it == args.steps - 1:
            with torch.no_grad():
                z, y = batch(20000); p = net(z)
                e = (p[:, :3] - y[:, :3]).norm(dim=-1) * 1000
            print(f"  step {it:6d}  held-out TCP position error: median {e.median():.3f} mm, p99 {e.quantile(.99):.3f} mm, "
                  f"6D rot err {(p[:, 3:] - y[:, 3:]).norm(dim=-1).median():.4f}", flush=True)
    torch.save({"state_dict": net.state_dict(), "latent": 128, "hidden": args.hidden}, args.out)
    print(f"[readout] -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=["expert_pp_v3", "expert_pp_v3_dag1", "expert_pp_v3_dag2"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--out", default=DEFAULT)
    a = ap.parse_args()
    if a.fit:
        fit(a)


# ---------------------------------------------------------------------------------------------------------
# EXACT local readout (no training): the latent is z = emb(f), f = FK features (18; f[:3] = TCP position).
# Around the true pose, a latent error dz maps to an FK-feature error df = (J^T J)^-1 J^T dz with
# J = d emb / d f (128 x 18) at f_true -- first-order exact, so "mm of TCP" without a learned network.
def latent_to_fk_projector(emb, f_true):
    """f_true (N, 18) -> P (N, 18, 128) such that df ~= P @ dz (no grad; depends only on the true pose)"""
    from torch.func import jacfwd, vmap
    with torch.no_grad():
        J = vmap(jacfwd(lambda x: emb(x[None])[0]))(f_true)                  # (N, 128, 18)
        Jt = J.transpose(1, 2)
        JtJ = (Jt @ J).double().cpu() + 1e-9 * torch.eye(J.shape[-1], dtype=torch.float64)
        # small 18x18 systems: solve on the CPU (the batched MAGMA solver on the GPU crashed with an illegal
        # memory access on this CUDA build), then multiply on the device
        inv = torch.linalg.inv(JtJ).to(J.device, J.dtype)
        return inv @ Jt                                                      # (N, 18, 128)


def tcp_error(emb, f_true, z_hat, z_true):
    """TCP position error vector (..., 3) in metres implied by the latent error z_hat - z_true"""
    shp = z_hat.shape[:-1]
    P = latent_to_fk_projector(emb, f_true.reshape(-1, f_true.shape[-1]))[:, :3]   # (N, 3, 128)
    dz = (z_hat - z_true).reshape(-1, z_hat.shape[-1], 1)
    return (P @ dz).squeeze(-1).reshape(*shp, 3)


def fk_error(emb, f_true, z_hat, z_true):
    """FK-feature error vector (..., 18) implied by the latent error: [:3] TCP position (m), [3:9] 6D rotation"""
    shp = z_hat.shape[:-1]
    P = latent_to_fk_projector(emb, f_true.reshape(-1, f_true.shape[-1]))          # (N, 18, 128)
    dz = (z_hat - z_true).reshape(-1, z_hat.shape[-1], 1)
    return (P @ dz).squeeze(-1).reshape(*shp, P.shape[1])


def rot6_angle(d6):
    """approx. rotation angle (rad) from the error of the 6D rotation (two unit columns): RMS of the column errors"""
    return torch.sqrt(0.5 * (d6[..., :3].pow(2).sum(-1) + d6[..., 3:6].pow(2).sum(-1)) + 1e-12)