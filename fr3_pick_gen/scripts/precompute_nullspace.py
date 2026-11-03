"""
Precompute null-space (elbow) variants of every expert episode ONCE, so decision-core
training does not pay for Jacobians / IK every batch.

For each episode and each α in ALPHAS:
  - every state q_t and action a_t is moved along FR3's null-space by α
    (TCP fixed, elbow rotates), direction signs kept continuous along the episode,
  - 3 Gauss-Newton steps restore the TCP exactly,
  - the variant is re-encoded with the frozen Stage-1 W_FR3.
Frames whose TCP moved > 2 mm or that hit a joint limit are marked invalid; the
dataset only uses a variant for a window if all its frames are valid.

Output: fr3_pick_gen/datasets/expert_v1/data_latent_ns/chunk-000/episode_*.parquet
        = the original columns + latent_ns{k}, state_ns{k}, action_ns{k}, valid_ns{k}

Run once:  python fr3_pick_gen/scripts/precompute_nullspace.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

R = os.path.expanduser("~/cross_embody")
sys.path.insert(0, R)
from decision_core.model import load_frozen_stage1

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", default="expert_v1", help="folder under fr3_pick_gen/datasets/")
_args = _ap.parse_args()
SRC = os.path.join(R, f"fr3_pick_gen/datasets/{_args.dataset}/data_latent/chunk-000")
DST = os.path.join(R, f"fr3_pick_gen/datasets/{_args.dataset}/data_latent_ns/chunk-000")
ALPHAS = [-1.0, -0.5, 0.5, 1.0]          # rad, joint-space norm of the null-space move
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LIM = torch.tensor([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
                    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169],
                    [-3.0159, 3.0159]], device=DEV)


def feat(fk, q):
    f = fk.forward(q)
    return torch.cat([f[:, :3], 0.3 * f[:, 3:9]], 1)


def jac(fk, q, eps=1e-4):
    n, d = q.shape
    qq = q.unsqueeze(1).repeat(1, d + 1, 1)
    qq[:, 1:] += eps * torch.eye(d, device=q.device)
    f = feat(fk, qq.reshape(-1, d)).view(n, d + 1, -1)
    return ((f[:, 1:] - f[:, :1]) / eps).transpose(1, 2), f[:, 0]


@torch.no_grad()
def null_dirs(fk, q):
    J, _ = jac(fk, q)
    v = torch.linalg.svd(J).Vh[:, -1, :]
    for t in range(1, len(v)):                       # continuous sign along the episode
        if (v[t] * v[t - 1]).sum() < 0:
            v[t] = -v[t]
    return v


@torch.no_grad()
def move(fk, q, v, alpha):
    f0 = feat(fk, q)
    q2 = q + alpha * v
    for _ in range(3):
        J, f2 = jac(fk, q2)
        q2 = q2 + (torch.linalg.pinv(J, rtol=1e-4) @ (f0 - f2).unsqueeze(-1)).squeeze(-1)
    clamped = ((q2 < LIM[:, 0]) | (q2 > LIM[:, 1])).any(1)
    q2 = torch.max(torch.min(q2, LIM[:, 1]), LIM[:, 0])
    res = (fk.forward(q2)[:, :3] - fk.forward(q)[:, :3]).norm(dim=1)
    return q2, (res < 0.002) & ~clamped


@torch.no_grad()
def main():
    fk, emb, _, _ = load_frozen_stage1(device=DEV)
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "episode_*.parquet")))
    stats = np.zeros(len(ALPHAS))
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        st = np.stack(df["observation.state"].values).astype(np.float32)
        ac = np.stack(df["action"].values).astype(np.float32)
        q = torch.tensor(st[:, :7], device=DEV)
        a = torch.tensor(ac[:, :7], device=DEV)
        vq = null_dirs(fk, q)
        Ja, _ = jac(fk, a)
        va = torch.linalg.svd(Ja).Vh[:, -1, :]
        va = va * torch.where((va * vq).sum(1, keepdim=True) < 0, -1.0, 1.0)   # same side as the state
        for k, alpha in enumerate(ALPHAS):
            q2, okq = move(fk, q, vq, alpha)
            a2, oka = move(fk, a, va, alpha)
            z2 = emb(fk.forward(q2)[:, :18])
            st2, ac2 = st.copy(), ac.copy()
            st2[:, :7] = q2.cpu().numpy(); ac2[:, :7] = a2.cpu().numpy()
            df[f"latent_ns{k}"] = list(z2.cpu().numpy())
            df[f"state_ns{k}"] = list(st2)
            df[f"action_ns{k}"] = list(ac2)
            df[f"valid_ns{k}"] = (okq & oka).cpu().numpy()
            stats[k] += (okq & oka).float().mean().item()
        df.to_parquet(os.path.join(DST, os.path.basename(f)))
        if i % 50 == 0:
            print(f"[{i + 1}/{len(files)}] {os.path.basename(f)}")
    print("valid frame fraction per α:",
          {a: f"{s / len(files):.1%}" for a, s in zip(ALPHAS, stats)})
    print(f"done -> {DST}")


if __name__ == "__main__":
    main()