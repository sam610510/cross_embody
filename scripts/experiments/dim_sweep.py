"""
latent_dim_sweep.py — 潜在维度消融 (单变量, 同条件)
====================================================
只变 latent 维度, 其他全部固定为最终架构:
  - 去 m (embedding 输入 18 维)
  - 固定噪声 0.03 (与正式训练一致)
  - 300 epochs + CosineAnnealing
  - decoder depth=2, hidden=256

扫描: latent = 32 / 64 / 128 / 256 / 512 / 1024
输出单张双 y 轴图: 末端误差(左轴) + reliance(右轴) vs 维度。
"""
import os, sys, time, csv
import numpy as np
import torch

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import MLPEmbedding
from layer2.decoder import Decoder

ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROBOT_NAME = "fr3"; ARM_DOF = 7
EMB_IN = 18
N_SAMPLES = 50000; BATCH = 512; LR = 1e-3; EPOCHS = 300
ALPHA = 0.1; BETA = 1.0
N_EVAL = 2000
Q_NOISE = 0.03

LATENT_DIMS = [32, 64, 128, 256, 512, 1024]

JOINT_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
], dtype=np.float32)


def sample_q(n):
    lo = JOINT_LIMITS[:, 0]; hi = JOINT_LIMITS[:, 1]
    u = np.random.uniform(0, 1, (n, ARM_DOF)).astype(np.float32)
    return lo + u * (hi - lo)


def run_one(latent_dim, enc, m_const, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    W = MLPEmbedding(EMB_IN, latent_dim).to(DEVICE)
    dec = Decoder(latent_dim, m_dim=3, out_dim=ARM_DOF, hidden=256,
                  depth=2, use_q_prev=True).to(DEVICE)
    opt = torch.optim.Adam(list(W.parameters()) + list(dec.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Q = torch.from_numpy(sample_q(N_SAMPLES)).to(DEVICE)
    n_batches = (N_SAMPLES + BATCH - 1) // BATCH

    W.train(); dec.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(N_SAMPLES, device=DEVICE)
        for b in range(n_batches):
            q = Q[perm[b*BATCH:(b+1)*BATCH]]; B = q.shape[0]
            fk21 = enc.forward(q)
            z = W(fk21[:, :EMB_IN])
            q_ref = q + torch.randn_like(q) * Q_NOISE
            q_pred = dec(z, m_const.expand(B, -1), q_prev=q_ref)
            loss_joint = ((q_pred - q) ** 2).mean()
            loss_ee = ((enc.forward(q_pred)[:, :EMB_IN] - fk21[:, :EMB_IN].detach()) ** 2).mean()
            loss = ALPHA * loss_joint + BETA * loss_ee
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(W.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
        sched.step()

    W.eval(); dec.eval()
    with torch.no_grad():
        Qv = torch.from_numpy(sample_q(N_EVAL)).to(DEVICE)
        fk = enc.forward(Qv); z = W(fk[:, :EMB_IN]); mb = m_const.expand(N_EVAL, -1)
        base_err = torch.norm(enc.forward(dec(z, mb, q_prev=Qv))[:, :3] - fk[:, :3], dim=1) * 1000
        z_shuf = z[torch.randperm(N_EVAL, device=DEVICE)]
        shuf_err = torch.norm(enc.forward(dec(z_shuf, mb, q_prev=Qv))[:, :3] - fk[:, :3], dim=1) * 1000
    return {
        "latent_dim": latent_dim,
        "ee_mm": base_err.mean().item(),
        "ee_median": base_err.median().item(),
        "ee_p99": torch.quantile(base_err, 0.99).item(),
        "reliance": (shuf_err.mean() / base_err.mean()).item(),
    }


def main():
    print(f"[init] device={DEVICE}  epochs={EPOCHS}  dims={LATENT_DIMS}")
    print(f"[config] 去m, 固定噪声 {Q_NOISE} (与正式训练一致)")
    enc = FKEncoder(ROBOT_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    m_const = enc.m.to(DEVICE).unsqueeze(0)

    rows = []
    t0 = time.time()
    for i, ld in enumerate(LATENT_DIMS):
        t = time.time()
        r = run_one(ld, enc, m_const)
        rows.append(r)
        print(f"[{i+1}/{len(LATENT_DIMS)}] latent={ld:4d}  "
              f"EE={r['ee_mm']:6.2f}mm  median={r['ee_median']:6.2f}  "
              f"p99={r['ee_p99']:6.2f}  reliance=x{r['reliance']:5.1f}   ({time.time()-t:.0f}s)")
    print(f"[done] total {time.time()-t0:.0f}s")

    _save_csv(rows, os.path.join(OUT_DIR, "latent_dim_sweep.csv"))
    _plot(rows, os.path.join(OUT_DIR, "latent_dim_sweep.png"))

    best = min(rows, key=lambda r: r["ee_mm"])
    print(f"\n最低末端误差: latent={best['latent_dim']} ({best['ee_mm']:.2f}mm)")
    print("趋势 (末端误差 vs 维度):")
    for r in rows:
        bar = "#" * int(r["ee_mm"])
        print(f"  {r['latent_dim']:4d}: {r['ee_mm']:6.2f}mm  {bar}")


def _save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["latent_dim","ee_mm","ee_median","ee_p99","reliance"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] saved {os.path.basename(path)}")


def _plot(rows, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = [r["latent_dim"] for r in rows]
    ee   = [r["ee_mm"] for r in rows]
    rel  = [r["reliance"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # 左轴: 末端误差
    c_ee = "#B5651D"
    l1 = ax1.plot(dims, ee, "o-", color=c_ee, lw=2, markersize=8, label="EE error (mm)")
    ax1.set_xlabel("latent dimension")
    ax1.set_ylabel("EE position error (mm)", color=c_ee)
    ax1.tick_params(axis="y", labelcolor=c_ee)
    for d, e in zip(dims, ee):
        ax1.annotate(f"{e:.1f}", (d, e), color=c_ee, fontsize=8,
                     xytext=(0, 8), textcoords="offset points", ha="center")
    best_i = int(np.argmin(ee))
    ax1.scatter([dims[best_i]], [ee[best_i]], s=180, facecolors="none",
                edgecolors="#2C6B3F", linewidths=2.5, zorder=5)

    # 右轴: reliance
    c_rel = "#2C6B3F"
    ax2 = ax1.twinx()
    l2 = ax2.plot(dims, rel, "s--", color=c_rel, lw=2, markersize=8, label="latent reliance (x)")
    ax2.set_ylabel("latent reliance (x)", color=c_rel)
    ax2.tick_params(axis="y", labelcolor=c_rel)
    for d, rl in zip(dims, rel):
        ax2.annotate(f"x{rl:.0f}", (d, rl), color=c_rel, fontsize=8,
                     xytext=(0, -14), textcoords="offset points", ha="center")

    ax1.set_xscale("log", base=2); ax1.set_xticks(dims); ax1.set_xticklabels(dims)
    ax1.set_title("End-effector error and latent reliance vs latent dimension",
                  fontweight="bold")
    ax1.grid(alpha=0.3)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")

    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"[done] saved {os.path.basename(path)}")


if __name__ == "__main__":
    main()