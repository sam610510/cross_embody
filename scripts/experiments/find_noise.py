"""
noise_sweep.py — Stage 1 噪声策略扫描 (为 noise 选择提供实验依据)
==================================================================
问题: q_ref 的噪声该设多少? 太小 decoder 抄 q_ref (作弊), 太大精度掉。
方法: 扫描多种噪声策略, 同时量测两个指标, 画出 trade-off 曲线。

两个指标:
  1. EE position error (mm)   — 精度 (越低越好)
  2. Latent reliance (×)      — decoder 对 latent 的依赖度 (越高越好)
     定义: 打乱 latent 后的误差 / 基线误差。×1 = 完全没用 latent (抄 q_ref)

扫描三组:
  A. 固定噪声:      0.005 / 0.01 / 0.02 / 0.03 / 0.05 / 0.08
  B. 混合噪声:      base 0.005 + heavy 0.03, heavy_prob = 0.25 / 0.50 / 0.75
  C. 均匀随机噪声:  uniform(0.005, 0.03)

选择准则 (事先定义, 避免事后找理由):
  在满足 EE error < TARGET_MM 的设定中, 选 latent reliance 最高者。

用法:
  python scripts/noise_sweep.py              # 完整扫描 (较久)
  python scripts/noise_sweep.py --quick      # 快速版 (少 epoch, 先看趋势)

输出:
  validation_out/noise_sweep.png   — trade-off 曲线图
  validation_out/noise_sweep.csv   — 原始数据
"""
import os, sys, time, argparse, csv
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
EMB_IN = 18; LATENT_DIM = 128
N_SAMPLES = 50000; BATCH = 512; LR = 1e-3
ALPHA = 0.1; BETA = 1.0
N_EVAL = 2000
TARGET_MM = 10.0          # 精度硬性要求 (来自 DTG < 1cm 目标)

JOINT_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
], dtype=np.float32)


# ---------- 噪声策略 ----------
class NoiseStrategy:
    """产生 q_ref = q + noise。不同策略只差在 noise 怎么来。"""
    def __init__(self, kind, label, **kw):
        self.kind = kind; self.label = label; self.kw = kw

    def apply(self, q):
        if self.kind == "fixed":
            return q + torch.randn_like(q) * self.kw["sigma"]
        if self.kind == "mixed":
            base = q + torch.randn_like(q) * self.kw["base"]
            heavy = q + torch.randn_like(q) * self.kw["heavy"]
            mask = (torch.rand(q.shape[0], 1, device=q.device) < self.kw["p"])
            return torch.where(mask, heavy, base)
        if self.kind == "uniform":
            lo, hi = self.kw["lo"], self.kw["hi"]
            sigma = torch.rand(q.shape[0], 1, device=q.device) * (hi - lo) + lo
            return q + torch.randn_like(q) * sigma
        raise ValueError(self.kind)


def build_strategies():
    S = []
    for s in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
        S.append(NoiseStrategy("fixed", f"fixed {s}", sigma=s))
    for p in [0.25, 0.50, 0.75]:
        S.append(NoiseStrategy("mixed", f"mixed p={p}", base=0.005, heavy=0.03, p=p))
    S.append(NoiseStrategy("uniform", "uniform .005-.03", lo=0.005, hi=0.03))
    return S


def sample_q(n):
    lo = JOINT_LIMITS[:, 0]; hi = JOINT_LIMITS[:, 1]
    u = np.random.uniform(0, 1, (n, ARM_DOF)).astype(np.float32)
    return lo + u * (hi - lo)


# ---------- 单次训练 + 评估 ----------
def run_one(strategy, enc, m_const, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    W = MLPEmbedding(EMB_IN, LATENT_DIM).to(DEVICE)
    dec = Decoder(LATENT_DIM, m_dim=3, out_dim=ARM_DOF, hidden=256,
                  depth=2, use_q_prev=True).to(DEVICE)
    opt = torch.optim.Adam(list(W.parameters()) + list(dec.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Q = torch.from_numpy(sample_q(N_SAMPLES)).to(DEVICE)
    n_batches = (N_SAMPLES + BATCH - 1) // BATCH

    W.train(); dec.train()
    for ep in range(epochs):
        perm = torch.randperm(N_SAMPLES, device=DEVICE)
        for b in range(n_batches):
            q = Q[perm[b*BATCH:(b+1)*BATCH]]; B = q.shape[0]
            fk21 = enc.forward(q)
            z = W(fk21[:, :EMB_IN])
            q_ref = strategy.apply(q)
            q_pred = dec(z, m_const.expand(B, -1), q_prev=q_ref)
            loss_joint = ((q_pred - q) ** 2).mean()
            loss_ee = ((enc.forward(q_pred)[:, :EMB_IN] - fk21[:, :EMB_IN].detach()) ** 2).mean()
            loss = ALPHA * loss_joint + BETA * loss_ee
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(W.parameters()) + list(dec.parameters()), 1.0)
            opt.step()
        sched.step()

    # ---- 评估: EE 误差 + latent reliance ----
    W.eval(); dec.eval()
    with torch.no_grad():
        Qv = torch.from_numpy(sample_q(N_EVAL)).to(DEVICE)
        fk = enc.forward(Qv); z = W(fk[:, :EMB_IN]); mb = m_const.expand(N_EVAL, -1)

        base_err = torch.norm(enc.forward(dec(z, mb, q_prev=Qv))[:, :3] - fk[:, :3], dim=1) * 1000
        z_shuf = z[torch.randperm(N_EVAL, device=DEVICE)]
        shuf_err = torch.norm(enc.forward(dec(z_shuf, mb, q_prev=Qv))[:, :3] - fk[:, :3], dim=1) * 1000

    ee = base_err.mean().item()
    reliance = (shuf_err.mean() / base_err.mean()).item()
    return ee, reliance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="少 epoch 快速看趋势")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    epochs = args.epochs if args.epochs else (60 if args.quick else 300)

    print(f"[init] device={DEVICE}  epochs={epochs}")
    enc = FKEncoder(ROBOT_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    m_const = enc.m.to(DEVICE).unsqueeze(0)

    strategies = build_strategies()
    rows = []
    t0 = time.time()
    for i, st in enumerate(strategies):
        t = time.time()
        ee, rel = run_one(st, enc, m_const, epochs)
        rows.append({"label": st.label, "kind": st.kind, "ee_mm": ee, "reliance": rel})
        print(f"[{i+1}/{len(strategies)}] {st.label:20s} "
              f"EE={ee:6.2f}mm  reliance=x{rel:6.2f}   ({time.time()-t:.0f}s)")
    print(f"[done] total {time.time()-t0:.0f}s")

    _save_csv(rows, os.path.join(OUT_DIR, "noise_sweep.csv"))
    _plot(rows, os.path.join(OUT_DIR, "noise_sweep.png"))
    _recommend(rows)


def _recommend(rows):
    ok = [r for r in rows if r["ee_mm"] < TARGET_MM]
    print("\n" + "="*66)
    print(f"选择准则: 在 EE error < {TARGET_MM}mm 的设定中, 选 latent reliance 最高者")
    print("="*66)
    if not ok:
        best = min(rows, key=lambda r: r["ee_mm"])
        print(f"  无设定满足 <{TARGET_MM}mm。精度最佳: {best['label']} "
              f"(EE={best['ee_mm']:.2f}mm, reliance=x{best['reliance']:.1f})")
        return
    best = max(ok, key=lambda r: r["reliance"])
    print(f"  满足精度的设定: {len(ok)}/{len(rows)}")
    for r in sorted(ok, key=lambda r: -r["reliance"]):
        mark = " <== 推荐" if r is best else ""
        print(f"    {r['label']:20s} EE={r['ee_mm']:6.2f}mm  reliance=x{r['reliance']:6.2f}{mark}")


def _save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "kind", "ee_mm", "reliance"])
        w.writeheader(); w.writerows(rows)
    print(f"[done] saved {os.path.basename(path)}")


def _plot(rows, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = [r for r in rows if r["kind"] == "fixed"]
    mixed = [r for r in rows if r["kind"] == "mixed"]
    unif  = [r for r in rows if r["kind"] == "uniform"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: 固定噪声下 两指标 vs sigma (双 y 轴, 显示 trade-off)
    sig = [float(r["label"].split()[1]) for r in fixed]
    ee  = [r["ee_mm"] for r in fixed]
    rel = [r["reliance"] for r in fixed]
    a0 = ax[0]
    l1 = a0.plot(sig, ee, "o-", color="#B5651D", lw=2, label="EE error (mm)")
    a0.set_xlabel("q_ref noise σ"); a0.set_ylabel("EE position error (mm)", color="#B5651D")
    a0.tick_params(axis="y", labelcolor="#B5651D")
    a0.axhline(TARGET_MM, ls="--", color="gray", lw=1)
    a0.text(sig[0], TARGET_MM*1.03, f"target {TARGET_MM:.0f}mm", fontsize=8, color="gray")
    a0b = a0.twinx()
    l2 = a0b.plot(sig, rel, "s-", color="#2C6B3F", lw=2, label="latent reliance (x)")
    a0b.set_ylabel("latent reliance  (shuffled / baseline)", color="#2C6B3F")
    a0b.tick_params(axis="y", labelcolor="#2C6B3F")
    a0.set_title("Fixed noise: accuracy vs latent reliance trade-off", fontweight="bold")
    a0.grid(alpha=0.3)
    lines = l1 + l2; a0.legend(lines, [l.get_label() for l in lines], loc="center right")

    # 右: 全部策略散点 (EE vs reliance), 右下角为佳
    a1 = ax[1]
    for grp, c, mk in [(fixed, "#B5651D", "o"), (mixed, "#2C6B3F", "s"), (unif, "#6B4C7A", "^")]:
        if not grp: continue
        a1.scatter([r["ee_mm"] for r in grp], [r["reliance"] for r in grp],
                   c=c, marker=mk, s=90, label=grp[0]["kind"], zorder=3)
        for r in grp:
            a1.annotate(r["label"], (r["ee_mm"], r["reliance"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    a1.axvline(TARGET_MM, ls="--", color="gray", lw=1)
    a1.text(TARGET_MM*1.02, a1.get_ylim()[1]*0.95, f"target {TARGET_MM:.0f}mm",
            fontsize=8, color="gray", rotation=90, va="top")
    a1.set_xlabel("EE position error (mm)   ← better")
    a1.set_ylabel("latent reliance (x)   better ↑")
    a1.set_title("All strategies (upper-left = accurate + latent-reliant)", fontweight="bold")
    a1.legend(); a1.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"[done] saved {os.path.basename(path)}")


if __name__ == "__main__":
    main()