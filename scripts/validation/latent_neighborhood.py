"""
latent_neighborhood.py — latent 邻域一致性检查
================================================
验证 latent space 映射品质: latent 空间里靠近的点, 解码出的末端位置也该靠近。
随机取大量 latent 对, 比较「latent 距离」vs「末端距离」, 算相关性。

好的 latent: 两者正相关强 (近的 latent -> 近的末端), 散点沿对角线分布。
差的 latent: 无相关 (散点乱), 表示 latent 空间没有保持几何结构。

用法: python scripts/latent_neighborhood.py
输出: validation_out/latent_neighborhood.png (散点图 + 相关系数)
"""
import os, sys
import numpy as np
import torch

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import MLPEmbedding

ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
ROBOT_NAME = "fr3"; ARM_DOF = 7
CKPT = os.path.join(PROJECT_ROOT, "checkpoints/stage1_franka.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SAMPLES = 3000     # 取多少姿态
N_PAIRS = 8000       # 随机配多少对

JOINT_LIMITS = np.array([
    [-2.7437,2.7437],[-1.7837,1.7837],[-2.9007,2.9007],
    [-3.0421,-0.1518],[-2.8065,2.8065],[0.5445,4.5169],[-3.0159,3.0159],
], dtype=np.float32)


def sample_q(n):
    lo=JOINT_LIMITS[:,0]; hi=JOINT_LIMITS[:,1]
    return (lo + np.random.uniform(0,1,(n,ARM_DOF)).astype(np.float32)*(hi-lo))


def main():
    enc = FKEncoder(ROBOT_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    LD = ck["latent_dim"]
    W = MLPEmbedding(18, LD).to(DEVICE)   # 输入 18 (不含 m)
    W.load_state_dict(ck["W"]); W.eval()
    print(f"[init] latent_dim={LD}, loaded {CKPT}")

    with torch.no_grad():
        Q = torch.from_numpy(sample_q(N_SAMPLES)).to(DEVICE)
        fk = enc.forward(Q)
        Z = W(fk[:, :18]).cpu().numpy()     # 只喂 p_ee+v_kp
        FULL = fk[:, :18].cpu().numpy()     # (N,18) 完整几何: p_ee(9)+v_kp(9), 不含 m 常数
        VKP = fk[:, 9:18].cpu().numpy()     # (N,9) 手臂姿态 v_kp
        EE = fk[:, :3].cpu().numpy()        # (N,3) 末端位置

    i = np.random.randint(0, N_SAMPLES, N_PAIRS)
    j = np.random.randint(0, N_SAMPLES, N_PAIRS)
    mask = i != j; i, j = i[mask], j[mask]
    d_latent = np.linalg.norm(Z[i] - Z[j], axis=1)
    d_full = np.linalg.norm(FULL[i] - FULL[j], axis=1)      # 完整构型距离 (你要的)
    d_vkp  = np.linalg.norm(VKP[i] - VKP[j], axis=1)
    d_ee   = np.linalg.norm(EE[i] - EE[j], axis=1) * 1000   # mm

    from scipy.stats import spearmanr
    def corr(a, b): return np.corrcoef(a,b)[0,1], spearmanr(a,b).correlation

    p_full, s_full = corr(d_latent, d_full)
    p_vkp,  s_vkp  = corr(d_latent, d_vkp)
    p_ee,   s_ee   = corr(d_latent, d_ee)

    print("[result] latent 距离 vs ...  (Pearson / Spearman)")
    print(f"  完整构型(21维 p_ee+v_kp) : {p_full:.4f} / {s_full:.4f}  <- 你要的: 完整构型保留, 这条高才好")
    print(f"  手臂姿态(v_kp 9维)       : {p_vkp:.4f} / {s_vkp:.4f}")
    print(f"  末端位置(3维, 对照)      : {p_ee:.4f} / {s_ee:.4f}  <- 之前只看这条 (过窄)")

    _plot_multi(d_latent, d_full, d_vkp, d_ee,
                (p_full,s_full),(p_vkp,s_vkp),(p_ee,s_ee),
                os.path.join(OUT_DIR, "latent_neighborhood.png"), LD)
    print(f"[done] saved latent_neighborhood.png")


def _plot_multi(d_latent, d_full, d_vkp, d_ee, cf, cv, ce, path, LD):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    specs = [
        (d_full, "full geometry |Δ(p_ee+v_kp)|", cf, "#2C6B3F", "full geometry"),
        (d_vkp,  "arm pose |Δv_kp|",             cv, "#6B4C7A", "arm pose"),
        (d_ee,   "end-effector |Δp| (mm)",       ce, "#B5651D", "end-effector"),
    ]
    for k,(d,lab,c,color,title) in enumerate(specs):
        ax[k].scatter(d_latent, d, s=4, alpha=0.22, color=color)
        ax[k].set_xlabel("latent distance |z_i - z_j|")
        ax[k].set_ylabel(lab)
        ax[k].set_title(f"{title}\nPearson={c[0]:.3f} Spearman={c[1]:.3f}", fontweight="bold")
        ax[k].grid(alpha=0.3)
    fig.suptitle(f"Latent neighborhood consistency (latent={LD})", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()