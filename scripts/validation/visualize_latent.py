"""
visualize_latent.py — 把一段轨迹编码到 64 维 latent, 用 PCA(3D) + UMAP 看空间变化
==================================================================================
载入 Stage 1 训练好的 W (没有就用未训练的 W, 自动退化到路线 B)。
一段轨迹 (demo pose 为中心摆动) -> FK Encoder(21) -> W(64) -> latent 序列
然后:
  - PCA 降 3D: 画轨迹曲线 (时间用颜色渐层), 看是否平滑移动
  - PCA 解释变异比例: 看 64 维变化集中在前几维
  - UMAP 2D: 看非线性结构 / 聚类

用法: python scripts/visualize_latent.py
输出: validation_out/latent_pca3d.png, latent_pca_variance.png, latent_umap.png
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import Embedding, MLPEmbedding

ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
ROBOT_NAME = "fr3"; ARM_DOF = 7; LATENT_DIM = 128
CKPT = os.path.join(PROJECT_ROOT, "checkpoints/stage1_franka.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEMO_POSE = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float32)
N_TRAJ = 300   # 轨迹点数 (够密看平滑)


def make_trajectory(n):
    """demo pose 为中心, 各关节相位错开摆动 (与 render/validate 一致)。"""
    t = np.linspace(0, 2*np.pi, n)
    traj = np.stack([DEMO_POSE[k] + 0.4*np.sin(t + k) for k in range(ARM_DOF)], axis=1)
    return traj.astype(np.float32), t


def main():
    enc = FKEncoder(ROBOT_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    W = MLPEmbedding(18, LATENT_DIM).to(DEVICE)   # 输入 18 (p_ee+v_kp, 不含 m)  # 与 train_stage1 (Z2) 一致

    trained = False
    if os.path.isfile(CKPT):
        ck = torch.load(CKPT, map_location=DEVICE)
        W.load_state_dict(ck["W"])
        trained = True
        print(f"[init] loaded trained W from {CKPT}")
    else:
        print(f"[init] 未找到 checkpoint, 用未训练 W (路线 B: 仅看轨迹连续性)")

    traj, t = make_trajectory(N_TRAJ)
    with torch.no_grad():
        q = torch.from_numpy(traj).to(DEVICE)
        fk21 = enc.forward(q)
        Z = W(fk21[:, :18]).cpu().numpy()   # 只喂 p_ee+v_kp
    print(f"[data] encoded trajectory -> latent {Z.shape}")

    tag = "trained" if trained else "untrained"
    _pca_3d(Z, t, os.path.join(OUT_DIR, "latent_pca3d.png"), tag)
    _pca_variance(Z, os.path.join(OUT_DIR, "latent_pca_variance.png"), tag)
    _umap_2d(Z, t, os.path.join(OUT_DIR, "latent_umap.png"), tag)
    print(f"[done] outputs in {OUT_DIR}")


def _pca_3d(Z, t, path, tag):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [skip] sklearn 未安装: pip install scikit-learn")
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    P = PCA(n_components=3).fit_transform(Z)
    fig = plt.figure(figsize=(8,7)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P[:,0], P[:,1], P[:,2], c=t, cmap="viridis", s=18)
    ax.plot(P[:,0], P[:,1], P[:,2], color="gray", alpha=0.4, linewidth=1)
    fig.colorbar(sc, ax=ax, label="trajectory time", shrink=0.6)
    ax.set_title(f"Latent trajectory in PCA-3D ({tag} W)", fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  -> latent_pca3d.png")


def _pca_variance(Z, path, tag):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [skip] sklearn 未安装: pip install scikit-learn")
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = PCA().fit(Z)
    ev = p.explained_variance_ratio_
    k = min(20, len(ev))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(range(1, k+1), ev[:k], color="#6B4C7A", alpha=0.8, label="per-PC")
    ax.plot(range(1, k+1), np.cumsum(ev[:k]), "-o", color="#B5651D", label="cumulative")
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio")
    ax.set_title(f"PCA explained variance ({tag} W)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> latent_pca_variance.png  (前3维占 {ev[:3].sum()*100:.1f}%)")


def _umap_2d(Z, t, path, tag):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import umap
    except ImportError:
        print("  [skip] UMAP 未安装: pip install umap-learn")
        return
    reducer = umap.UMAP(n_components=2, random_state=0)
    U = reducer.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(U[:,0], U[:,1], c=t, cmap="viridis", s=18)
    ax.plot(U[:,0], U[:,1], color="gray", alpha=0.4, linewidth=1)
    fig.colorbar(sc, ax=ax, label="trajectory time")
    ax.set_title(f"Latent trajectory in UMAP-2D ({tag} W)", fontweight="bold")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  -> latent_umap.png")


if __name__ == "__main__":
    main()