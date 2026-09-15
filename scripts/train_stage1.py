import os, sys
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import Embedding, MLPEmbedding
from layer2.decoder import Decoder

ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
ROBOT_NAME = "fr3"; ARM_DOF = 7
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

JOINT_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
], dtype=np.float32)

N_SAMPLES = 50000; LATENT_DIM = 128
EPOCHS = 300; BATCH = 512; LR = 1e-3
ALPHA = 0.1; BETA = 1.0            # 末端项主导 (joint 项被 IK 多解卡死, 降权)
Q_NOISE = 0.03                     # 固定噪声 (noise_sweep 显示精度/reliance 最佳折衷)
EMB_IN = 18                        # embedding 只吃 p_ee+v_kp, 去掉 m (m 是机器人身份)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def sample_q(n):
    lo = JOINT_LIMITS[:, 0]; hi = JOINT_LIMITS[:, 1]
    u = np.random.uniform(0, 1, size=(n, ARM_DOF)).astype(np.float32)
    return lo + u * (hi - lo)


def main():
    print(f"[init] device = {DEVICE}")
    enc = FKEncoder(ROBOT_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    m_const = enc.m.to(DEVICE).unsqueeze(0)

    W = MLPEmbedding(EMB_IN, LATENT_DIM).to(DEVICE)
    dec = Decoder(LATENT_DIM, m_dim=3, out_dim=ARM_DOF, hidden=256,
                  depth=2, use_q_prev=True).to(DEVICE)
    opt = torch.optim.Adam(list(W.parameters()) + list(dec.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Q = torch.from_numpy(sample_q(N_SAMPLES)).to(DEVICE)
    print(f"[data] sampled {N_SAMPLES} random q within real joint limits")
    print(f"[config] latent={LATENT_DIM}, noise={Q_NOISE} (fixed), epochs={EPOCHS}, 去m")

    n_batches = (N_SAMPLES + BATCH - 1) // BATCH
    hist = {"total": [], "joint": [], "ee": []}

    for ep in range(EPOCHS):
        perm = torch.randperm(N_SAMPLES, device=DEVICE)
        ep_t = ep_j = ep_e = 0.0
        W.train(); dec.train()

        for b in range(n_batches):
            idx = perm[b*BATCH:(b+1)*BATCH]
            q = Q[idx]; B = q.shape[0]

            fk21 = enc.forward(q)
            z = W(fk21[:, :EMB_IN])              # 只喂 p_ee+v_kp (去 m)
            m_batch = m_const.expand(B, -1)

            q_ref = q + torch.randn_like(q) * Q_NOISE   # 固定噪声

            q_pred = dec(z, m_batch, q_prev=q_ref)

            loss_joint = ((q_pred - q) ** 2).mean()
            fk_pred = enc.forward(q_pred)
            loss_ee = ((fk_pred[:, :EMB_IN] - fk21[:, :EMB_IN].detach()) ** 2).mean()
            loss = ALPHA * loss_joint + BETA * loss_ee

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(W.parameters()) + list(dec.parameters()), 1.0)
            opt.step()

            ep_t += loss.item()*B; ep_j += loss_joint.item()*B; ep_e += loss_ee.item()*B

        scheduler.step()
        ep_t/=N_SAMPLES; ep_j/=N_SAMPLES; ep_e/=N_SAMPLES
        hist["total"].append(ep_t); hist["joint"].append(ep_j); hist["ee"].append(ep_e)
        if ep % 10 == 0 or ep == EPOCHS-1:
            lr = scheduler.get_last_lr()[0]
            print(f"[ep {ep:3d}] lr={lr:.6f} total={ep_t:.6f}  joint={ep_j:.6f}  ee={ep_e:.6f}")

    ckpt = os.path.join(CKPT_DIR, "stage1_franka.pt")
    torch.save({"W": W.state_dict(), "decoder": dec.state_dict(),
                "latent_dim": LATENT_DIM,
                "config": {"alpha": ALPHA, "beta": BETA, "noise": Q_NOISE}}, ckpt)
    print(f"[done] saved {ckpt}")

    _plot_loss(hist, os.path.join(OUT_DIR, "stage1_loss.png"))
    print(f"[done] saved stage1_loss.png")

    # ================= 验证 =================
    W.eval(); dec.eval()
    with torch.no_grad():
        Qv = torch.from_numpy(sample_q(2000)).to(DEVICE)
        fk21 = enc.forward(Qv); z = W(fk21[:, :EMB_IN])
        q_pred = dec(z, m_const.expand(Qv.shape[0], -1), q_prev=Qv)   # 验证给真 q_ref
        fk_pred = enc.forward(q_pred)
        ee = torch.norm(fk_pred[:, :3] - fk21[:, :3], dim=1) * 1000
        print(f"\n[eval] EE position reconstruction error (mm): "
              f"mean={ee.mean():.2f}  median={ee.median():.2f}  "
              f"p99={torch.quantile(ee,0.99):.2f}  max={ee.max():.2f}")


def _plot_loss(hist, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(hist["total"], label="total", color="#333", linewidth=2)
    ax.plot(hist["joint"], label="joint-space |q'-q|^2", color="#2F5C9E")
    ax.plot(hist["ee"], label="EE-space |FK(q')-FK(q)|^2", color="#B5651D")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.legend(); ax.set_title("Stage 1 reconstruction loss"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()