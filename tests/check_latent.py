"""
check_latent_usage.py — 验证 decoder 是真的用 latent, 还是在抄 q_ref 作弊
==========================================================================
noise 很小时 (0.01), decoder 可能忽略 latent、直接抄 q_ref。此脚本用消融验证:
  测试1 [latent 消融]: z 置零, 只留 q_ref -> 误差暴增=真用latent; 不变=抄q_ref
  测试2 [q_ref 消融]: q_ref 置零, 只留 z   -> 看 latent 单独能撑多少
  测试3 [错误 q_ref]: 给别的姿态当 q_ref   -> 被带跑=抄; 靠 latent 修正=真用

用法: python scripts/check_latent_usage.py
"""
import os, sys
import numpy as np
import torch

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import MLPEmbedding
from layer2.decoder import Decoder

ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
CKPT = os.path.join(PROJECT_ROOT, "checkpoints/stage1_franka.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARM_DOF = 7
JOINT_LIMITS = np.array([
    [-2.7437,2.7437],[-1.7837,1.7837],[-2.9007,2.9007],
    [-3.0421,-0.1518],[-2.8065,2.8065],[0.5445,4.5169],[-3.0159,3.0159],
], dtype=np.float32)


def sample_q(n):
    lo=JOINT_LIMITS[:,0]; hi=JOINT_LIMITS[:,1]
    return (lo+np.random.uniform(0,1,(n,ARM_DOF)).astype(np.float32)*(hi-lo))


def ee_err(enc, q_pred, fk_ref):
    return (torch.norm(enc.forward(q_pred)[:,:3]-fk_ref[:,:3],dim=1)*1000)


def main():
    enc = FKEncoder("fr3", ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE); LD = ck["latent_dim"]
    W = MLPEmbedding(18, LD).to(DEVICE); W.load_state_dict(ck["W"]); W.eval()
    dec = Decoder(LD, m_dim=3, out_dim=ARM_DOF, hidden=256, depth=2, use_q_prev=True).to(DEVICE)
    dec.load_state_dict(ck["decoder"]); dec.eval()
    m = enc.m.to(DEVICE).unsqueeze(0)
    N = 2000

    with torch.no_grad():
        Q = torch.from_numpy(sample_q(N)).to(DEVICE)
        fk = enc.forward(Q); z = W(fk[:, :18]); mb = m.expand(N,-1)

        # 基线: 正常 (z + 真 q_ref)
        base = ee_err(enc, dec(z, mb, q_prev=Q), fk)

        # 测试1: latent 置零 (只留 q_ref)
        z0 = torch.zeros_like(z)
        t1 = ee_err(enc, dec(z0, mb, q_prev=Q), fk)

        # 测试1b: latent 换成随机 (打乱)
        zr = z[torch.randperm(N)]
        t1b = ee_err(enc, dec(zr, mb, q_prev=Q), fk)

        # 测试2: q_ref 置零 (只留 latent)
        t2 = ee_err(enc, dec(z, mb, q_prev=torch.zeros_like(Q)), fk)

        # 测试3: 错误 q_ref (给别的姿态)
        Q_wrong = Q[torch.randperm(N)]
        t3 = ee_err(enc, dec(z, mb, q_prev=Q_wrong), fk)

    def stat(x): return f"mean={x.mean():.2f}mm  median={x.median():.2f}mm  max={x.max():.2f}mm"
    print("="*70)
    print(f"[基线]     正常 (z + 真q_ref)      : {stat(base)}")
    print("-"*70)
    print(f"[测试1]    latent置零 (只留q_ref)   : {stat(t1)}")
    print(f"[测试1b]   latent打乱 (乱z+真q_ref) : {stat(t1b)}")
    print(f"[测试2]    q_ref置零 (只留latent)    : {stat(t2)}")
    print(f"[测试3]    错误q_ref (z+别人的q_ref) : {stat(t3)}")
    print("="*70)

    # 判读
    print("\n判读:")
    r1 = t1.mean()/base.mean()
    r1b = t1b.mean()/base.mean()
    r3 = t3.mean()/base.mean()
    print(f"  latent置零后误差 x{r1:.1f} , latent打乱后 x{r1b:.1f}")
    if r1b < 1.5:
        print("  ❌ 警告: 打乱 latent 误差几乎不变 -> decoder 在抄 q_ref, latent 没用! (作弊)")
    elif r1b < 3:
        print("  ⚠️  latent 有用但 q_ref 主导 -> 部分依赖 latent, 可考虑加大 noise")
    else:
        print("  ✅ 打乱 latent 误差大增 -> decoder 真的依赖 latent (不是抄)")
    print(f"  错误q_ref后误差 x{r3:.1f}")
    if r3 > 5:
        print("  ⚠️  错误 q_ref 让误差爆炸 -> decoder 高度依赖 q_ref (小心作弊)")
    else:
        print("  ✅ 错误 q_ref 影响可控 -> latent 有在修正 q_ref")


if __name__ == "__main__":
    main()