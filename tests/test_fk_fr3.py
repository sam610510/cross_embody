"""
test_fk_fr3.py — 用真实 FR3 验证 FKEncoder。
检查: 输出维度=19、m 是结构常数 (不随姿态变)、p_ee 数值合理、quaternion 单位长。

可从任何目录执行: `python tests/test_fk_fr3.py` 或 `cd tests && python test_fk_fr3.py` 皆可。
"""
import os
import sys

# 让本档无论从哪执行都能 import fk_encoder、找到 configs/ 与 urdfs/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # project root (~/cross_embody)
sys.path.insert(0, os.path.join(ROOT, "fk_encoder"))

import torch
from fk_encoder import FKEncoder

ROBOTS_YAML = os.path.join(ROOT, "configs", "robots.yaml")


def main():
    enc = FKEncoder("fr3", ROBOTS_YAML, project_root=ROOT)

    print(f"robot=fr3  arm_dof={enc.arm_dof}  n_actuated={enc.n_actuated}")
    print(f"L (upper-arm) = {enc.L.item():.4f} m")
    print(f"m = [arm_ratio={enc.m[0]:.4f}, workspace_r={enc.m[1]:.4f}, dof_norm={enc.m[2]:.4f}]")
    print()

    qs = [
        torch.zeros(7),
        torch.tensor([0.3, -0.5, 0.2, -1.8, 0.1, 1.6, 0.7]),
        torch.tensor([-0.8, 0.4, 1.0, -1.0, -0.5, 1.2, 0.0]),
    ]

    m_ref = None
    for i, q in enumerate(qs):
        out = enc(q)
        assert out.shape == (1, 19), out.shape
        v = out[0]
        p_ee, v_kp, m = v[:7], v[7:16], v[16:19]
        quat = p_ee[3:7]

        assert abs(quat.norm().item() - 1.0) < 1e-4, "quaternion not unit"
        if m_ref is None:
            m_ref = m.clone()
        else:
            assert torch.allclose(m, m_ref, atol=1e-5), "m drifted across poses!"

        pos = p_ee[:3].tolist()
        print(f"trial {i}: p_ee_pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})  "
              f"|quat|={quat.norm():.4f}  v_kp[0:3]={v_kp[:3].tolist()}")

    print("\n[OK] 维度=19、m 结构常数不变、quaternion 单位长。")

    qb = torch.stack(qs)
    outb = enc(qb)
    assert outb.shape == (3, 19), outb.shape
    print(f"[OK] batch forward: {tuple(qb.shape)} -> {tuple(outb.shape)}")


if __name__ == "__main__":
    main()
