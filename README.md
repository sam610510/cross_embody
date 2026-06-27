# cross_embody

Decoupling Decision and Control for Cross-Embodiment Learning.
单一策略于 Franka 模拟训练,zero-shot 部署至 UR5e 实机,无需任务示范。

## 目录结构

```
cross_embody/
├── configs/
│   ├── dims.yaml         # 维度契约 (Interface Contract):所有模组共用,不得各自定义
│   └── robots.yaml       # 机器人注册表:加新手臂只需在此加 entry + 放 URDF
├── fk_encoder/
│   └── fk_encoder.py     # Layer 1: FK Encoder (非学习,纯几何),输出 19 维物理特征
├── tests/
│   └── test_fk_fr3.py    # FR3 验证
├── urdfs/
│   └── franka_description/fr3.urdf
├── policy_io/            # (待:Stage 2 transformer 介面)
└── scripts/              # (待:训练脚本)
```

## Layer 1: FK Encoder

非学习、设定驱动的前向运动学编码器。把任意 DoF 的手臂关节角映射成统一的
19 维物理特征,让不同形态机器人在同一空间可比较。

输出 19 维: `[ p_ee(7) | v_kp(9) | m(3) ]`
- `p_ee` : EE position(3) + quaternion(4),以 base 为原点
- `v_kp` : 三个 keyvector (v_SE, v_SW, v_SEE),以 upper-arm 长度 L 正规化
- `m`    : `[arm_ratio, workspace_r, dof_norm]`,结构常数,init 算一次

### 加新手臂 (零改程式)
1. 放 URDF 到 `urdfs/<robot>/`
2. 在 `configs/robots.yaml` 加一个 entry,指定 `urdf` / `root_link` / `arm_dof` /
   `key_links` (shoulder/elbow/wrist/ee) / `base_rotation`
3. 若该机器人 base 轴向与统一约定不同,填 `base_rotation` (固定旋转矩阵)

## 执行测试

```bash
conda activate cross_embody
cd ~/cross_embody
python tests/test_fk_fr3.py    # 从任何目录执行皆可
```

## 环境
- conda env: `cross_embody` (Python 3.10)
- 主要相依: `torch`, `pytorch_kinematics`, `numpy`, `pyyaml`
