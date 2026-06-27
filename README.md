# cross_embody

Decoupling Decision and Control for Cross-Embodiment Learning.
A single policy trained in Franka simulation, deployed zero-shot to a UR5e real robot without task demonstrations.

## Directory Structure

```
cross_embody/
├── configs/
│   ├── dims.yaml         # Interface contract: shared dim spec, no module redefines it
│   └── robots.yaml       # Robot registry: add a new arm by adding an entry + a URDF
├── fk_encoder/
│   └── fk_encoder.py     # Layer 1: FK Encoder (non-learnable, pure geometry), 19-dim output
├── tests/
│   └── test_fk_fr3.py    # FR3 validation
├── urdfs/
│   └── franka_description/fr3.urdf
├── policy_io/            # (TODO: Stage 2 transformer interface)
└── scripts/              # (TODO: training scripts)
```

## Layer 1: FK Encoder

A non-learnable, config-driven forward-kinematics encoder. It maps joint angles of
an arbitrary-DoF arm into a unified 19-dim physical feature vector, so robots of
different morphologies become comparable in the same space.

Output (19-dim): `[ p_ee(7) | v_kp(9) | m(3) ]`
- `p_ee` : EE position(3) + quaternion(4), expressed in the base frame
- `v_kp` : three keyvectors (v_SE, v_SW, v_SEE), normalized by upper-arm length L
- `m`    : `[arm_ratio, workspace_r, dof_norm]`, structural constants computed once at init

### Adding a new arm (no code changes)
1. Place the URDF under `urdfs/<robot>/`
2. Add an entry in `configs/robots.yaml` specifying `urdf` / `root_link` / `arm_dof` /
   `key_links` (shoulder/elbow/wrist/ee) / `base_rotation`
3. If the robot's base axes differ from the unified convention, set `base_rotation`
   (a fixed rotation matrix)

## Running the test

```bash
conda activate cross_embody
cd ~/cross_embody
python tests/test_fk_fr3.py    # runnable from any directory
```

## Environment
- conda env: `cross_embody` (Python 3.10)
- Key dependencies: `torch`, `pytorch_kinematics`, `numpy`, `pyyaml`
