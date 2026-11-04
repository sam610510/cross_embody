#!/usr/bin/env python3
"""
Which robots.yaml ee_rotation makes a new arm's TCP ORIENTATION mean the same as the FR3's?

The decision core speaks FR3: "gripper pointing down, fingers closing along world y" has one rot6d in the
FR3 FK encoder. A new arm must produce the SAME rot6d for the same physical gripper pose, otherwise the core
asks for wrong wrist angles (the FK position check cannot see this).

Physical gripper frame, from the geometry of the simulated model (not from any naming convention):
  z = approach direction (FR3: hand z axis; 2-finger grippers on a new arm: flange -> TCP)
  y = finger closing axis   (FR3: hand y axis = finger slide axis; new arm: right pad -> left pad)
  x = y cross z
C_fr3 = R_phys^T R_enc for the FR3 (constant), and the new arm needs R_enc = R_phys C_fr3.
With R_enc = R_tcp_body @ ee_rotation (checked on the arm's CURRENT entry, see below):
  ee_rotation = R_tcp_body^T R_phys C_fr3

  python scripts/calib_ee_rotation.py --arm ur5e_menagerie     # VALIDATION: must reproduce its ee_rotation
  python scripts/calib_ee_rotation.py --arm kuka_iiwa14        # -> the line to put into robots.yaml
"""
import argparse, os, sys
import numpy as np
import torch
import mujoco

R = os.path.expanduser("~/cross_embody")
for p in (R, os.path.join(R, "scripts"), os.path.join(R, "fr3_pick_gen", "scripts")):
    sys.path.insert(0, p)
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
os.environ["NEW_ARM"] = a.arm

import yaml
import train_stage2_align as T
from fk_encoder.fk_encoder import FKEncoder


def r6_to_R(r6):
    """6D rotation -> 3x3 (column order detected as in eval_pp_ur5e.py)"""
    cands = [(r6[:3], r6[3:6]), (r6[0::2], r6[1::2])]
    x, y = min(cands, key=lambda c: abs(np.linalg.norm(c[0]) - 1) + abs(np.linalg.norm(c[1]) - 1) + abs(c[0] @ c[1]))
    x = x / np.linalg.norm(x); y = y - x * (x @ y); y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], 1)


def frame(z, y):
    z = z / np.linalg.norm(z); y = y - z * (y @ z); y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], 1)


def ang(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


rng = np.random.default_rng(0)

# ------------------------------------------------------------ FR3: C_fr3 = R_phys^T R_enc
import build_scene
from generate_fr3_expert_data import resolve_joint_and_actuator_ids
mf = build_scene.build_model(); df = mujoco.MjData(mf)
fj = resolve_joint_and_actuator_ids(mf)[0]
fq = [mf.jnt_qposadr[j] for j in fj]
hand = mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_BODY, "hand_hand")
enc_f = FKEncoder("fr3", T.ROBOTS_YAML, project_root=R, device="cpu")
limf = np.asarray(T.FR3_JOINT_LIMITS, float)
Cs = []
for _ in range(a.n):
    q = rng.uniform(limf[:, 0], limf[:, 1]); df.qpos[fq] = q; mujoco.mj_kinematics(mf, df)
    Rh = df.xmat[hand].reshape(3, 3)
    Rp = frame(Rh[:, 2], Rh[:, 1])
    Re = r6_to_R(enc_f.forward(torch.tensor(q[None], dtype=torch.float32))[0, 3:9].numpy())
    Cs.append(Rp.T @ Re)
C = Cs[0]
print(f"[fr3] C_fr3 constant over {a.n} poses: max deviation {max(ang(C, c) for c in Cs):.3f} deg")

# ------------------------------------------------------------ new arm
import build_scene_arm as B
cfg = B.robot_cfg(); dof = int(cfg["arm_dof"]); gp = B.PREFIX + B.gripper_prefix(cfg)
m = B.build_model(); d = mujoco.MjData(m)
mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, B.MOUNT)


def under(b):
    while b > 0:
        if b == mount:
            return True
        b = m.body_parentid[b]
    return False


hinge = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
arm_j = [j for j in range(m.njnt) if under(m.jnt_bodyid[j]) and int(m.jnt_type[j]) in hinge
         and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(gp)][:dof]
qadr = [m.jnt_qposadr[j] for j in arm_j]
tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, B.TCP_SITE)
flange = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, B.PREFIX + (cfg.get("gripper") or {}).get("attach_site", "attachment_site"))
fb = (cfg.get("gripper") or {}).get("finger_bodies") or ["left_pad", "right_pad"]
pl, pr = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, gp + n) for n in fb)
assert min(tcp, flange, pl, pr) >= 0, "tcp site / flange site / finger bodies not found"
enc = FKEncoder(a.arm, T.ROBOTS_YAML, project_root=R, device="cpu"); T.setup_target(a.arm, enc)
lim = np.asarray(T.TGT_JOINT_LIMITS, float).reshape(dof, 2).clip(-np.pi, np.pi)
E_cur = np.asarray(cfg.get("ee_rotation", np.eye(3)), float)
need, need_sw, post, pre = [], [], [], []
FLIP = np.diag([-1.0, -1.0, 1.0])
for _ in range(a.n):
    q = rng.uniform(lim[:, 0], lim[:, 1]); d.qpos[qadr] = q; mujoco.mj_kinematics(m, d)
    Rb = d.site_xmat[tcp].reshape(3, 3)
    Rp = frame(d.site_xpos[tcp] - d.site_xpos[flange], d.xpos[pl] - d.xpos[pr])
    Re = r6_to_R(enc.forward(torch.tensor(q[None], dtype=torch.float32))[0, 3:9].numpy())
    post.append(Rb.T @ Re); pre.append(Re @ Rb.T)
    need.append(Rb.T @ Rp @ C); need_sw.append(Rb.T @ Rp @ FLIP @ C)
dev = lambda L: max(ang(L[0], x) for x in L)
print(f"[{a.arm}] current ee_rotation in robots.yaml:\n{np.round(E_cur, 4)}")
print(f"  how the encoder applies it: R_body^T R_enc constant within {dev(post):.3f} deg "
      f"(= ee_rotation within {ang(post[0], E_cur):.3f} deg) | R_enc R_body^T constant within {dev(pre):.3f} deg")
for nm, L in (("pads as named (right -> left)", need), ("pads mirrored (left -> right)", need_sw)):
    E = L[0]
    print(f"  required ee_rotation, {nm}: constant within {dev(L):.3f} deg, {ang(E, E_cur):.2f} deg from the current entry")
    print(f"      ee_rotation: {np.round(E, 8).tolist()}")
print("For ur5e_menagerie the required value must equal the current one (validation; a 180 deg difference only means\n"
      "the pad naming is mirrored -- then use the 'fingers swapped' variant for every 2F-85 arm).")