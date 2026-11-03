"""
DAgger-lite: periodically (during training) roll out the CURRENT trunk in
the real simulator, so it visits states its own imperfect predictions lead
to -- not just the clean expert trajectory. Each visited state is labeled
with a fresh oracle action, so the trunk gets real "you're here, off the
expert path -- here's the correction" training signal, which teacher-forced
training on the static dataset can never provide (see
diagnose_compounding_error.py's finding: teacher-forced error stays flat
while closed-loop drift explodes mid-episode).

The oracle is IK-free, matching the project's "no online IK" rule: TARGET_Q
is a fixed, known joint configuration, so "what should happen from here" is
just a per-joint capped step directly toward TARGET_Q -- pure joint-space
arithmetic, no inverse kinematics anywhere.

NOTE on why this is NOT a fresh quintic replan (an earlier version was, and
it was a real bug): quintic (minimum-jerk) trajectories have ZERO velocity
at t=0 by construction. Re-planning a fresh quintic from wherever the model
currently is and taking its second waypoint means ALWAYS sampling the
near-zero-velocity onset of a brand new curve, regardless of how far from
the target the model actually is -- every single oracle label came out as
an almost-imperceptible step. Confirmed empirically: after training on
that oracle, mean rollout joint movement dropped from ~3.0 rad to ~1.08
rad (the model learned "the correct move is to barely move"), and success
rate collapsed from 45% to 0%. The fixed version below uses a constant,
capped per-joint step toward the target instead -- no replan, no
zero-velocity-onset artifact, and it naturally decelerates near the target
on its own (once |delta| < the cap, the step just covers the remaining
distance).

DaggerDataset exposes the SAME __getitem__ dict shape as
DecisionCoreDataset, so it can be concatenated with the static dataset via
torch.utils.data.ConcatDataset and used identically by train.py.
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "fr3_pick_gen", "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "fr3_pick_gen", "scripts"))

import mujoco
from generate_fr3_expert_data import (
    sample_random_init, HOME_Q, CONTROL_HZ, SIM_HZ,
    INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES, GRIPPER_OPEN,
)
from decision_core_dataset import T_HISTORY, PADDING_NOISE_STD
from decision_core.model import compute_goal_input

N_ARM_JOINTS = 7
ORACLE_MAX_JOINT_STEP = 0.05  # rad/control-tick, per joint -- matches the
                                # natural single-step scale already
                                # calibrated project-wide (same figure as
                                # HUBER_DELTA). If a joint is closer than
                                # this to target, the step covers the full
                                # remaining distance (natural deceleration
                                # near the target, no separate "arrived"
                                # case needed).


def oracle_step(q_current: np.ndarray, target_q: np.ndarray,
                 max_step: float = ORACLE_MAX_JOINT_STEP) -> np.ndarray:
    """Per-joint capped step directly toward target_q. No replanning, no
    quintic, no zero-velocity-onset artifact -- see the module docstring
    for why the earlier quintic-replan version was a real bug."""
    delta = target_q - q_current
    step = np.clip(delta, -max_step, max_step)
    return q_current + step


@torch.no_grad()
def _encode(fk, embedding, q_arm_np, device):
    q = torch.as_tensor(q_arm_np, dtype=torch.float32, device=device).unsqueeze(0)
    return embedding(fk.forward(q)[:, :18]).squeeze(0)


@torch.no_grad()
def collect_dagger_episode(model_mj, data_mj, jnt_ids, qpos_adr, arm_act_ids, gripper_act_id,
                            fk, embedding, decoder, m_const, trunk, target_q, goal_xyz,
                            max_steps, device):
    """
    Drives the sim with the CURRENT trunk's own predictions (off-policy
    exploration -- visits the model's actual mistakes), but labels every
    visited state with a fresh oracle quintic-to-target. Returns a dict of
    per-step arrays matching DecisionCoreDataset's fields.
    """
    init_q = sample_random_init(model_mj, data_mj, HOME_Q, jnt_ids,
                                 INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES)
    data_mj.qpos[jnt_ids] = init_q
    data_mj.qvel[:] = 0.0
    mujoco.mj_forward(model_mj, data_mj)

    goal_t = torch.as_tensor(goal_xyz, dtype=torch.float32, device=device).unsqueeze(0)
    z0 = _encode(fk, embedding, init_q, device)
    history = [z0 + torch.randn_like(z0) * PADDING_NOISE_STD for _ in range(T_HISTORY - 1)]
    history.append(z0.clone())
    substeps = max(1, round(SIM_HZ / CONTROL_HZ))

    log = {"z_hist": [], "q_ref": [], "goal": [], "action_true": [], "z_true": []}

    for step in range(max_steps):
        q_current = data_mj.qpos[qpos_adr].copy()
        z_hist_snapshot = torch.stack(history, dim=0).clone()   # (T,128), THIS step's history
        q_ref_t = torch.as_tensor(q_current, dtype=torch.float32, device=device).unsqueeze(0)

        # oracle label for the state the model actually visited: a
        # constant, capped step toward target_q -- no replan, no IK
        oracle_next_q = oracle_step(q_current, target_q)
        oracle_next_z = _encode(fk, embedding, oracle_next_q, device)

        log["z_hist"].append(z_hist_snapshot.cpu().numpy())
        log["q_ref"].append(q_current.copy())
        log["goal"].append(np.asarray(goal_xyz, dtype=np.float32))
        log["action_true"].append(oracle_next_q.astype(np.float32))
        log["z_true"].append(oracle_next_z.cpu().numpy())

        # advance the sim using the MODEL's own (possibly wrong) action --
        # this is what makes it DAgger: the model visits its own mistakes,
        # the oracle just supplies the correct label for wherever it ends up
        goal_input = compute_goal_input(fk, goal_t, q_ref_t)
        z_hat = trunk(z_hist_snapshot.unsqueeze(0), goal_input)
        q_pred = decoder(z_hat, m_const, q_prev=q_ref_t)
        q_target_model = q_pred.squeeze(0).cpu().numpy()

        data_mj.ctrl[arm_act_ids] = q_target_model
        if gripper_act_id is not None:
            data_mj.ctrl[gripper_act_id] = GRIPPER_OPEN
        for _ in range(substeps):
            mujoco.mj_step(model_mj, data_mj)

        q_new = data_mj.qpos[qpos_adr].copy()
        history.pop(0)
        history.append(_encode(fk, embedding, q_new, device))

    return log


def collect_dagger_dataset(model_mj, data_mj, jnt_ids, qpos_adr, arm_act_ids, gripper_act_id,
                            fk, embedding, decoder, m_const, trunk, target_q, goal_xyz,
                            n_episodes, max_steps, device):
    all_logs = {"z_hist": [], "q_ref": [], "goal": [], "action_true": [], "z_true": []}
    trunk.eval()
    for _ in range(n_episodes):
        ep_log = collect_dagger_episode(model_mj, data_mj, jnt_ids, qpos_adr, arm_act_ids,
                                         gripper_act_id, fk, embedding, decoder, m_const,
                                         trunk, target_q, goal_xyz, max_steps, device)
        for k in all_logs:
            all_logs[k].extend(ep_log[k])
    return DaggerDataset(all_logs)


class DaggerDataset(Dataset):
    """Same __getitem__ dict shape as DecisionCoreDataset -- concatenate
    freely with torch.utils.data.ConcatDataset."""

    def __init__(self, logs: dict):
        self.z_hist = logs["z_hist"]
        self.q_ref = logs["q_ref"]
        self.goal = logs["goal"]
        self.action_true = logs["action_true"]
        self.z_true = logs["z_true"]

    def __len__(self):
        return len(self.z_hist)

    def __getitem__(self, idx):
        return {
            "z_hist": torch.as_tensor(self.z_hist[idx], dtype=torch.float32),
            "q_ref": torch.as_tensor(self.q_ref[idx], dtype=torch.float32),
            "goal": torch.as_tensor(self.goal[idx], dtype=torch.float32),
            "action_true": torch.as_tensor(self.action_true[idx], dtype=torch.float32),
            "z_true": torch.as_tensor(self.z_true[idx], dtype=torch.float32),
        }