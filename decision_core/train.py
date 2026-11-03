import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "fr3_pick_gen", "scripts"))

import mujoco
from build_scene import build_model, OBJECT_POS
from generate_fr3_expert_data import resolve_joint_and_actuator_ids, TARGET_Q
from decision_core.model import (DecisionCoreModel, load_frozen_stage1,
                                  add_qref_noise, compute_goal_input, decoder_input)
import decision_core_dataset as _dcd
from decision_core_dataset import build_train_val_datasets, T_HISTORY
from decision_core.dagger_buffer import collect_dagger_dataset

_parser = argparse.ArgumentParser()
_parser.add_argument("--n_layers", type=int, default=10)
_parser.add_argument("--n_heads", type=int, default=4)
_parser.add_argument("--seed", type=int, default=42,
                      help="Controls weight init, dropout masks, batch sampling "
                           "order, and DAgger's own random init sampling during "
                           "collection -- everything randomness-related in "
                           "training. Different from rollout_eval.py's SEED, "
                           "which only controls which 20 starts get evaluated.")
_parser.add_argument("--aux_lambda", type=float, default=1.0,
                      help="weight of the latent loss ||z_hat - z_true||^2 (main objective)")
_parser.add_argument("--huber_lambda", type=float, default=0.1,
                      help="weight of the joint-space Huber loss through the frozen decoder (auxiliary)")
_parser.add_argument("--max_epochs", type=int, default=100)
_parser.add_argument("--z_target", choices=["next_state", "action"], default="action",
                      help="latent target for z_hat: Encoder(q_{t+1}) = state actually reached after PD "
                           "tracking (lags the command), or Encoder(action_t) = the commanded next joint "
                           "target (on-manifold AND same step size as the expert)")
_parser.add_argument("--ns_aug_prob", type=float, default=0.5,
                      help="probability that a training sample uses a precomputed null-space (elbow) "
                           "variant (precompute_nullspace.py)")
_parser.add_argument("--ns_aug_rad", type=float, default=0.0,
                      help="ONLINE null-space augmentation, max |α| (rad). Slow (Jacobians every batch); "
                           "default 0 = use the precomputed variants instead")
_parser.add_argument("--fr3_ckpt", type=str, default=None,
                      help="FR3 Stage-1 checkpoint (default stage1_franka.pt; retrained delta-input "
                           "decoder: checkpoints/stage1_franka_dec2.pt)")
_parser.add_argument("--no_dagger", action="store_true")
_parser.add_argument("--tag", type=str, default="_latent",
                      help="appended to the checkpoint name so earlier checkpoints are never overwritten")
_args = _parser.parse_args()

BATCH_SIZE = 32
PEAK_LR = 2e-4
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.01
HUBER_DELTA = 0.05
# Latent loss is the MAIN objective (proposal eq. 3-4): z_hat must land ON the
# latent manifold (= Encoder(q_{t+1})), so any robot's decoder can read it.
# Before: huber + 0.1 * mean-over-128-dims MSE -> the latent term was ~1000x
# too weak and z_hat drifted into a "dialect" only the FR3 decoder understood
# (|z_hat - z(next)| ≈ 0.30; the UR5e decoder then moved in a random direction).
AUX_LAMBDA = _args.aux_lambda
HUBER_LAMBDA = _args.huber_lambda
MAX_EPOCHS = _args.max_epochs
PATIENCE = 30
DROPOUT = 0.1

DAGGER_ENABLED = not _args.no_dagger
DAGGER_START_EPOCH = 15
DAGGER_INTERVAL_EPOCHS = 10
DAGGER_N_EPISODES = 20
DAGGER_MAX_STEPS = 150
DAGGER_SAMPLE_WEIGHT = 2.0
USE_WEIGHTED_SAMPLING = False
N_LAYERS_OVERRIDE = _args.n_layers
N_HEADS_OVERRIDE = _args.n_heads  # checkpoint filename encodes both, so
                                    # sweeping either one never overwrites
                                    # another run's result
                                    # (decision_core_l{layers}_h{heads}_s{seed}{tag}.pt)
HOVER_HEIGHT = 0.10
GOAL_XYZ = tuple(np.array(OBJECT_POS) + np.array([0.0, 0.0, HOVER_HEIGHT]))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING = None   # frozen Stage-1 W_FR3, set in main() (needed for z_target == "action")
TRAIN_SEED = _args.seed
CKPT_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints",
    f"decision_core_l{N_LAYERS_OVERRIDE}_h{N_HEADS_OVERRIDE}_s{TRAIN_SEED}{_args.tag}.pt")


def lr_lambda(step: int, warmup_steps: int, total_steps: int):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================ null-space augmentation ============================
# FR3 has 7 joints for a 6-DoF TCP pose: the same TCP can be held with infinitely
# many elbow configurations. The decision core should decide from the TCP, not from
# which elbow it happens to see -- otherwise a new arm's alignment has to guess the
# expert's elbow "dialect". Here every chosen sample is moved along the null-space
# (TCP fixed, elbow rotates) by the SAME α for its whole history, current pose,
# action and next state, then re-encoded with the frozen W_FR3. Targets move with
# the history, so the core learns to keep whatever elbow style its input has.
FR3_LIMITS = torch.tensor([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
                           [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169],
                           [-3.0159, 3.0159]])


def _tcp_feat(fk, q):
    f = fk.forward(q)
    return torch.cat([f[:, :3], 0.3 * f[:, 3:9]], 1)        # position + weighted rot6d


def _jac(fk, q, eps=1e-4):
    n, d = q.shape
    qq = q.unsqueeze(1).repeat(1, d + 1, 1)
    qq[:, 1:] += eps * torch.eye(d, device=q.device)
    f = _tcp_feat(fk, qq.reshape(-1, d)).view(n, d + 1, -1)
    return ((f[:, 1:] - f[:, :1]) / eps).transpose(1, 2), f[:, 0]   # (n,9,d), (n,9)


@torch.no_grad()
def nullspace_augment(fk, emb, batch, prob, max_rad):
    q_hist, q_ref = batch["q_hist"], batch["q_ref"]
    B, T, d = q_hist.shape
    mask = (torch.rand(B, device=DEVICE) < prob) & batch["has_q"]
    if max_rad <= 0 or not mask.any():
        return batch, 0.0
    Q = torch.cat([q_hist.reshape(B * T, d), q_ref, batch["action_true"], batch["q_next"]], 0)
    owner = torch.cat([torch.arange(B, device=DEVICE).repeat_interleave(T)] +
                      [torch.arange(B, device=DEVICE)] * 3)
    J, f0 = _jac(fk, Q)
    v = torch.linalg.svd(J).Vh[:, -1, :]                       # null-space direction (n,7)
    v_ref = v[B * T:B * T + B][owner]                           # align signs with q_ref's
    v = v * torch.where((v * v_ref).sum(1, keepdim=True) < 0, -1.0, 1.0)
    alpha = (torch.rand(B, device=DEVICE) * 2 - 1) * max_rad * mask
    Q2 = Q + alpha[owner, None] * v
    lim = FR3_LIMITS.to(DEVICE)
    for _ in range(3):                                          # restore the TCP exactly
        J2, f2 = _jac(fk, Q2)
        dq = (torch.linalg.pinv(J2, rtol=1e-4) @ (f0 - f2).unsqueeze(-1)).squeeze(-1)   # J has rank 6
        Q2 = torch.max(torch.min(Q2 + dq, lim[:, 1]), lim[:, 0])
    res = (fk.forward(Q2)[:, :3] - fk.forward(Q)[:, :3]).norm(dim=1)
    bad = torch.zeros(B, device=DEVICE).index_add_(0, owner, (res > 0.002).float()) > 0  # any row off by >2 mm
    mask = mask & ~bad
    if not mask.any():
        return batch, 0.0
    qh2 = Q2[:B * T].view(B, T, d)
    qr2, act2, qn2 = Q2[B * T:B * T + B], Q2[B * T + B:B * T + 2 * B], Q2[B * T + 2 * B:]
    z_h2 = emb(fk.forward(qh2.reshape(-1, d))[:, :18]).view(B, T, -1) + batch["pad_noise"]
    z_t2 = emb(fk.forward(qn2)[:, :18])
    m1, m3 = mask[:, None], mask[:, None, None]
    out = dict(batch)
    out["z_hist"] = torch.where(m3, z_h2, batch["z_hist"])
    out["q_ref"] = torch.where(m1, qr2, q_ref)
    out["action_true"] = torch.where(m1, act2, batch["action_true"])
    out["z_true"] = torch.where(m1, z_t2, batch["z_true"])
    return out, mask.float().mean().item()


class _WithQ(torch.utils.data.Dataset):
    """DAgger samples carry no FR3 joint history: add the extra keys (has_q=False)
    so they batch together with the main dataset and are simply never augmented."""
    def __init__(self, ds, t_history):
        self.ds, self.T = ds, t_history

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = dict(self.ds[i])
        if "q_hist" not in item:
            item["q_hist"] = torch.zeros(self.T, 7)
            item["q_next"] = torch.zeros(7)
            item["pad_noise"] = torch.zeros(self.T, item["z_hist"].shape[-1])
            item["has_q"] = torch.tensor(False)
        return item


def run_epoch(model, loader, optimizer, scheduler, train: bool):
    model.trunk.train(mode=train)
    total_loss = total_huber = total_aux = total_zgap = 0.0
    n_samples = 0

    with torch.set_grad_enabled(train):
        aug_frac = 0.0; n_batches = 0
        for batch in loader:
            if train and _args.ns_aug_rad > 0:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                batch, fr = nullspace_augment(model.fk, EMBEDDING, batch,
                                              _args.ns_aug_prob, _args.ns_aug_rad)
                aug_frac += fr; n_batches += 1
            z_hist = batch["z_hist"].to(DEVICE)
            goal = batch["goal"].to(DEVICE)
            q_ref = batch["q_ref"].to(DEVICE)
            action_true = batch["action_true"].to(DEVICE)
            z_true = batch["z_true"].to(DEVICE)
            if _args.z_target == "action":
                with torch.no_grad():
                    z_true = EMBEDDING(model.fk.forward(action_true)[:, :18])
            B = z_hist.shape[0]

            q_ref_used = add_qref_noise(q_ref, model.qref_noise_std) if train else q_ref
            goal_input = compute_goal_input(model.fk, goal, q_ref_used)
            z_hat = model.trunk(z_hist, goal_input)
            m_batch = model.m_const.expand(B, -1)
            z_in = decoder_input(z_hat, q_ref_used, model.fk, EMBEDDING, model.decoder)
            q_pred = model.decoder(z_in, m_batch, q_prev=q_ref_used)

            huber = F.huber_loss(q_pred, action_true, delta=HUBER_DELTA)
            aux = ((z_hat - z_true) ** 2).sum(dim=1).mean()   # squared L2 per sample (not per-dim mean)
            loss = HUBER_LAMBDA * huber + AUX_LAMBDA * aux
            zgap = (z_hat - z_true).norm(dim=1).mean()       # comparable to the 0.13 alignment gap

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.trunk.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * B
            total_huber += huber.item() * B
            total_aux += aux.item() * B
            total_zgap += zgap.item() * B
            n_samples += B

    return {"loss": total_loss / n_samples, "huber": total_huber / n_samples,
            "aux": total_aux / n_samples, "zgap": total_zgap / n_samples,
            "aug": aug_frac / max(1, n_batches)}


def main():
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    print(f"[init] device = {DEVICE}  seed = {TRAIN_SEED}")
    print(f"[init] null-space variants: prob={_args.ns_aug_prob} (precomputed)  "
          f"online augmentation max |α|={_args.ns_aug_rad} rad (0 = off)")
    print(f"[init] z_target={_args.z_target}  aux_lambda={AUX_LAMBDA}  huber_lambda={HUBER_LAMBDA}  "
          f"max_epochs={MAX_EPOCHS}  ckpt={CKPT_PATH}")

    _dcd.NS_PROB = _args.ns_aug_prob
    train_ds, val_ds = build_train_val_datasets(t_history=T_HISTORY)
    if USE_WEIGHTED_SAMPLING:
        train_sampler = WeightedRandomSampler(
            weights=train_ds.weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                                   num_workers=4, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4)

    fk, embedding, decoder, m_const = load_frozen_stage1(device=DEVICE, ckpt_path=_args.fr3_ckpt)
    if decoder.delta_scale is not None and DAGGER_ENABLED:
        raise SystemExit(
            "[stop] the delta-input FR3 decoder needs dagger_buffer.py to build its decoder input "
            "with decoder_input(); update it first, or run with --no_dagger")
    global EMBEDDING
    EMBEDDING = embedding
    model = DecisionCoreModel(fk, decoder, m_const, t_history=T_HISTORY,
                               n_layers=N_LAYERS_OVERRIDE, n_heads=N_HEADS_OVERRIDE,
                               dropout=DROPOUT, embedding=embedding).to(DEVICE)
    print(f"[init] n_layers={N_LAYERS_OVERRIDE} n_heads={N_HEADS_OVERRIDE}  "
          f"trunk params: {sum(p.numel() for p in model.trunk.parameters()):,}")

    if DAGGER_ENABLED:
        dagger_model_mj = build_model()
        dagger_data_mj = mujoco.MjData(dagger_model_mj)
        dagger_jnt_ids, dagger_qpos_adr, dagger_arm_act_ids, dagger_gripper_act_id, _ = \
            resolve_joint_and_actuator_ids(dagger_model_mj)
        print(f"[dagger] starts at epoch {DAGGER_START_EPOCH}, "
              f"every {DAGGER_INTERVAL_EPOCHS} epochs, "
              f"{DAGGER_N_EPISODES}x{DAGGER_MAX_STEPS} steps")

    current_train_loader = train_loader

    optimizer = torch.optim.AdamW(model.trunk.parameters(), lr=PEAK_LR,
                                   weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, WARMUP_STEPS, total_steps))

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        if (DAGGER_ENABLED and epoch >= DAGGER_START_EPOCH and
                (epoch - DAGGER_START_EPOCH) % DAGGER_INTERVAL_EPOCHS == 0):
            print(f"[dagger] collecting (epoch {epoch})...")
            t_dagger = time.time()
            dagger_ds = collect_dagger_dataset(
                dagger_model_mj, dagger_data_mj, dagger_jnt_ids, dagger_qpos_adr,
                dagger_arm_act_ids, dagger_gripper_act_id, fk, embedding, decoder,
                m_const, model.trunk, TARGET_Q, GOAL_XYZ,
                DAGGER_N_EPISODES, DAGGER_MAX_STEPS, DEVICE)
            print(f"[dagger] {len(dagger_ds)} samples ({time.time()-t_dagger:.1f}s)")

            combined_ds = ConcatDataset([train_ds, _WithQ(dagger_ds, T_HISTORY)])
            if USE_WEIGHTED_SAMPLING:
                combined_weights = list(train_ds.weights) + [DAGGER_SAMPLE_WEIGHT] * len(dagger_ds)
            else:
                combined_weights = [1.0] * len(train_ds) + [DAGGER_SAMPLE_WEIGHT] * len(dagger_ds)
            combined_sampler = WeightedRandomSampler(
                weights=combined_weights, num_samples=len(combined_ds), replacement=True)
            current_train_loader = DataLoader(
                combined_ds, batch_size=BATCH_SIZE, sampler=combined_sampler,
                num_workers=0, drop_last=True)

        train_stats = run_epoch(model, current_train_loader, optimizer, scheduler, True)
        val_stats = run_epoch(model, val_loader, optimizer, scheduler, False)
        dt = time.time() - t0

        print(f"[ep {epoch:3d}] ({dt:5.1f}s) "
              f"train_loss={train_stats['loss']:.5f}  val_loss={val_stats['loss']:.5f}  "
              f"val_aux={val_stats['aux']:.4f}  val_huber={val_stats['huber']:.6f}  "
              f"val_zgap={val_stats['zgap']:.3f}  (target < 0.13)  "
              f"null-space-augmented={train_stats['aug']:.0%}")

        task_loss = val_stats["loss"]

        if task_loss < best_val_loss:
            best_val_loss = task_loss
            epochs_no_improve = 0
            torch.save({
                "trunk_state_dict": model.trunk.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_zgap": val_stats["zgap"],
                "config": {
                    "d_model": model.trunk.d_model, "n_layers": len(model.trunk.blocks),
                    "n_heads": N_HEADS_OVERRIDE,
                    "t_history": T_HISTORY, "huber_delta": HUBER_DELTA,
                    "aux_lambda": AUX_LAMBDA, "huber_lambda": HUBER_LAMBDA,
                    "aux_form": "sum_sq_l2", "z_target": _args.z_target,
                    "ns_aug_prob": _args.ns_aug_prob, "ns_aug_rad": _args.ns_aug_rad,
                    "fr3_ckpt": _args.fr3_ckpt,
                },
            }, CKPT_PATH)
            print(f"         -> new best (task_loss={task_loss:.5f}, zgap={val_stats['zgap']:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[stop] no improvement for {PATIENCE} epochs "
                      f"(best task_loss = {best_val_loss:.5f})")
                break
    else:
        print(f"[stop] reached max_epochs={MAX_EPOCHS} -- the cosine schedule ended, "
              f"check whether val_zgap was still falling")

    print("[done]")


if __name__ == "__main__":
    main()