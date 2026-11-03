"""
Dataset for decision-core training. Reads the latent-augmented parquet
files written by precompute_latents.py and builds T-step sliding-window
samples. Split is done at the EPISODE level (whole episodes held out for
val), never by shuffling individual timesteps -- see the reasoning in
learnings.md.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

N_ARM_JOINTS = 7
LATENT_DIM = 128
T_HISTORY = 8
VAL_FRACTION = 0.2
SPLIT_SEED = 0
PADDING_NOISE_STD = 0.15  # padded rows get independent Gaussian noise added,
                           # not exact duplicates -- verified empirically that
                           # exact duplication makes the goal-query pooling
                           # mechanism mathematically insensitive to goal
                           # regardless of training (attention over near-
                           # identical values pools to ~the same output no
                           # matter what the query is). rollout_eval*.py's
                           # bootstrap must use the SAME noise scheme, or
                           # every rollout's opening step re-hits this same
                           # dead zone regardless of how well training went.

LATENT_DIR = Path(os.path.expanduser(
    "~/cross_embody/fr3_pick_gen/datasets/expert_v1/data_latent/chunk-000"))
# Same episodes + precomputed null-space (elbow) variants, written by
# precompute_nullspace.py. Used automatically when it exists.
LATENT_NS_DIR = Path(os.path.expanduser(
    "~/cross_embody/fr3_pick_gen/datasets/expert_v1/data_latent_ns/chunk-000"))
NS_PROB = 0.0   # probability that a TRAIN sample uses a null-space variant (set by train.py)
CHUNK = 0       # >0: action chunking / TCP baseline: also return the next CHUNK states / gripper commands (set by train_pp_chunk.py)
HORIZON = 1     # q_future = state HORIZON steps ahead (clipped at the episode end; set by train.py)
# EDIT ME if you go back to the single-goal dataset for any reason --
# this MUST point at whichever dataset you actually want to train on.
# (This was silently still pointing at expert_v1's single fixed goal for
# a long stretch of debugging -- every sample had the identical goal_xyz,
# which made the contrastive loss's "wrong goal" trivially equal to the
# "correct" one, guaranteeing dist=0 no matter what the architecture or
# training recipe did. Always sanity-check with debug_contrastive.py's
# "goal vs goal_wrong" printout before trusting any contrastive-loss result.)

# Fallback only for OLD datasets generated before goal_xyz was logged per-row
# (the original single-goal expert_v1 set). Any new dataset (including the
# multi-goal one) has a real goal_xyz column and never touches this.
try:
    from build_scene import OBJECT_POS as _FALLBACK_GOAL_XYZ
except ImportError:
    _FALLBACK_GOAL_XYZ = (0.4, 0.0, 0.0)


def _episode_split(files, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(round(len(files) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    return train_files, val_files


class DecisionCoreDataset(Dataset):
    """
    Each item:
      z_hist      (T, 128)  float32 -- latent history, oldest to newest,
                                        ending at the "current" timestep t
      q_ref       (7,)      float32 -- current arm qpos (== the qpos that
                                        produced z_hist[-1])
      goal        (3,)      float32 -- fixed reach target xyz
      action_true (7,)      float32 -- ground-truth arm Delta q at step t
      z_true      (128,)    float32 -- Encoder(q_{t+1}), i.e. next row's
                                        cached latent (see precompute_latents.py)
      q_hist      (T, 7)    float32 -- FR3 joint angles behind z_hist (padding repeats frame 0)
      q_next      (7,)      float32 -- q_{t+1}
      pad_noise   (T, 128)  float32 -- the noise added to padded latent rows (0 for real rows)
      has_q       ()        bool    -- True (DAgger samples get False via train.py's wrapper)
    """

    def __init__(self, files, t_history: int = T_HISTORY, goal_only_boost: float = 5.0,
                 augment: bool = False):
        self.t_history = t_history
        self.augment = augment
        self.ns_valid = []      # per episode: (K, n_frames) bool, or None
        self.samples = []       # list of (df_index, start_frame)
        self.weights = []       # sampling weight per sample -- boosted for
                                  # samples where history is still padded
                                  # (t < t_history-1), since goal is the ONLY
                                  # signal available there. See the ablation
                                  # finding in learnings.md: history alone
                                  # explains almost every "normal" sample,
                                  # drowning out the gradient signal that
                                  # would otherwise teach the trunk to use goal.
        self.episode_dfs = []   # one dataframe per episode, loaded once

        for df_idx, f in enumerate(files):
            df = pd.read_parquet(f)
            self.episode_dfs.append(df)
            ks = sorted(int(c[len("latent_ns"):]) for c in df.columns if c.startswith("latent_ns"))
            self.ns_valid.append(np.stack([df[f"valid_ns{k}"].values for k in ks]) if ks else None)
            n_frames = len(df)
            # need one more frame (t+1) to read off z_true -- so the last
            # valid t is n_frames - 2. Start from t=0 (NOT t_history-1): the
            # first t_history-1 positions don't have a full real history yet,
            # but rollout's very first control step is EXACTLY this situation
            # (history padded/repeated because the episode just started), so
            # skipping them left training with zero examples of what every
            # single rollout episode's opening step actually looks like.
            last_valid_t = n_frames - 2
            for t in range(0, last_valid_t + 1):
                self.samples.append((df_idx, t))
                self.weights.append(goal_only_boost if t < t_history - 1 else 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        df_idx, t = self.samples[idx]
        df = self.episode_dfs[df_idx]
        start = t - self.t_history + 1

        # null-space variant: same TCP trajectory, elbow rotated by a fixed α
        lat_col, st_col, ac_col = "latent", "observation.state", "action"
        valid = self.ns_valid[df_idx]
        if self.augment and valid is not None and np.random.rand() < NS_PROB:
            ok = valid[:, max(0, start):t + 2].all(1)
            if ok.any():
                k = np.random.choice(np.flatnonzero(ok))
                lat_col, st_col, ac_col = f"latent_ns{k}", f"state_ns{k}", f"action_ns{k}"

        st_all = np.stack(df[st_col].values)
        q_all = st_all[:, :N_ARM_JOINTS]
        fing = st_all[:, N_ARM_JOINTS] if st_all.shape[1] > N_ARM_JOINTS else np.zeros(len(st_all))
        lo = max(0, start)
        f_hist = np.concatenate([np.repeat(fing[0:1], max(0, -start)), fing[lo:t + 1]]).astype(np.float32)
        pad_noise = np.zeros((self.t_history, LATENT_DIM), dtype=np.float32)
        if start < 0:
            # not enough real history yet -- pad by repeating the FIRST
            # available frame (frame 0), same convention rollout_eval.py
            # uses to bootstrap history at the start of an episode, PLUS
            # independent noise per padded row (see PADDING_NOISE_STD --
            # exact duplicates make attention pooling structurally
            # goal-insensitive for these samples, defeating the whole
            # point of oversampling them).
            real = np.stack(df[lat_col].values[0:t + 1])        # (t+1, 128)
            n_pad = self.t_history - real.shape[0]
            pad = np.repeat(real[0:1], n_pad, axis=0).copy()     # (n_pad, 128)
            noise = np.random.randn(*pad.shape).astype(np.float32) * PADDING_NOISE_STD
            pad += noise
            pad_noise[:n_pad] = noise
            z_hist = np.concatenate([pad, real], axis=0).copy()  # (T,128)
            q_hist = np.concatenate([np.repeat(q_all[0:1], n_pad, axis=0), q_all[0:t + 1]], axis=0)
        else:
            z_hist = np.stack(df[lat_col].values[start:t + 1]).copy()  # (T,128)
            q_hist = q_all[start:t + 1]

        q_ref = np.asarray(df[st_col].iloc[t])[:N_ARM_JOINTS].copy()
        action_true = np.asarray(df[ac_col].iloc[t])[:N_ARM_JOINTS].copy()
        z_true = np.asarray(df[lat_col].iloc[t + 1]).copy()            # (128,)

        if "goal_xyz" in df.columns:
            goal = np.asarray(df["goal_xyz"].iloc[t]).copy()
        else:
            goal = np.asarray(_FALLBACK_GOAL_XYZ, dtype=np.float32)

        item = {
            "z_hist": torch.as_tensor(z_hist, dtype=torch.float32),
            "q_ref": torch.as_tensor(q_ref, dtype=torch.float32),
            "goal": torch.as_tensor(goal, dtype=torch.float32),
            "action_true": torch.as_tensor(action_true, dtype=torch.float32),
            "z_true": torch.as_tensor(z_true, dtype=torch.float32),
            # FR3 joint angles behind the latents -- used by train.py's null-space
            # augmentation (same TCP, different elbow -> re-encoded latents)
            "q_hist": torch.as_tensor(q_hist.copy(), dtype=torch.float32),   # (T,7)
            "q_next": torch.as_tensor(q_all[t + 1].copy(), dtype=torch.float32),
            # sub-goal ('intent') target: the state HORIZON steps ahead, same null-space variant
            "q_future": torch.as_tensor(q_all[min(t + HORIZON, len(q_all) - 1)].copy(), dtype=torch.float32),
            "pad_noise": torch.as_tensor(pad_noise, dtype=torch.float32),     # (T,128)
            "has_q": torch.tensor(True),
            # pick-and-place fields (zeros for the reaching dataset)
            "finger_hist": torch.as_tensor(f_hist, dtype=torch.float32),               # (T,)
            "place": torch.as_tensor(np.array(df["place_xyz"].iloc[t], dtype=np.float32)   # copy:
                                     if "place_xyz" in df.columns else np.zeros(3, np.float32)),  # writable
            "grip_cmd": torch.tensor(float(df["gripper_closed"].iloc[t]) if "gripper_closed" in df.columns
                                     else 0.0),
            # posture goal of the episode (swivel angles for the pick / place waypoints), if recorded
            "psi_goal": torch.as_tensor(np.array(df["psi_goal"].iloc[t], dtype=np.float32)
                                        if "psi_goal" in df.columns else np.zeros(2, np.float32)),
            "psi_now": torch.tensor(float(df["psi_now"].iloc[t]) if "psi_now" in df.columns else 0.0),
        }
        if CHUNK > 0:                       # chunk targets, same null-space variant, clipped at the episode end
            n = len(q_all)
            ks = np.minimum(np.arange(t + 1, t + CHUNK + 1), n - 1)
            item["q_chunk"] = torch.as_tensor(q_all[ks].copy(), dtype=torch.float32)          # (K,7)
            gc = df["gripper_closed"].values if "gripper_closed" in df.columns else np.zeros(n)
            item["grip_chunk"] = torch.as_tensor(gc[np.minimum(ks - 1, n - 1)].astype(np.float32))  # (K,)
            # which chunk steps are real (not repeats of the last frame)
            item["chunk_valid"] = torch.as_tensor(np.arange(t + 1, t + CHUNK + 1) <= n - 1)
        return item


def build_train_val_datasets(t_history: int = T_HISTORY, latent_dir: Path = None, dataset: str = None):
    if dataset is not None:                 # e.g. "expert_pp_v1": prefer its null-space variants
        root = Path(os.path.expanduser(f"~/cross_embody/fr3_pick_gen/datasets/{dataset}"))
        ns = root / "data_latent_ns" / "chunk-000"
        latent_dir = ns if ns.exists() else root / "data_latent" / "chunk-000"
        files = sorted(glob.glob(str(latent_dir / "episode_*.parquet")))
        if not files:
            raise FileNotFoundError(f"no episodes in {latent_dir} -- run precompute_latents_pp.py first")
        print(f"[dataset] {latent_dir}  (null-space variants: {'yes' if latent_dir == ns else 'no'}, "
              f"train prob={NS_PROB})")
        tr, va = _episode_split(files)
        return (DecisionCoreDataset(tr, t_history, augment=True),
                DecisionCoreDataset(va, t_history, augment=False))
    if latent_dir is None:
        latent_dir = LATENT_NS_DIR if LATENT_NS_DIR.exists() else LATENT_DIR
    print(f"[dataset] {latent_dir}  (null-space variants: "
          f"{'yes' if latent_dir == LATENT_NS_DIR else 'no'}, train prob={NS_PROB})")
    files = sorted(glob.glob(str(latent_dir / "episode_*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No latent-augmented episodes found in {latent_dir}. "
            f"Run precompute_latents.py first.")
    train_files, val_files = _episode_split(files)
    print(f"{len(files)} episodes total -> {len(train_files)} train / "
          f"{len(val_files)} val")
    return (DecisionCoreDataset(train_files, t_history, augment=True),
            DecisionCoreDataset(val_files, t_history, augment=False))


if __name__ == "__main__":
    train_ds, val_ds = build_train_val_datasets()
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    sample = train_ds[0]
    for k, v in sample.items():
        print(f"  {k:12s} {tuple(v.shape)}")