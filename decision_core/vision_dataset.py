"""
Pick-and-place samples for the VISION decision core (vision_core.DecisionCoreVision), frozen encoder.

Wraps decision_core_dataset.DecisionCoreDataset unchanged (sliding windows, padding, null-space
variants, chunk targets, episode-level split) and adds the object tokens precomputed by
vision/encode_episode_tokens.py (images/ep_XXXXXX/obj_tokens.npz: keypoint uv, visibility, latent z):

  cube_hist  (T, tok)  cube tokens of the same T frames as z_hist (padding repeats frame 0, like z_hist)
  tape_goal  (tok,)    tape token of the first frame where the tape is seen, fixed for the episode
  cam_pos / cam_xmat / cam_K  (T, 3) / (T, 9) / (T, 3, 3)   camera of the same T frames (meta.npz of the
                       render), for the TCP token (vision_core.tcp_tokens, computed on the GPU in training)

Occlusion handling (training AND evaluation, see hold_last_visible):
  when the cube's visibility is below hold_thr (gripper over it, carried in the hand), its keypoint is
  replaced by the last keypoint where it was seen; the visibility stays as measured, so the core knows
  the position is old. Without this the keypoint of a covered cube jumps to wherever the heatmap peaks.

Training-only augmentation:
  pair_swap : a frame with a paired render (same frame, other arm colour) uses that render's token
  uv_noise  : Gaussian noise on the keypoints, in pixels of a --img_w x --img_h image
  z_noise   : Gaussian noise on z, as a fraction of its per-dimension std
  tok_drop  : per history frame, probability to treat the cube as unseen (old keypoint, visibility 0)
"""
import os
from pathlib import Path

import numpy as np
import torch

import decision_core_dataset as dcd
from decision_core.vision_core import object_tokens, xyz_tokens


def hold_last_valid_xyz(xyz, ok):
    """xyz (F, 3), ok (F,) bool -> held positions (last valid; before the first valid one: the first valid one)
    and a 'fresh' flag (F,) (False where a held value is used)"""
    out = xyz.copy()
    idx = np.flatnonzero(ok)
    if not len(idx):
        return np.zeros_like(xyz), np.zeros(len(xyz), bool)
    last = xyz[idx[0]]
    for i in range(len(xyz)):
        if ok[i]:
            last = xyz[i]
        else:
            out[i] = last
    return out.astype(np.float32), ok.copy()


def hold_last_visible(uv, vis, thr):
    """uv (F, 2), vis (F,): keypoints of frames with vis < thr -> last keypoint with vis >= thr (if any)"""
    out = uv.copy()
    last = None
    for i in range(len(uv)):
        if vis[i] >= thr:
            last = uv[i]
        elif last is not None:
            out[i] = last
    return out


class KeypointFilter:
    """Object permanence for ONE object keypoint, label-free (works the same in training and deployment).

    accept the new keypoint if it is confident (conf >= thr) AND, while the gripper is open (the object cannot
    move by itself), it lies within jump_px of the last accepted one. A confident keypoint that jumps is held
    back; if it stays at the new place for `reacq` consecutive frames it is accepted (re-acquire after a real
    move, e.g. the cube was pushed, or a wrong first lock). With the gripper closed the object may move with
    the hand, so only the confidence test applies."""

    def __init__(self, thr=0.75, jump_px=15.0, reacq=5, wh=(768, 432)):
        self.thr, self.jump, self.reacq, self.wh = thr, jump_px, reacq, np.asarray(wh, np.float32)
        self.reset()

    def reset(self):
        self.last, self.cand, self.n = None, None, 0

    def __call__(self, uv, conf, closed):
        uv = np.asarray(uv, np.float32)
        if conf < self.thr:
            self.cand, self.n = None, 0
            return uv if self.last is None else self.last
        if self.last is None or closed or self.jump <= 0 or np.linalg.norm((uv - self.last) * self.wh) <= self.jump:
            self.last, self.cand, self.n = uv, None, 0
            return uv
        if self.cand is not None and np.linalg.norm((uv - self.cand) * self.wh) <= self.jump:
            self.n += 1
        else:
            self.cand, self.n = uv, 1
        if self.n >= self.reacq:
            self.last, self.cand, self.n = uv, None, 0
            return uv
        return self.last


def filter_track(uv, conf, closed, **kw):
    """offline version over a whole episode: uv (F, 2), conf (F,), closed (F,) bool -> filtered uv (F, 2)"""
    f = KeypointFilter(**kw)
    return np.stack([f(uv[i], conf[i], bool(closed[i])) for i in range(len(uv))]).astype(np.float32)


def _ep_index(path):
    return int(os.path.basename(path).split("_")[1].split(".")[0])


class VisionCoreDataset(dcd.DecisionCoreDataset):
    def __init__(self, files, images_root, t_history=dcd.T_HISTORY, augment=False, name="obj_tokens",
                 n_freq=6, use_z=True, pair_swap=0.0, uv_noise_px=0.0, z_noise=0.0, img_wh=(768, 432),
                 vis_thr=0.5, z_std=None, hold_thr=0.5, tok_drop=0.0, min_t=None, oracle=False, vis_source="vis",
                 jump_px=0.0, coord="uv", tok_dir="tokens2", pos_noise_mm=0.0, expert_frames=None):
        super().__init__(files, t_history, augment=augment)
        self.coord = coord
        if min_t:                                   # DAgger episodes: frames before the branch point are history only
            first = [int(min_t.get(os.path.basename(f), 0)) for f in files]
            ok = None
            if expert_frames:                       # multi-intervention DAgger: only expert-controlled rows are targets,
                ok = []                             # and the whole target chunk (t .. t+K) must be expert too
                K = max(1, dcd.CHUNK)
                for f, df in zip(files, self.episode_dfs):
                    iv = expert_frames.get(os.path.basename(f))
                    m = np.ones(len(df) + K + 1, bool) if iv is None else np.zeros(len(df) + K + 1, bool)
                    for a_, b_ in (iv or []):
                        m[a_:b_] = True
                    m[len(df):] = True              # past the end the chunk is padded with the last row
                    good = np.array([m[t:t + K + 1].all() for t in range(len(df))], bool)
                    ok.append(good)
            keep = [i for i, (e, t) in enumerate(self.samples) if t >= first[e] and (ok is None or ok[e][t])]
            self.samples = [self.samples[i] for i in keep]
            self.weights = [self.weights[i] for i in keep]
        self._load_labels(files)
        self.n_freq, self.use_z = n_freq, use_z
        self.pair_swap, self.z_noise = pair_swap, z_noise
        self.hold_thr, self.tok_drop = hold_thr, tok_drop
        self.uv_noise = np.array([uv_noise_px / img_wh[0], uv_noise_px / img_wh[1]], np.float32)
        self.obj, self.ckpt = [], None
        if coord == "xyz":             # two fixed cameras, triangulated (vision/render_encode_2cam.py)
            self.uv_noise = np.full(3, pos_noise_mm / 1000.0, np.float32)
            self._load_xyz(files, Path(images_root).parent / tok_dir, oracle)
            return
        for f, df in zip(files, self.episode_dfs):
            p = Path(images_root) / f"ep_{_ep_index(f):06d}" / f"{name}.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing -- run vision/render_episodes.py and encode_episode_tokens.py")
            d = np.load(p)
            if len(d["uv"]) != len(df):
                raise ValueError(f"{p}: {len(d['uv'])} image frames but {len(df)} rows in {f}")
            ck = str(d["ckpt"])
            if self.ckpt is None:
                self.ckpt = ck
            elif ck != self.ckpt:
                raise ValueError(f"{p} was encoded with {ck}, others with {self.ckpt}")
            if vis_source == "conf":       # label-free heatmap confidence instead of the (untrained) vis head
                if "conf" not in d:
                    raise KeyError(f"{p} has no 'conf' -- re-run vision/encode_episode_tokens.py --overwrite")
                V = d["conf"].astype(np.float32)
            else:
                V = d["vis"].astype(np.float32)
            seen = np.flatnonzero(V[:, 1] > vis_thr)
            g = int(seen[0]) if len(seen) else 0
            pairs = {int(i): k for k, i in enumerate(d["pair_frames"])}
            uv = d["uv"].astype(np.float32)
            if jump_px > 0:               # confidence + object-permanence filter (KeypointFilter), cube only
                closed = df["gripper_closed"].values > 0.5 if "gripper_closed" in df.columns else np.zeros(len(df), bool)
                uv[:, 0] = filter_track(uv[:, 0], V[:, 0], closed, thr=hold_thr, jump_px=jump_px, wh=img_wh)
            elif hold_thr > 0:
                uv[:, 0] = hold_last_visible(uv[:, 0], V[:, 0], hold_thr)
            mp = Path(images_root) / f"ep_{_ep_index(f):06d}" / "meta.npz"
            mt = np.load(mp)
            cam = (mt["cam_pos"].astype(np.float32), mt["cam_xmat"].reshape(-1, 9).astype(np.float32),
                   mt["K"].astype(np.float32))
            uvp = d["uv_pair"].astype(np.float32)
            if jump_px > 0 and len(uvp):          # paired renders: same filtered cube keypoint at that frame
                uvp[:, 0] = uv[np.asarray(d["pair_frames"], int), 0]
            visp = (d["conf_pair"] if vis_source == "conf" else d["vis_pair"]).astype(np.float32)
            if oracle:      # perfect perception: the render's own projections of the cube / tape centre, always seen
                W, H = 2 * mt["K"][0, 0, 2], 2 * mt["K"][0, 1, 2]
                uv = np.stack([mt["cube_uv"] / [W, H], mt["tape_uv"] / [W, H]], 1).astype(np.float32)
                V = np.ones_like(V); g = 0
                uvp = uv[np.asarray(d["pair_frames"], int)].copy(); visp = np.ones_like(visp)
            self.obj.append(dict(cam=cam, uv=uv, vis=V, z=d["z"].astype(np.float32),
                                 uvp=uvp, visp=visp,
                                 zp=d["z_pair"].astype(np.float32), pairs=pairs, goal=g))
        if oracle:
            self.ckpt = "oracle"
        allz = np.concatenate([o["z"].reshape(-1, o["z"].shape[-1]) for o in self.obj])
        self.z_dim = allz.shape[-1]
        self.z_std = z_std if z_std is not None else (allz.std(0).astype(np.float32) if self.z_dim else allz[:0].sum(0))

    def _load_labels(self, files):
        """state-expert labels (dagger_pp_vision.py --trigger state): per frame the expert's chunk for THAT state
        (label_q (K*7), label_grip (K)). They replace the 'next K rows' targets; read from the raw episode file
        (the latent files may not carry the extra columns). Null-space variants are off for these episodes (the
        label is for the recorded elbow)."""
        import pandas as pd
        self.labels = []
        n_lab = 0
        for i, f in enumerate(files):
            raw = Path(str(f).replace("/data_latent_ns/", "/data/").replace("/data_latent/", "/data/"))
            df = self.episode_dfs[i]
            src = df if "label_q" in df.columns else None
            if src is None and raw.exists():
                cols = pd.read_parquet(raw, columns=None).columns
                if "label_q" in cols:
                    src = pd.read_parquet(raw, columns=["label_q", "label_grip"])
            if src is None:
                self.labels.append(None); continue
            if len(src) != len(df):
                raise ValueError(f"{raw}: {len(src)} label rows but {len(df)} rows in {f}")
            lq = np.stack(src["label_q"].values).astype(np.float32)
            lg = np.stack(src["label_grip"].values).astype(np.float32)
            K = lg.shape[1]
            self.labels.append((lq.reshape(len(df), K, -1), lg))
            if hasattr(self, "ns_valid") and self.ns_valid is not None and i < len(self.ns_valid):
                self.ns_valid[i] = None
            n_lab += 1
        if n_lab:
            print(f"[labels] {n_lab}/{len(files)} episodes carry state-expert chunk labels")

    def _load_xyz(self, files, tok_root, oracle):
        """per episode: cube position per frame (held while not observable), tape (goal) position"""
        for f, df in zip(files, self.episode_dfs):
            p = Path(tok_root) / f"ep_{_ep_index(f):06d}.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing -- run vision/render_encode_2cam.py")
            d = np.load(p)
            if len(d["xyz"]) != len(df):
                raise ValueError(f"{p}: {len(d['xyz'])} frames but {len(df)} rows in {f}")
            ck = str(d["encoder"])
            if self.ckpt is None:
                self.ckpt = ck
            elif ck != self.ckpt:
                raise ValueError(f"{p} was encoded with {ck}, others with {self.ckpt}")
            xyz = d["xyz"].astype(np.float32); nv = d["nviews"].astype(np.float32)
            if oracle:                 # perfect perception: true positions, always seen
                xyz = np.stack([d["cube_pos"], np.repeat(d["place_pos"][None], len(df), 0)], 1).astype(np.float32)
                nv = np.full_like(nv, 2.0)
            ok_c = (nv[:, 0] > 0) & np.isfinite(xyz[:, 0]).all(-1)
            cube, fresh = hold_last_valid_xyz(np.nan_to_num(xyz[:, 0]), ok_c)
            vis_c = np.where(fresh, nv[:, 0] / 2.0, 0.0).astype(np.float32)        # 1 = both cameras, 0 = held
            ok_t = (nv[:, 1] > 0) & np.isfinite(xyz[:, 1]).all(-1)
            g = int(np.flatnonzero(ok_t)[0]) if ok_t.any() else 0
            tape = np.nan_to_num(xyz[g, 1]).astype(np.float32)
            pf = np.asarray(d["pair_frames"], int)
            pairs = {int(i): k for k, i in enumerate(pf)}
            cp = cube[pf].copy() if len(pf) else np.zeros((0, 3), np.float32)
            vp = vis_c[pf].copy() if len(pf) else np.zeros(0, np.float32)
            if len(pf) and "xyz_pair" in d and len(d["xyz_pair"]) and not oracle:
                xp, nvp = d["xyz_pair"].astype(np.float32), d["nviews_pair"].astype(np.float32)
                okp = (nvp[:, 0] > 0) & np.isfinite(xp[:, 0]).all(-1)
                cp[okp] = xp[okp, 0]; vp[okp] = nvp[okp, 0] / 2.0
            F = len(df)
            self.obj.append(dict(cam=(np.zeros((F, 3), np.float32), np.zeros((F, 9), np.float32),
                                      np.zeros((F, 3, 3), np.float32)),
                                 uv=np.stack([cube, np.repeat(tape[None], F, 0)], 1),
                                 vis=np.stack([vis_c, np.ones(F, np.float32)], 1),
                                 z=np.zeros((F, 2, 0), np.float32),
                                 uvp=np.stack([cp, np.repeat(tape[None], len(pf), 0)], 1) if len(pf) else np.zeros((0, 2, 3), np.float32),
                                 visp=np.stack([vp, np.ones(len(pf), np.float32)], 1) if len(pf) else np.zeros((0, 2), np.float32),
                                 zp=np.zeros((len(pf), 2, 0), np.float32), pairs=pairs, goal=g))
        if oracle:
            self.ckpt = "oracle"
        self.z_dim = 0
        self.z_std = np.zeros(0, np.float32)

    def token(self, uv, vis, z):
        if self.coord == "xyz":
            return xyz_tokens(uv, vis, self.n_freq)
        return object_tokens(uv, vis, z, self.n_freq, self.use_z)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        e, t = self.samples[idx]
        lab = self.labels[e] if getattr(self, "labels", None) else None
        if lab is not None and "q_chunk" in item:            # the expert's chunk for THIS state
            K = item["q_chunk"].shape[0]
            if lab[0].shape[1] < K:
                raise ValueError(f"labels have chunk {lab[0].shape[1]} < training chunk {K}")
            item["q_chunk"] = torch.as_tensor(lab[0][t, :K, :7].copy())
            item["grip_chunk"] = torch.as_tensor(lab[1][t, :K].copy())
        o = self.obj[e]
        frames = np.clip(np.arange(t - self.t_history + 1, t + 1), 0, None)
        uv, vis, z = o["uv"][frames, 0].copy(), o["vis"][frames, 0].copy(), o["z"][frames, 0].copy()
        g = o["goal"]
        tuv, tvis, tz = o["uv"][g, 1].copy(), o["vis"][g, 1].copy(), o["z"][g, 1].copy()
        if self.augment:
            if self.pair_swap > 0:
                for j, fr in enumerate(frames):
                    if fr in o["pairs"] and np.random.rand() < self.pair_swap:
                        k = o["pairs"][fr]
                        uv[j], vis[j], z[j] = o["uvp"][k, 0], o["visp"][k, 0], o["zp"][k, 0]
            if self.tok_drop > 0:                      # pretend the cube is not seen in some history frames
                for j in range(1, len(frames)):
                    if np.random.rand() < self.tok_drop:
                        uv[j], vis[j] = uv[j - 1], 0.0
            uv += np.random.randn(*uv.shape).astype(np.float32) * self.uv_noise
            tuv += np.random.randn(*tuv.shape).astype(np.float32) * self.uv_noise
            if self.z_noise > 0 and self.z_dim:
                z += np.random.randn(*z.shape).astype(np.float32) * self.z_noise * self.z_std
                tz += np.random.randn(*tz.shape).astype(np.float32) * self.z_noise * self.z_std
        cp, cx, ck = o["cam"]
        item["cam_pos"], item["cam_xmat"], item["cam_K"] = (torch.as_tensor(a[frames]) for a in (cp, cx, ck))
        df = self.episode_dfs[e]                 # task-space context of THIS frame, for the loss weights only
        item["obj_now"] = torch.as_tensor(np.array(df["object_xyz"].iloc[t], np.float32)
                                          if "object_xyz" in df.columns else np.zeros(3, np.float32))
        item["place_now"] = torch.as_tensor(np.array(df["place_xyz"].iloc[t], np.float32)
                                            if "place_xyz" in df.columns else np.zeros(3, np.float32))
        item["closed_now"] = torch.tensor(float(df["gripper_closed"].iloc[t]) if "gripper_closed" in df.columns else 0.0)
        item["cube_hist"] = self.token(uv, vis, z)
        item["tape_goal"] = self.token(tuv, tvis, tz)
        return item


def build_extra_dataset(dataset, t_history=dcd.T_HISTORY, z_std=None, **kw):
    """an extra TRAINING dataset (e.g. DAgger episodes): all its episodes, with augmentation; if
    dagger_meta.json exists, only samples at / after each episode's branch point"""
    import json
    root = Path(os.path.expanduser(f"~/cross_embody/fr3_pick_gen/datasets/{dataset}"))
    ns = root / "data_latent_ns" / "chunk-000"
    latent_dir = ns if ns.exists() else root / "data_latent" / "chunk-000"
    files = sorted(str(p) for p in latent_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episodes in {latent_dir} -- run precompute_latents_pp.py first")
    mp = root / "dagger_meta.json"
    min_t = json.load(open(mp)) if mp.exists() else None
    ep_ = root / "expert_frames.json"
    ef = json.load(open(ep_)) if ep_.exists() else None
    ds = VisionCoreDataset(files, root / "images", t_history, True, z_std=z_std, min_t=min_t, expert_frames=ef, **kw)
    print(f"[extra dataset] {latent_dir}: {len(files)} episodes, {len(ds)} samples"
          f"{' (after the DAgger branch points)' if min_t else ''}")
    return ds


def build_vision_datasets(dataset, t_history=dcd.T_HISTORY, **kw):
    root = Path(os.path.expanduser(f"~/cross_embody/fr3_pick_gen/datasets/{dataset}"))
    ns = root / "data_latent_ns" / "chunk-000"
    latent_dir = ns if ns.exists() else root / "data_latent" / "chunk-000"
    files = sorted(str(p) for p in latent_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episodes in {latent_dir} -- run precompute_latents_pp.py first")
    tr, va = dcd._episode_split(files)
    print(f"[vision dataset] {latent_dir}  images {root / 'images'}  {len(tr)} train / {len(va)} val episodes")
    aug = {k: kw.pop(k) for k in ("pair_swap", "uv_noise_px", "z_noise", "tok_drop", "pos_noise_mm") if k in kw}
    dtr = VisionCoreDataset(tr, root / "images", t_history, True, **kw, **aug)
    dva = VisionCoreDataset(va, root / "images", t_history, False, **kw, z_std=dtr.z_std)
    return dtr, dva