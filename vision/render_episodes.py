#!/usr/bin/env python3
"""Render d435 images for existing pick-and-place episodes (replay, no new physics).

Reads the parquet episodes written by generate_fr3_pickplace_data.py and, frame by frame, puts the
recorded state back into the vision scene (arm joints + finger, cube position, place target) and
renders the camera. The expert data stay exactly as they are; only images are added.

Per episode (seeded by --seed and the episode index):
  scene randomization once (table, lights, camera jitter, cube/tape colour)  -- like gen_vision_data
  arm colour once per episode, and every --pair_every frames a second render of the SAME frame with
  another arm colour (positive pair for the invariance loss)
  no floating occluders / distractors (the deployment scene does not have them)

Output (next to the dataset by default):
  <out>/ep_000123/rgb/00042.jpg          main render of frame 42
  <out>/ep_000123/pair/00040.jpg         same frame, other arm colour (every --pair_every frames)
  <out>/ep_000123/meta.npz               cam_pos, cam_xmat, K, cube_uv/tape_uv per frame, pair frames

  python render_episodes.py $P $A --episodes ~/cross_embody/fr3_pick_gen/datasets/expert_pp_v3/data/chunk-000 \
      --out ~/cross_embody/fr3_pick_gen/datasets/expert_pp_v3/images --workers 16
"""
import argparse
import glob
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import gen_vision_data as G  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True, help="directory with episode_*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair_every", type=int, default=5, help="0 = no paired renders")
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="episode indices (debug)")
    a, rest = ap.parse_known_args()
    g = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"])   # scene / camera / DR flags
    return a, g


def episode_job(job):
    a, g, files, wid = job
    sc = G.Scene(g)
    m, d = sc.m, sc.d
    fing = [m.jnt_qposadr[j] for j in sc.finger_j]
    t0, n_done = time.time(), 0
    for f in files:
        e = int(os.path.basename(f).split("_")[1].split(".")[0])
        out = os.path.join(a.out, f"ep_{e:06d}")
        if os.path.exists(os.path.join(out, "meta.npz")):
            continue
        os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
        if a.pair_every:
            os.makedirs(os.path.join(out, "pair"), exist_ok=True)
        df = pd.read_parquet(f)
        state = np.stack(df["observation.state"].values)                       # (F, 8): arm 7 + finger
        obj = np.stack(df["object_xyz"].values)
        place = np.asarray(df["place_xyz"].iloc[0], float)
        quat = np.stack(df["object_quat"].values) if "object_quat" in df else None

        rs = np.random.default_rng([g.seed, e])
        ra = np.random.default_rng([g.seed, e, 1000])
        rp = np.random.default_rng([g.seed, e, 2000])
        d.qpos[:] = sc.qpos_init
        sc.randomize_appearance(rs, ra)                                           # once per episode
        arm_main = m.geom_rgba[sc.robot_geoms].copy()
        sc.tape.set_pose(m, d, place[0], place[1], 0.0)
        cams, uvs, pairs = [], [], []
        for i in range(len(df)):
            d.qpos[sc.arm_q] = state[i, :len(sc.arm_q)]
            for q in fing:
                d.qpos[q] = state[i, len(sc.arm_q)]
            d.qpos[sc.cube.qadr:sc.cube.qadr + 3] = obj[i]
            d.qpos[sc.cube.qadr + 3:sc.cube.qadr + 7] = quat[i] if quat is not None else sc.cube.quat0
            mujoco.mj_forward(m, d)
            cp, cx = d.cam_xpos[sc.cam].copy(), d.cam_xmat[sc.cam].copy()
            K = G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam])
            sc.r_rgb.update_scene(d, camera=sc.cam, scene_option=sc.opt)
            Image.fromarray(sc.r_rgb.render()).save(os.path.join(out, "rgb", f"{i:05d}.jpg"), quality=a.jpeg_quality)
            if a.pair_every and i % a.pair_every == 0:
                sc.randomize_arm(rp)
                sc.r_rgb.update_scene(d, camera=sc.cam, scene_option=sc.opt)
                Image.fromarray(sc.r_rgb.render()).save(os.path.join(out, "pair", f"{i:05d}.jpg"),
                                                        quality=a.jpeg_quality)
                m.geom_rgba[sc.robot_geoms] = arm_main
                pairs.append(i)
            cams.append((cp, cx, K))
            uvs.append([G.project(sc.cube.pos(d), cp, cx, K)[0], G.project(sc.tape.pos(d), cp, cx, K)[0]])
        np.savez(os.path.join(out, "meta.npz"), cam_pos=np.stack([c[0] for c in cams]),
                 cam_xmat=np.stack([c[1] for c in cams]), K=np.stack([c[2] for c in cams]),
                 cube_uv=np.array([u[0] for u in uvs]), tape_uv=np.array([u[1] for u in uvs]),
                 cube_pos=obj, place_pos=place, pair_frames=np.array(pairs, int), n_frames=len(df))
        n_done += 1
        if wid == 0:
            print(f"  worker0 episode {e} ({len(df)} frames) done, {n_done}/{len(files)} "
                  f"({(time.time() - t0) / n_done:.1f} s/episode)", flush=True)
    return n_done


def main():
    a, g = parse()
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.episodes), "episode_*.parquet")))
    if a.only is not None:
        files = [f for f in files if int(os.path.basename(f).split("_")[1].split(".")[0]) in a.only]
    if not files:
        raise SystemExit(f"no episode_*.parquet in {a.episodes}")
    a.out = os.path.expanduser(a.out)
    os.makedirs(a.out, exist_ok=True)
    chunks = [c.tolist() for c in np.array_split(np.array(files), max(1, g.workers)) if len(c)]
    jobs = [(a, g, c, w) for w, c in enumerate(chunks)]
    t0 = time.time()
    if len(jobs) == 1:
        n = episode_job(jobs[0])
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            n = sum(pool.map(episode_job, jobs))
    print(f"rendered {n} episodes -> {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())