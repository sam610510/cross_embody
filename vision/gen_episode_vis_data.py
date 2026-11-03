#!/usr/bin/env python3
"""Encoder training data from REAL expert trajectories: the gripper around / holding / carrying the cube.

gen_vision_data.py only has the cube resting on the table (random arm postures). In closed loop the encoder also
sees the fingers closing on the cube and the cube in the hand, where it then failed (one camera off by 50 px).
This replays expert episodes (recorded states) in the two-camera set-up and writes, per camera, a dataset in the
SAME format as gen_vision_data.py (rgb/*.png, seg/*.png, labels_*.npz, meta.json) so train_vision_encoder.py can
mix it with the random-scene data. Frames are sampled with emphasis on the grasp and the carry:
  near grasp  TCP within --near mm of the cube, gripper open      (fingers around the cube)
  in hand     gripper closed                                       (closing, lifting, carrying, lowering)
  other       the rest of the episode                              (--p_other of these)
Appearance is randomized per episode like the training renders (table, lights, arm colour, camera mount jitter).

  cd ~/cross_embody/vision
  python gen_episode_vis_data.py $P --episodes ~/cross_embody/fr3_pick_gen/datasets/expert_pp_v3/data/chunk-000 \\
      --first 0 --n_eps 400 --out_prefix data/vis2ep --workers 8 --seed 30            # -> data/vis2ep_camA, _camB
"""
import argparse, glob, json, os, sys, time

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import gen_vision_data as G  # noqa: E402
from render_encode_2cam import CAM_A, CAM_B, LOOKAT  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--first", type=int, default=0, help="first episode (sorted order)")
    ap.add_argument("--n_eps", type=int, default=400)
    ap.add_argument("--out_prefix", required=True, help="writes <prefix>_camA and <prefix>_camB")
    ap.add_argument("--every", type=int, default=5, help="take every N-th frame of the near-grasp / in-hand parts")
    ap.add_argument("--p_other", type=float, default=0.1, help="share of the remaining frames that is kept")
    ap.add_argument("--near", type=float, default=100.0, help="near grasp: TCP-cube distance below this [mm]")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fovy", type=float, default=42.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=30)
    a, rest = ap.parse_known_args()
    common = ["--ee_site", "tcp_site", "--cube", "object", "--table_geom", "table_top", "--width", str(a.width),
              "--height", str(a.height), "--fovy", str(a.fovy), "--cam_name=", "--cam_lookat", *map(str, LOOKAT),
              "--n_occ", "0", "--p_distractor", "0"]
    gs = [G.parse_args(rest + common + ["--cam_pos", *map(str, pos)]) for pos in (CAM_A, CAM_B)]
    return a, gs


def worker(job):
    a, gs, files, wid, idx0 = job
    scenes = [G.Scene(g) for g in gs]
    outs = [f"{a.out_prefix}_cam{c}" for c in "AB"]
    labels = [[], []]
    fid = idx0
    t0 = time.time()
    for n_ep, f in enumerate(files):
        e = int(os.path.basename(f).split("_")[1].split(".")[0])
        df = pd.read_parquet(f)
        st = np.stack(df["observation.state"].values); ob = np.stack(df["object_xyz"].values)
        place = np.asarray(df["place_xyz"].iloc[0], float)
        quat = np.stack(df["object_quat"].values) if "object_quat" in df else None
        closed = df["gripper_closed"].values > 0.5
        rng = np.random.default_rng([a.seed, e])
        for k, sc in enumerate(scenes):
            m, d = sc.m, sc.d
            d.qpos[:] = sc.qpos_init
            sc.randomize_appearance(np.random.default_rng([a.seed, e]), np.random.default_rng([a.seed, e, 1000]))
            rj = np.random.default_rng([a.seed, e, 3000 + k])
            m.cam_pos[sc.cam] = sc.cam_pos0 + rj.normal(0, sc.a.cam_pos_std, 3)
            m.cam_quat[sc.cam] = G.quat_mul(sc.cam_quat0, G.small_rot_quat(rj, np.deg2rad(sc.a.cam_rot_std)))
            sc.tape.set_pose(m, d, place[0], place[1], 0.0)
        scenes[1].m.geom_rgba[scenes[1].robot_geoms] = scenes[0].m.geom_rgba[scenes[0].robot_geoms]
        # frames to keep
        keep = []
        for i in range(len(df)):
            sc = scenes[0]; m, d = sc.m, sc.d
            d.qpos[sc.arm_q] = st[i, :len(sc.arm_q)]
            mujoco.mj_forward(m, d)
            dist = np.linalg.norm(d.site_xpos[sc.ee] - ob[i]) * 1000
            special = closed[i] or dist < a.near
            if (special and i % a.every == 0) or (not special and rng.random() < a.p_other):
                keep.append((i, 2 if (dist < a.near and not closed[i]) else (0 if closed[i] else 1)))
        for i, mode in keep:
            for k, sc in enumerate(scenes):
                m, d = sc.m, sc.d
                d.qpos[sc.arm_q] = st[i, :len(sc.arm_q)]
                for j in sc.finger_j:
                    d.qpos[m.jnt_qposadr[j]] = st[i, len(sc.arm_q)]
                d.qpos[sc.cube.qadr:sc.cube.qadr + 3] = ob[i]
                d.qpos[sc.cube.qadr + 3:sc.cube.qadr + 7] = quat[i] if quat is not None else sc.cube.quat0
                mujoco.mj_forward(m, d)
                cp, cx = d.cam_xpos[sc.cam].copy(), d.cam_xmat[sc.cam].copy()
                K = G.intrinsics(sc.a.width, sc.a.height, m.cam_fovy[sc.cam])
                sc.r_rgb.update_scene(d, camera=sc.cam, scene_option=sc.opt)
                rgb = sc.r_rgb.render().copy()
                sc.r_seg.update_scene(d, camera=sc.cam, scene_option=sc.opt)
                lab = sc.seg_counts(sc.r_seg.render())
                Image.fromarray(rgb).save(os.path.join(outs[k], "rgb", f"{fid:06d}.jpg"), quality=95)   # ~10x smaller
                Image.fromarray(lab).save(os.path.join(outs[k], "seg", f"{fid:06d}.png"), compress_level=1)
                lbl = {"idx": fid, "scene": e, "variant": -1, "mode": mode, "full_occ": False,
                       "q": d.qpos[sc.arm_q].copy(), "ee_pos": d.site_xpos[sc.ee].copy(),
                       "cam_pos": cp, "cam_xmat": cx, "K": K}
                for key, obj, sid in (("cube", sc.cube, G.SEG_CUBE), ("tape", sc.tape, G.SEG_TAPE)):
                    p = obj.pos(d)
                    uv, dep = G.project(p, cp, cx, K)
                    n_vis = int((lab == sid).sum()); n_iso = sc.isolated_count(obj)
                    lbl.update({f"{key}_pos": p, f"{key}_uv": uv, f"{key}_depth": dep, f"{key}_npix": n_iso,
                                f"{key}_vis": n_vis / n_iso if n_iso > 0 else 0.0})
                lbl["cube_yaw"] = 0.0
                lbl["ee_cube_dist"] = float(np.linalg.norm(lbl["ee_pos"] - lbl["cube_pos"]))
                labels[k].append(lbl)
            fid += 1
        if wid == 0:
            print(f"  worker0 episode {n_ep + 1}/{len(files)}: {len(keep)} frames x 2 cameras "
                  f"({(time.time() - t0) / (n_ep + 1):.1f} s/episode)", flush=True)
    return labels


def main():
    a, gs = parse()
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.episodes), "episode_*.parquet")))[a.first:a.first + a.n_eps]
    for c in "AB":
        for sub in ("rgb", "seg"):
            os.makedirs(os.path.join(f"{a.out_prefix}_cam{c}", sub), exist_ok=True)
    chunks = [c.tolist() for c in np.array_split(np.array(files), max(1, a.workers)) if len(c)]
    jobs = [(a, gs, c, w, 10_000_000 + w * 1_000_000) for w, c in enumerate(chunks)]   # disjoint image ids
    if len(jobs) == 1:
        res = [worker(jobs[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            res = pool.map(worker, jobs)
    for k, c in enumerate("AB"):
        L = sorted([l for r in res for l in r[k]], key=lambda l: l["idx"])
        arr = {key: np.stack([np.asarray(l[key]) for l in L]) for key in L[0]}
        out = f"{a.out_prefix}_cam{c}"
        np.savez(os.path.join(out, "labels_episodes.npz"), **arr)
        meta = {"source": "gen_episode_vis_data.py", "episodes": a.episodes, "modes": ["in_hand", "other", "near_grasp"],
                "width": a.width, "height": a.height, "uv_convention": "continuous pixels, pixel i spans [i, i+1), v down",
                "cube_center_z": float(arr["cube_pos"][:, 2].mean()), "tape_center_z": float(arr["tape_pos"][:, 2].mean())}
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        v = arr["cube_vis"]
        print(f"[{out}] {len(L)} images | modes in-hand/other/near {np.bincount(arr['mode'], minlength=3).tolist()} | "
              f"cube visible>0.9 {np.mean(v > 0.9):.2f}, partly {np.mean((v > 0.1) & (v <= 0.9)):.2f}, hidden {np.mean(v <= 0.1):.2f}")


if __name__ == "__main__":
    sys.exit(main())