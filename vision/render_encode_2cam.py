#!/usr/bin/env python3
"""Two fixed cameras: render each episode frame, encode it, triangulate -- keep ONLY the numbers, no images.

For every episode_*.parquet (expert or DAgger data), replays the recorded states in two scenes (camera A and B,
same world: same colours / lights per episode, each camera with its own extrinsic jitter = a slightly-off mount
that has been calibrated) and for every frame:
  render A and B (JPEG round trip, like the training images) -> encoder -> keypoint (u, v) + heatmap confidence
  per camera for cube and tape -> triangulate.py -> 3D position of cube and tape in the world / robot-base frame
Single-view fallback: the tape always lies on the table (z = table); the cube only while the gripper is open.
Every --pair_every frames the arm gets other colours and the frame is encoded again (pair_swap augmentation).

Output: <out>/ep_000123.npz
  uv (F, cam, obj, 2) px | conf (F, cam, obj) | xyz (F, obj, 3) m, NaN if not observable | nviews (F, obj) 0/1/2
  reproj (F, obj) px | cam_K (cam, 3, 3) | cam_pos (cam, 3) | cam_xmat (cam, 9)   [what calibration would give]
  true_uv (F, cam, obj, 2) | cube_pos (F, 3) | place_pos (3,)                       [labels, diagnostics only]
  pair_frames (P,) | uv_pair (P, cam, obj, 2) | conf_pair | xyz_pair | nviews_pair
obj 0 = cube, 1 = tape.  About 10 kB per episode.

  cd ~/cross_embody/vision
  python render_encode_2cam.py $P --encoder runs/vis2_fusion/best.pt \\
      --episodes ~/cross_embody/fr3_pick_gen/datasets/expert_pp_v3/data/chunk-000 \\
      --out ~/cross_embody/fr3_pick_gen/datasets/expert_pp_v3/tokens2 --workers 4
"""
import argparse, glob, io, os, sys, time

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import gen_vision_data as G  # noqa: E402
from triangulate import triangulate  # noqa: E402

CAM_A = [0.85, 0.65, 0.50]
CAM_B = [0.85, -0.65, 0.50]
LOOKAT = [0.475, 0.025, 0.0]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--camA", type=float, nargs=3, default=CAM_A)
    ap.add_argument("--camB", type=float, nargs=3, default=CAM_B)
    ap.add_argument("--lookat", type=float, nargs=3, default=LOOKAT)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fovy", type=float, default=42.0)
    ap.add_argument("--pair_every", type=int, default=5)
    ap.add_argument("--jpeg", type=int, default=92)
    ap.add_argument("--min_conf", type=float, default=0.3, help="a view is used for triangulation above this confidence")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a, rest = ap.parse_known_args()
    common = ["--ee_site", "tcp_site", "--cube", "object", "--table_geom", "table_top", "--width", str(a.width),
              "--height", str(a.height), "--fovy", str(a.fovy), "--cam_name=", "--cam_lookat", *map(str, a.lookat),
              "--n_occ", "0", "--p_distractor", "0"]
    gs = [G.parse_args(rest + common + ["--cam_pos", *map(str, pos)]) for pos in (a.camA, a.camB)]
    return a, gs


def jpeg(rgb, q):
    if not q:
        return rgb
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return np.array(Image.open(buf).convert("RGB"))


def worker(job):
    a, gs, files, wid = job
    import torch
    import torch.nn.functional as F
    from object_latent import load_any_encoder
    from kp_conf import heatmap_conf
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, in_hw, _ = load_any_encoder(a.encoder, dev)
    scenes = [G.Scene(g) for g in gs]
    W, H = a.width, a.height
    table_z = scenes[0].a.table_z or 0.0
    half = scenes[0].cube_half_h

    @torch.no_grad()
    def encode(imgs):
        UV, C = [], []
        for s in range(0, len(imgs), a.batch):
            x = torch.from_numpy(np.stack(imgs[s:s + a.batch])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            if tuple(x.shape[-2:]) != tuple(in_hw):
                x = F.interpolate(x, size=tuple(in_hw), mode="bilinear", align_corners=False)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = enc(x)
            uv = o["uv"].float() / torch.tensor([in_hw[1], in_hw[0]], device=dev) * torch.tensor([W, H], device=dev)
            UV.append(uv.cpu().numpy()); C.append(heatmap_conf(o["prob"]).cpu().numpy())
        return np.concatenate(UV), np.concatenate(C)          # (N, obj, 2) px, (N, obj)

    def tri(uv2, c2, cams, closed, prior=None):
        """prior: last cube position -- used only while the gripper is closed (cube in the hand, no table plane)"""
        xyz, nv, err = np.full((2, 3), np.nan), np.zeros(2, int), np.full(2, np.nan)
        for k in range(2):
            plane = (table_z + half) if k == 0 and not closed else (table_z if k == 1 else None)
            pr = prior if (k == 0 and closed and prior is not None and np.all(np.isfinite(prior))) else None
            q, info = triangulate(uv2[:, k], c2[:, k], cams, plane_z=plane, min_w=a.min_conf, prior=pr)
            if q is not None:
                xyz[k], nv[k] = q, len(info["views"])
                err[k] = float(np.max(info["err"])) if info["err"] is not None else np.nan
        return xyz, nv, err

    t0, n_done = time.time(), 0
    for f in files:
        e = int(os.path.basename(f).split("_")[1].split(".")[0])
        out = os.path.join(a.out, f"ep_{e:06d}.npz")
        if os.path.exists(out) and not a.overwrite:
            continue
        df = pd.read_parquet(f)
        st = np.stack(df["observation.state"].values); ob = np.stack(df["object_xyz"].values)
        place = np.asarray(df["place_xyz"].iloc[0], float)
        quat = np.stack(df["object_quat"].values) if "object_quat" in df else None
        closed = df["gripper_closed"].values > 0.5 if "gripper_closed" in df else np.zeros(len(df), bool)
        Fn = len(df)
        cams = []
        for k, sc in enumerate(scenes):                      # same world for both cameras, own mount jitter
            m, d = sc.m, sc.d
            d.qpos[:] = sc.qpos_init
            sc.randomize_appearance(np.random.default_rng([a.seed, e]), np.random.default_rng([a.seed, e, 1000]))
            rj = np.random.default_rng([a.seed, e, 3000 + k])
            m.cam_pos[sc.cam] = sc.cam_pos0 + rj.normal(0, sc.a.cam_pos_std, 3)
            m.cam_quat[sc.cam] = G.quat_mul(sc.cam_quat0, G.small_rot_quat(rj, np.deg2rad(sc.a.cam_rot_std)))
            m.cam_fovy[sc.cam] = sc.fovy0 + rj.normal(0, sc.a.fovy_std)
            sc.tape.set_pose(m, d, place[0], place[1], 0.0)
            mujoco.mj_forward(m, d)
            cams.append((G.intrinsics(W, H, m.cam_fovy[sc.cam]), d.cam_xpos[sc.cam].copy(), d.cam_xmat[sc.cam].copy()))
        arm0 = [sc.m.geom_rgba[sc.robot_geoms].copy() for sc in scenes]
        rp = np.random.default_rng([a.seed, e, 2000])
        imgs, true_uv, pair_imgs, pairs = [], np.zeros((Fn, 2, 2, 2), np.float32), [], []
        enc_uv, enc_c = [], []                                # encoded in chunks: never hold a whole episode of images
        for i in range(Fn):
            if len(imgs) >= 4 * a.batch:
                u_, c_ = encode(imgs); enc_uv.append(u_); enc_c.append(c_); imgs = []
            for k, sc in enumerate(scenes):
                m, d = sc.m, sc.d
                d.qpos[sc.arm_q] = st[i, :len(sc.arm_q)]
                for j in sc.finger_j:
                    d.qpos[m.jnt_qposadr[j]] = st[i, len(sc.arm_q)]
                d.qpos[sc.cube.qadr:sc.cube.qadr + 3] = ob[i]
                d.qpos[sc.cube.qadr + 3:sc.cube.qadr + 7] = quat[i] if quat is not None else sc.cube.quat0
                mujoco.mj_forward(m, d)
                sc.r_rgb.update_scene(d, camera=sc.cam, scene_option=sc.opt)
                imgs.append(jpeg(sc.r_rgb.render(), a.jpeg))
                K, cp, cx = cams[k]
                true_uv[i, k, 0] = G.project(sc.cube.pos(d), cp, cx, K)[0]
                true_uv[i, k, 1] = G.project(sc.tape.pos(d), cp, cx, K)[0]
            if a.pair_every and i % a.pair_every == 0:
                cols = None
                for k, sc in enumerate(scenes):              # the same other arm colours in both cameras
                    if cols is None:
                        sc.randomize_arm(rp); cols = sc.m.geom_rgba[sc.robot_geoms].copy()
                    else:
                        sc.m.geom_rgba[sc.robot_geoms] = cols
                    sc.r_rgb.update_scene(sc.d, camera=sc.cam, scene_option=sc.opt)
                    pair_imgs.append(jpeg(sc.r_rgb.render(), a.jpeg))
                    sc.m.geom_rgba[sc.robot_geoms] = arm0[k]
                pairs.append(i)
        if imgs:
            u_, c_ = encode(imgs); enc_uv.append(u_); enc_c.append(c_)
        uv, conf = np.concatenate(enc_uv), np.concatenate(enc_c)
        uv = uv.reshape(Fn, 2, 2, 2); conf = conf.reshape(Fn, 2, 2)
        xyz = np.zeros((Fn, 2, 3), np.float32); nv = np.zeros((Fn, 2), np.int8); rep = np.zeros((Fn, 2), np.float32)
        last = None
        for i in range(Fn):
            xyz[i], nv[i], rep[i] = tri(uv[i], conf[i], cams, bool(closed[i]), last)
            if np.all(np.isfinite(xyz[i, 0])):
                last = xyz[i, 0].copy()
        P = len(pairs)
        if P:
            uvp, confp = encode(pair_imgs)
            uvp = uvp.reshape(P, 2, 2, 2); confp = confp.reshape(P, 2, 2)
            xyzp = np.zeros((P, 2, 3), np.float32); nvp = np.zeros((P, 2), np.int8)
            for n, i in enumerate(pairs):
                xyzp[n], nvp[n], _ = tri(uvp[n], confp[n], cams, bool(closed[i]), xyz[max(0, i - 1), 0])
        else:
            uvp = confp = xyzp = nvp = np.zeros(0)
        np.savez(out, uv=uv.astype(np.float32), conf=conf.astype(np.float16), xyz=xyz, nviews=nv, reproj=rep,
                 cam_K=np.stack([c[0] for c in cams]), cam_pos=np.stack([c[1] for c in cams]),
                 cam_xmat=np.stack([c[2] for c in cams]), true_uv=true_uv, cube_pos=ob.astype(np.float32),
                 place_pos=place, pair_frames=np.array(pairs, int), uv_pair=uvp, conf_pair=confp, xyz_pair=xyzp,
                 nviews_pair=nvp, encoder=a.encoder, width=W, height=H)
        n_done += 1
        if wid == 0:
            ce = np.linalg.norm(xyz[:, 0] - ob, axis=1) * 1000
            te = np.linalg.norm(xyz[0, 1, :2] - place[:2]) * 1000
            print(f"  worker0 ep {e} ({Fn} frames) cube 3D err median {np.nanmedian(ce):.1f} mm (p95 "
                  f"{np.nanpercentile(ce, 95):.1f}), tape {te:.1f} mm, cube 2 views {100 * np.mean(nv[:, 0] == 2):.0f}% | "
                  f"{n_done}/{len(files)} ({(time.time() - t0) / n_done:.1f} s/ep)", flush=True)
    return n_done


def main():
    a, gs = parse()
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.episodes), "episode_*.parquet")))
    if a.only is not None:
        files = [f for f in files if int(os.path.basename(f).split("_")[1].split(".")[0]) in a.only]
    if not files:
        raise SystemExit(f"no episode_*.parquet in {a.episodes}")
    a.out = os.path.expanduser(a.out); a.encoder = os.path.abspath(os.path.expanduser(a.encoder))
    os.makedirs(a.out, exist_ok=True)
    chunks = [c.tolist() for c in np.array_split(np.array(files), max(1, a.workers)) if len(c)]
    jobs = [(a, gs, c, w) for w, c in enumerate(chunks)]
    t0 = time.time()
    if len(jobs) == 1:
        n = worker(jobs[0])
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            n = sum(pool.map(worker, jobs))
    # summary over everything in --out
    errs, tapes, two = [], [], []
    for f in sorted(glob.glob(os.path.join(a.out, "ep_*.npz")))[:300]:
        z = np.load(f)
        errs.append(np.linalg.norm(z["xyz"][:, 0] - z["cube_pos"], axis=1) * 1000)
        tapes.append(np.linalg.norm(z["xyz"][0, 1, :2] - z["place_pos"][:2]) * 1000)
        two.append(np.mean(z["nviews"][:, 0] == 2))
    if errs:
        E = np.concatenate(errs)
        print(f"[summary] cube 3D error median {np.nanmedian(E):.2f} mm, p95 {np.nanpercentile(E, 95):.2f} mm, "
              f"not observable {100 * np.mean(~np.isfinite(E)):.1f}% | tape (goal) median {np.median(tapes):.2f} mm, "
              f"max {np.max(tapes):.2f} mm | cube seen by both cameras {100 * np.mean(two):.0f}% of frames")
    print(f"encoded {n} episodes -> {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())