"""
FR3 pick-and-place, closed loop in MuJoCo, with the VISION decision core (train_pp_vision_e2e.py).

Same control loop as eval_pp_fr3.py (latent history + gripper -> chunk -> temporal ensembling ->
inverse decode -> TCP intent + own IK -> command shaping -> joint tracker), but the cube and the
place point are NOT given: every control step the d435 is rendered, the image encoder turns it into
the cube / tape latents, and the core reads
  cube tokens = cube latents of the last T rendered frames
  goal query  = tape latent of the first frame of the episode (as in training)

Scene appearance per episode as in the training renders (render_episodes.py), or --clean.

Run (same scene flags as render_episodes.py):
  P="--pick_gen $HOME/cross_embody/fr3_pick_gen/scripts --scene_module build_scene --cam_name d435"
  A="--ee_site tcp_site --cube object --table_geom table_top --width 768 --height 432"
  MUJOCO_GL=egl python decision_core/eval_pp_vision.py --ckpt checkpoints/decision_core_vision_e2e_....pt \
      --episodes 50 $P $A
  ... --target_split holdout --holdout_which place        (place point inside the held-out box)
  ... --video 3                                           (d435 view with overlay, validation_out/)
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import mujoco

R = os.path.expanduser("~/cross_embody")
for p in (R, os.path.join(R, "fr3_pick_gen", "scripts"), os.path.join(R, "vision")):
    sys.path.insert(0, p)
from build_scene import OBJECT_HALF_SIZE
from generate_fr3_expert_data import (resolve_joint_and_actuator_ids, sample_random_init, HOME_Q, CONTROL_HZ,
                                      SIM_HZ, INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES,
                                      GRIPPER_OPEN, GRIPPER_CLOSED)
from decision_core_dataset import T_HISTORY, PADDING_NOISE_STD
from decision_core.model import load_frozen_stage1, inverse_decode, tcp_ik
from decision_core.model_chunk import ChunkRunner
from decision_core.vision_core import load_vision_core, object_tokens, tcp_tokens, xyz_tokens
from decision_core.anchor_blend import AnchorBlend, add_args as add_anchor_args
from decision_core.new_arm import NewArm, add_args as add_arm_args, scene_flags, robot_entry
from decision_core.vision_dataset import KeypointFilter
from joint_tracker import JointTracker, FR3_TAU, CommandShaper
from object_latent import load_any_encoder
from kp_conf import heatmap_conf
import gen_vision_data as G

ap = argparse.ArgumentParser(allow_abbrev=False)   # --cube etc. belong to the scene flags
ap.add_argument("--ckpt", required=True)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--steps", type=int, default=900)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--clean", action="store_true", help="no appearance randomization (original colours)")
ap.add_argument("--test_scene", action="store_true",
                help="deployment-like test: the arm keeps its real colours and the camera sits at its nominal "
                     "(calibrated) pose every episode; table / lights / cube colour still vary")
ap.add_argument("--cube_x", type=float, nargs=2, default=[0.30, 0.65])
ap.add_argument("--cube_y", type=float, nargs=2, default=[-0.30, 0.35])
ap.add_argument("--place_x", type=float, nargs=2, default=[0.30, 0.65])
ap.add_argument("--place_y", type=float, nargs=2, default=[-0.30, 0.35])
ap.add_argument("--place_z", type=float, default=0.0)
ap.add_argument("--min_sep", type=float, default=0.12)
ap.add_argument("--holdout_box", type=float, nargs=4, default=[0.44, 0.54, 0.00, 0.15])
ap.add_argument("--target_split", choices=["train", "holdout"], default="train")
ap.add_argument("--holdout_which", choices=["cube", "place", "both"], default="place")
ap.add_argument("--chunk_exec", type=int, default=1)
ap.add_argument("--no_ensemble", action="store_true")
ap.add_argument("--ens_m", type=float, default=0.1)
add_anchor_args(ap)
add_arm_args(ap)
ap.add_argument("--smooth", type=float, default=0.5)
ap.add_argument("--accel_steps", type=int, default=8)
ap.add_argument("--video", type=int, default=0, help="save the d435 view of the first N episodes")
ap.add_argument("--overlay", action="store_true",
                help="with --video: draw what the perception gives the core -- cube (red) / tape (green) heatmaps, "
                     "the raw encoder keypoint (circle), the keypoint the core actually gets after the hold (cross), the "
                     "true projection (white dot), confidence and pixel error per object")
ap.add_argument("--from_dataset", choices=["", "train", "val"], default="",
                help="replay the cube / place / start pose of episodes from the decision-core dataset split: 'train' "
                     "fails too = closed-loop compounding; only 'val'/random fails = generalization")
ap.add_argument("--jpeg", type=int, default=92, help="JPEG-compress each frame like the training images (0 = off)")
ap.add_argument("--video_dir", default=None)
ap.add_argument("--reflex", type=float, default=0.0,
                help="emulate the FR3 collision reflex: stop the episode (failure 'collision') when an OPEN gripper / the "
                     "arm presses on the cube or the table with more than this normal force [N] (e.g. 20); 0 = only report")
ap.add_argument("--encoder_kp", default="",
                help="for an ORACLE-trained core: feed it the keypoints of this image encoder instead of the exact ones "
                     "(visibility 1, no z, no hold) -- same core, real perception")
ap.add_argument("--viewer", action="store_true",
                help="live MuJoCo window (needs a display: run on the server's own screen or ssh -X, with MUJOCO_GL=glfw)")
ap.add_argument("--realtime", type=float, default=1.0, help="with --viewer: playback speed (1 = real time, 0 = as fast as possible)")
ap.add_argument("--succ_mm", type=float, default=10.0,
                help="success = cube within this xy distance of the place point (and resting, gripper open); "
                     "the summary also prints the rate at 20 mm (the old threshold) for comparison")
ap.add_argument("--allow_roll", action="store_true",
                help="count a cube that rolled onto another face but lies flat on the target as a success (old behaviour)")
ap.add_argument("--place_yaw_deg", type=float, default=0.0,
                help="TASK: the place target's yaw is uniform in +- this [deg] (0 = task A, aligned with the world)")
ap.add_argument("--start_yaw_deg", type=float, default=0.0, help="TASK: the cube's start yaw is uniform in +- this [deg]")
ap.add_argument("--succ_yaw_deg", type=float, default=10.0,
                help="success also needs the cube square to the tape: |cube yaw - tape yaw| mod 90 deg below this "
                     "(0 = do not check)")
ap.add_argument("--trace", type=int, default=0,
                help="print a step-by-step trace (every --trace_every steps) for the first N episodes")
ap.add_argument("--trace_every", type=int, default=10)
ap.add_argument("--only", type=int, nargs="*", default=None,
                help="run only these episode indices (same targets / start poses as in the full run), e.g. --only 5 10 13")
args, rest = ap.parse_known_args()
if args.robot != "fr3":                        # same scene, the robots.yaml arm instead of the FR3 (build_scene_arm.py)
    import yaml
    sys.path.insert(0, os.path.join(R, "scripts"))
    import train_stage2_align as _TA
    args.robot = robot_entry(args.robot)
    rest = rest + scene_flags(yaml.safe_load(open(_TA.ROBOTS_YAML))[args.robot])
    os.environ["NEW_ARM"] = args.robot
    if args.from_dataset:
        raise SystemExit("--from_dataset replays FR3 start poses; not available with --robot <new arm>")
g = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"])
# coord = "xyz" cores: two fixed cameras (camera A drives the physics scene, camera B renders a copy of it),
# keypoints triangulated to 3D every step -- the same set-up as vision/render_encode_2cam.py
XYZ = torch.load(os.path.join(R, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt, map_location="cpu",
                 weights_only=False)["config"].get("coord", "uv") == "xyz"
gB = None
if XYZ:
    from render_encode_2cam import CAM_A, CAM_B, LOOKAT
    from triangulate import triangulate
    _cam = ["--width", "960", "--height", "540", "--fovy", "42", "--cam_name=", "--cam_lookat", *map(str, LOOKAT)]
    g = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"] + _cam + ["--cam_pos", *map(str, CAM_A)])
    gB = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"] + _cam + ["--cam_pos", *map(str, CAM_B)])

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VSTEP = torch.tensor([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], device=DEV) / CONTROL_HZ
SUB = max(1, round(SIM_HZ / CONTROL_HZ))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def sample_targets(rng):
    x0, x1, y0, y1 = args.holdout_box
    inside = lambda p: x0 < p[0] < x1 and y0 < p[1] < y1
    in_box = lambda z: np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), z])
    for _ in range(10000):
        cube = np.array([rng.uniform(*args.cube_x), rng.uniform(*args.cube_y), OBJECT_HALF_SIZE])
        place = np.array([rng.uniform(*args.place_x), rng.uniform(*args.place_y), args.place_z])
        if args.target_split == "holdout":
            if args.holdout_which in ("cube", "both"):
                cube = in_box(OBJECT_HALF_SIZE)
            if args.holdout_which in ("place", "both"):
                place = in_box(args.place_z)
            ok = args.holdout_which == "both" or not inside(cube if args.holdout_which == "place" else place)
        else:
            ok = not inside(cube) and not inside(place)
        if ok and np.linalg.norm(place[:2] - cube[:2]) >= args.min_sep:
            return cube, place
    raise RuntimeError("could not sample targets")


def load(path):
    core, c = load_vision_core(path, DEV)
    if c.get("oracle_kp") and args.encoder_kp:
        p = args.encoder_kp if os.path.isabs(args.encoder_kp) else os.path.join(R, args.encoder_kp)
        enc, in_hw, _ = load_any_encoder(p, DEV)
        return core, c, enc, in_hw
    if c.get("oracle_kp"):
        return core, c, None, (g.height, g.width)
    enc, in_hw, _ = load_any_encoder(c["encoder_ckpt"], DEV)
    return core, c, enc, in_hw


@torch.no_grad()
def main():
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    sc = G.Scene(g)
    m, d = sc.m, sc.d
    scB = G.Scene(gB) if XYZ else None           # second camera: a render-only copy, synced every step
    xyz_now = {"cube": None, "tape": None}
    if XYZ:
        print(f"[vision] coord xyz: cameras A {CAM_A} and B {CAM_B}, 960x540, triangulated; the 'kp err' numbers "
              f"below are 3D errors in mm")
    fk, emb, _, _ = load_frozen_stage1(device=DEV)
    UR = args.robot != "fr3"                               # (UR = any new arm)
    if UR:                                                 # cross-arm: the new arm's joints / tracker / gripper / IK
        arm = NewArm(sc, fk, emb, args, DEV, CONTROL_HZ, SIM_HZ, OBJECT_HALF_SIZE)
        qadr, vadr, trk = arm.qadr, arm.vadr, arm.trk
        jnt_ids = arm_act = grip_act = g_qadr = lim = None
        tcp_fk = arm.tcp                                   # TCP of the arm that is actually there
    else:
        jnt_ids, qadr, arm_act, grip_act, g_qadr = resolve_joint_and_actuator_ids(m)
        qadr = np.asarray(qadr); vadr = np.array([m.jnt_dofadr[j] for j in jnt_ids])
        trk = JointTracker(m, qadr, vadr, FR3_TAU, actuator_ids=arm_act)
        lim = torch.tensor(m.jnt_range[jnt_ids], dtype=torch.float32, device=DEV)
        tcp_fk = fk.forward
    finger_now = (lambda: arm.finger()) if UR else (lambda: float(d.qpos[g_qadr]) if g_qadr is not None else 0.0)
    o_adr = sc.cube.qadr
    obj = sc.cube.bid
    fing = [m.jnt_qposadr[j] for j in sc.finger_j]
    core, cfg, enc, in_hw = load(args.ckpt)
    K = int(cfg["chunk"])
    runner = ChunkRunner(K, args.chunk_exec, not args.no_ensemble, args.ens_m)
    blend = AnchorBlend(core, args)
    anc_store = [None]
    if blend.on:
        print(f"[anchor] final approach from the anchor head: k={blend.k}, anchor only < {1000 * blend.r_in:.0f} mm, "
              f"residual only > {1000 * blend.r_out:.0f} mm")
    print(f"[vision] {args.ckpt}  K={K}  token {'kp+z' if cfg['use_z'] else 'kp'}  encoder {cfg['encoder_ckpt']} "
          f"input {in_hw}  appearance {'clean' if args.clean else 'randomized'}")
    zarm = arm.z if UR else (lambda q: emb(fk.forward(q)[:, :18])[0])
    use_tcp = bool(cfg.get("use_tcp", False))
    rng_t = np.random.default_rng(args.seed + 200)

    ds_eps = []
    if args.from_dataset:
        import glob, pandas as pd
        from decision_core_dataset import _episode_split
        root = os.path.join(R, "fr3_pick_gen", "datasets", cfg["dataset"])
        files = sorted(glob.glob(os.path.join(root, "data", "chunk-000", "episode_*.parquet")))
        tr_f, va_f = _episode_split(files)
        for fpath in (tr_f if args.from_dataset == "train" else va_f)[:args.episodes]:
            dfe = pd.read_parquet(fpath, columns=["observation.state", "object_xyz", "place_xyz"])
            ds_eps.append(dict(q0=np.asarray(dfe["observation.state"].iloc[0], np.float64)[:7],
                               cube=np.asarray(dfe["object_xyz"].iloc[0], np.float64),
                               place=np.asarray(dfe["place_xyz"].iloc[0], np.float64)))
        print(f"[vision] replaying {len(ds_eps)} {args.from_dataset} episodes (cube, place, start pose)")
    hold = float(cfg.get("hold_thr", 0.0))
    print(f"[vision] visibility = {cfg.get('vis_source', 'vis')}; hold last confident cube keypoint below {hold}, "
          f"JPEG {args.jpeg or 'off'}")
    last_uv = [None]
    jump = float(cfg.get("jump_px", 0.0))
    kpf = KeypointFilter(thr=hold, jump_px=jump, wh=(g.width, g.height)) if jump > 0 else None
    gstate = [False]                                      # gripper commanded closed (for the filter)
    if kpf is not None:
        print(f"[vision] object-permanence filter: conf >= {hold}, jump <= {jump:.0f} px while the gripper is open")
    ep_video = [False]
    dbg = {}

    def overlay(rgb, step, grip_closed, tape_goal_uv):
        """perception debug frame: heatmaps + keypoints + true projections"""
        from PIL import Image, ImageDraw
        H_, W_ = rgb.shape[:2]
        img = rgb.astype(np.float32)
        if "prob" in dbg:
            for k, col in ((0, (255, 40, 40)), (1, (40, 255, 40))):
                h = dbg["prob"][k]; h = h / (h.max() + 1e-12)
                h = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize((W_, H_), Image.BILINEAR)) / 255.0
                a = (0.6 * h)[..., None]
                img = img * (1 - a) + np.array(col, np.float32) * a
        im = Image.fromarray(img.clip(0, 255).astype(np.uint8)); dr = ImageDraw.Draw(im)
        Kc = G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam])
        true = [G.project(p, d.cam_xpos[sc.cam], d.cam_xmat[sc.cam], Kc)[0] for p in (d.xpos[obj], sc.tape.pos(d))]
        sx, sy = W_ / g.width, H_ / g.height
        lines = [f"t {step:4d}  gripper {'CLOSED' if grip_closed else 'open'}"]
        for k, (name, col) in enumerate((("cube", (255, 80, 80)), ("tape", (80, 255, 80)))):
            tx, ty = true[k][0] * sx, true[k][1] * sy
            dr.ellipse([tx - 3, ty - 3, tx + 3, ty + 3], fill=(255, 255, 255))
            if "uv_raw" in dbg:
                rx, ry = dbg["uv_raw"][k][0] * W_, dbg["uv_raw"][k][1] * H_
                dr.ellipse([rx - 8, ry - 8, rx + 8, ry + 8], outline=col, width=2)
                ux, uy = (dbg["uv_used"][k] if k == 0 else tape_goal_uv) * [W_, H_]
                dr.line([ux - 9, uy, ux + 9, uy], fill=(255, 255, 0), width=2)
                dr.line([ux, uy - 9, ux, uy + 9], fill=(255, 255, 0), width=2)
                held = k == 0 and np.abs(dbg["uv_used"][0] - dbg["uv_raw"][0]).max() > 1e-6
                lines.append(f"{name}: conf {dbg['conf'][k]:.2f}  raw err {np.hypot(rx - tx, ry - ty) / sx:5.1f}px  "
                             f"core gets err {np.hypot(ux - tx, uy - ty) / sx:5.1f}px" + ("  HOLD" if held else "")
                             + ("  (goal = first frame)" if k == 1 else ""))
        for i, t_ in enumerate(lines):
            dr.rectangle([4, 4 + 16 * i, 8 + 7 * len(t_), 19 + 16 * i], fill=(0, 0, 0))
            dr.text((6, 5 + 16 * i), t_, fill=(255, 255, 255))
        return np.asarray(im)

    def tcp_tok(q):                     # TCP of the shared FK, projected with the (calibrated) camera
        if XYZ:                         # ... or, with two cameras, the TCP position itself
            return tcp_tokens(arm.enc_u if UR else fk, q, None, None, None, cfg["n_freq"], coord="xyz")[0]
        Kc = torch.tensor(G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam]), dtype=torch.float32, device=DEV)
        cp = torch.tensor(d.cam_xpos[sc.cam], dtype=torch.float32, device=DEV)
        cx = torch.tensor(d.cam_xmat[sc.cam], dtype=torch.float32, device=DEV)
        return tcp_tokens(fk, q, cp[None], cx[None], Kc[None], cfg["n_freq"])[0]

    T = int(cfg.get("t_history", T_HISTORY))
    kp_only = bool(cfg.get("oracle_kp", False)) and bool(args.encoder_kp)   # oracle core, encoder keypoints
    oracle = bool(cfg.get("oracle_kp", False)) and not kp_only
    if kp_only:
        print(f"[vision] oracle-trained core fed with ENCODER keypoints ({args.encoder_kp}), vis = 1, no z")
    if oracle:
        print("[vision] ORACLE keypoints (exact projections, no image, no encoder) -- diagnostic upper bound")

    def see_xyz():
        mB, dB = scB.m, scB.d
        dB.qpos[:] = d.qpos; mujoco.mj_forward(mB, dB)
        imgs, cams = [], []
        for s_ in (sc, scB):
            s_.r_rgb.update_scene(s_.d, camera=s_.cam, scene_option=s_.opt)
            r_ = s_.r_rgb.render()
            if args.jpeg:
                import io
                from PIL import Image
                buf = io.BytesIO(); Image.fromarray(r_).save(buf, format="JPEG", quality=args.jpeg)
                r_ = np.array(Image.open(buf).convert("RGB"))
            imgs.append(r_)
            cams.append((G.intrinsics(g.width, g.height, s_.m.cam_fovy[s_.cam]), s_.d.cam_xpos[s_.cam].copy(),
                         s_.d.cam_xmat[s_.cam].copy()))
        x = torch.from_numpy(np.stack(imgs)).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        if tuple(x.shape[-2:]) != tuple(in_hw):
            x = F.interpolate(x, size=tuple(in_hw), mode="bilinear", align_corners=False)
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            o = enc(x)
        uv = (o["uv"].float() / torch.tensor([in_hw[1], in_hw[0]], device=DEV)).cpu().numpy() * [g.width, g.height]
        conf = heatmap_conf(o["prob"]).cpu().numpy()                          # (cam, obj)
        toks, vis = [], []
        for k, key in enumerate(("cube", "tape")):
            plane = (OBJECT_HALF_SIZE if not gstate[0] else None) if k == 0 else 0.0
            pr = xyz_now[key] if (k == 0 and gstate[0] and xyz_now[key] is not None) else None   # cube in the hand
            q_, info = triangulate(uv[:, k], conf[:, k], cams, plane_z=plane, min_w=0.3, prior=pr)
            if q_ is not None:
                xyz_now[key] = q_; v = len(info["views"]) / 2.0
            else:                                                              # not observable: hold the last one
                v = 0.0
                if xyz_now[key] is None:
                    xyz_now[key] = np.array([0.475, 0.025, OBJECT_HALF_SIZE])
            toks.append(xyz_tokens(torch.tensor(xyz_now[key], dtype=torch.float32, device=DEV),
                                   torch.tensor(v, device=DEV), cfg["n_freq"]))
            vis.append(v)
        return torch.stack(toks), torch.tensor(vis, device=DEV), imgs[0]

    def cube_err():
        """error of the cube position the core gets: px (uv cores) or 3D mm (xyz cores)"""
        if XYZ:
            return float(np.linalg.norm(xyz_now["cube"] - d.xpos[obj]) * 1000)
        Kc_ = G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam])
        uvt = G.project(d.xpos[obj], d.cam_xpos[sc.cam], d.cam_xmat[sc.cam], Kc_)[0]
        return float(np.linalg.norm((ch_ref[0][-1][:2].cpu().numpy() + 1) / 2 * [g.width, g.height] - uvt))
    ch_ref = [None]

    def see():
        if XYZ:
            return see_xyz()
        if oracle:
            Kc = G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam])
            uvs = [G.project(p, d.cam_xpos[sc.cam], d.cam_xmat[sc.cam], Kc)[0] / [g.width, g.height]
                   for p in (d.xpos[obj], sc.tape.pos(d))]
            uv = torch.tensor(np.stack(uvs), dtype=torch.float32, device=DEV)
            vis = torch.ones(2, device=DEV)
            rgb = None
            if ep_video[0]:
                sc.r_rgb.update_scene(d, camera=sc.cam, scene_option=sc.opt); rgb = sc.r_rgb.render()
            return object_tokens(uv, vis, torch.zeros(2, 0, device=DEV), cfg["n_freq"], False).to(DEV), vis, rgb
        sc.r_rgb.update_scene(d, camera=sc.cam, scene_option=sc.opt)
        rgb = sc.r_rgb.render()
        if args.jpeg:
            import io
            from PIL import Image
            buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="JPEG", quality=args.jpeg)
            rgb = np.array(Image.open(buf).convert("RGB"))
        x = torch.from_numpy(rgb.copy()).to(DEV).permute(2, 0, 1)[None].float() / 255.0
        if tuple(x.shape[-2:]) != in_hw:
            x = F.interpolate(x, size=in_hw, mode="bilinear", align_corners=False)
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            o = enc(x)
        uv = o["uv"].float()[0] / torch.tensor([in_hw[1], in_hw[0]], device=DEV)
        vis = (heatmap_conf(o["prob"])[0] if cfg.get("vis_source") == "conf"
               else torch.sigmoid(o["vis_logit"].float())[0])
        if args.overlay and ep_video[0]:
            dbg["prob"] = o["prob"].float()[0].cpu().numpy(); dbg["uv_raw"] = uv.cpu().numpy().copy()
            dbg["conf"] = heatmap_conf(o["prob"])[0].cpu().numpy()
        z = o["z"].float()[0] if "z" in o else torch.zeros(2, 0, device=DEV)
        if kp_only:
            v1 = torch.ones(2, device=DEV)
            return object_tokens(uv, v1, torch.zeros(2, 0, device=DEV), cfg["n_freq"], False).to(DEV), vis, rgb
        if kpf is not None:
            uv = uv.clone()
            uv[0] = torch.tensor(kpf(uv[0].cpu().numpy(), float(vis[0]), gstate[0]), device=DEV)
        elif hold > 0:                                             # cube covered: keep its last seen keypoint
            if float(vis[0]) >= hold or last_uv[0] is None:
                last_uv[0] = uv[0].clone()
            else:
                uv = uv.clone(); uv[0] = last_uv[0]
        tok = object_tokens(uv, vis, z[:, :cfg["z_dim"]], cfg["n_freq"], cfg["use_z"]).to(DEV)   # (2, tok)
        if args.overlay and ep_video[0]:
            dbg["uv_used"] = uv.cpu().numpy().copy()
        return tok, vis, rgb

    robot_g = set(int(x) for x in sc.robot_geoms)
    cube_g = set(int(x) for x in np.flatnonzero(m.geom_bodyid == obj))
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g.table_geom)
    f6 = np.zeros(6)

    def cube_yaw_now():
        """cube yaw [deg] about its most vertical axis (nan if tipped)"""
        R_ = np.zeros(9); mujoco.mju_quat2Mat(R_, d.xquat[obj]); R_ = R_.reshape(3, 3)
        u_ = int(np.argmax(np.abs(R_[2, :]))); h_ = R_[:, (u_ + 1) % 3]
        return float(np.degrees(np.arctan2(h_[1], h_[0]))) if abs(R_[2, u_]) > 0.95 else np.nan

    def tcp_yaw():
        """gripper yaw [deg]: direction of the TCP x axis in the table plane"""
        X_ = d.site_xmat[sc.ee].reshape(3, 3)[:, 0]
        return float(np.degrees(np.arctan2(X_[1], X_[0])))

    m90 = lambda a: abs((a + 45.0) % 90.0 - 45.0) if np.isfinite(a) else np.nan   # cube is 90 deg symmetric

    def bump_force():
        """largest normal force robot -> cube (gripper open) and robot -> table, this step [N]"""
        fc = ft = 0.0
        for i in range(d.ncon):
            c = d.contact[i]; a_, b_ = int(c.geom1), int(c.geom2)
            if a_ not in robot_g and b_ not in robot_g:
                continue
            o = b_ if a_ in robot_g else a_
            if o in cube_g or o == tid:
                mujoco.mj_contactForce(m, d, i, f6)
                if o == tid:
                    ft = max(ft, abs(f6[0]))
                else:
                    fc = max(fc, abs(f6[0]))
        return fc, ft

    viewer = None
    if args.viewer:
        import time as _time
        from mujoco import viewer as mj_viewer        # (a plain 'import mujoco.viewer' here would make mujoco local)
        viewer = mj_viewer.launch_passive(m, d)
    res = []
    n_skip = 0                                    # new arm: FR3 starts without a twin pose (failures)
    for ep in range(args.episodes):
        if args.only is not None and ep not in args.only:
            if not ds_eps:
                sample_targets(rng_t)            # keep the target sequence (start poses are exact only with --from_dataset)
            continue
        q_from = None
        if ds_eps:
            Ed = ds_eps[ep % len(ds_eps)]
            cube_np, place, q_from = Ed["cube"], Ed["place"], Ed["q0"]
        else:
            cube_np, place = sample_targets(rng_t)
        ry = np.random.default_rng([args.seed, ep, 77])                        # task: goal / start yaw
        pyaw = float(np.deg2rad(ry.uniform(-args.place_yaw_deg, args.place_yaw_deg)))
        cyaw = float(np.deg2rad(ry.uniform(-args.start_yaw_deg, args.start_yaw_deg)))
        cq = [np.cos(cyaw / 2), 0.0, 0.0, np.sin(cyaw / 2)]
        pyaw_deg = float(np.degrees(pyaw))
        mujoco.mj_resetData(m, d)
        d.qpos[:] = sc.qpos_init
        if args.clean:
            g.no_dr = True
        sc.randomize_appearance(np.random.default_rng([args.seed, ep]), np.random.default_rng([args.seed, ep, 1000]))
        if args.test_scene:
            m.geom_rgba[sc.robot_geoms] = sc.robot_rgba0
            m.cam_pos[sc.cam] = sc.cam_pos0; m.cam_quat[sc.cam] = sc.cam_quat0
        d.qpos[o_adr:o_adr + 3] = cube_np; d.qpos[o_adr + 3:o_adr + 7] = cq
        if UR:
            q0 = arm.start(cube_np, sample_random_init, HOME_Q, INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES)
            if q0 is None:                     # the FR3 start has no elbow-up, table-clear twin pose: a failure
                print(f"  ep{ep:2d}: fail     no {args.robot} start pose for this FR3 start (counted as failure)", flush=True)
                n_skip += 1
                continue
        else:
            q0 = (q_from.copy() if q_from is not None else
                  np.asarray(sample_random_init(m, d, HOME_Q, jnt_ids, INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES)))
        mujoco.mj_resetData(m, d)
        d.qpos[qadr] = q0; d.qpos[o_adr:o_adr + 3] = cube_np; d.qpos[o_adr + 3:o_adr + 7] = cq
        sc.tape.set_pose(m, d, place[0], place[1], pyaw)
        if XYZ:                                   # same world for camera B; its own mount offset
            scB.d.qpos[:] = scB.qpos_init
            scB.randomize_appearance(np.random.default_rng([args.seed, ep]), np.random.default_rng([args.seed, ep, 1000]))
            if args.test_scene:
                scB.m.geom_rgba[scB.robot_geoms] = scB.robot_rgba0
                scB.m.cam_pos[scB.cam] = scB.cam_pos0; scB.m.cam_quat[scB.cam] = scB.cam_quat0
            else:
                rj = np.random.default_rng([args.seed, ep, 3001])
                scB.m.cam_pos[scB.cam] = scB.cam_pos0 + rj.normal(0, gB.cam_pos_std, 3)
                scB.m.cam_quat[scB.cam] = G.quat_mul(scB.cam_quat0, G.small_rot_quat(rj, np.deg2rad(gB.cam_rot_std)))
            scB.m.geom_rgba[scB.robot_geoms] = m.geom_rgba[sc.robot_geoms]
            scB.tape.set_pose(scB.m, scB.d, place[0], place[1], pyaw)
            xyz_now["cube"] = xyz_now["tape"] = None
        if UR:
            arm.set_grip(False)
        else:
            d.ctrl[arm_act] = q0
            if grip_act is not None:
                d.ctrl[grip_act] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)

        qt = torch.tensor(np.asarray(q0, np.float32)[None], device=DEV)
        z0 = zarm(qt)
        ep_video[0] = ep < args.video
        zh = [z0 + torch.randn_like(z0) * PADDING_NOISE_STD for _ in range(T - 1)] + [z0.clone()]
        f0 = finger_now()
        fh = [f0] * T
        last_uv[0] = None
        gstate[0] = False
        if kpf is not None:
            kpf.reset()
        zo, vis0, rgb = see()
        tape_goal = zo[1].clone()                                  # goal: tape token of the first frame
        tape_uv0 = (tape_goal[:2].cpu().numpy() + 1) / 2              # goal keypoint the core keeps (normalized)
        # how wrong is the GOAL the core gets? tape keypoint (first frame) vs the projected tape centre
        Kc0 = G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam])
        uv_tape = G.project(sc.tape.pos(d), d.cam_xpos[sc.cam], d.cam_xmat[sc.cam], Kc0)[0]
        tape_err0 = float(np.linalg.norm((tape_goal[:2].cpu().numpy() + 1) / 2 * [g.width, g.height] - uv_tape))
        if XYZ:
            tape_err0 = float(np.linalg.norm(xyz_now["tape"][:2] - place[:2]) * 1000)
        ch = [zo[0].clone() for _ in range(T)]             # cube tokens (padding repeats frame 0)
        ch_ref[0] = ch
        th = [tcp_tok(qt) for _ in range(T)] if use_tcp else None
        runner.reset()
        shaper = CommandShaper(arm.vstep if UR else VSTEP.cpu().numpy(), args.accel_steps, args.smooth)
        grip_closed, lifted, closes, frames = False, False, 0, []
        ev = {"t_close": None, "t_lift": None, "t_open": None, "close_err": None, "close_xy": None, "close_dz": None, "release_xy": None,
              "dropped": False, "set_down_off": False, "tape_kp": tape_err0, "cube_vis_min": float(vis0[0]), "bump_cube": 0.0, "bump_table": 0.0,
              "collision": False, "kp_err": [], "kp_close": None,
              "yaw0": m90(cube_yaw_now() - pyaw_deg), "grip_rel_close": None, "yaw_close": None, "yaw_lift": None, "yaw_open": None}
        for step in range(args.steps):
            qt = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
            f_hist = torch.tensor([fh], dtype=torch.float32, device=DEV)
            hist = torch.stack(zh)[None]; cube_hist = torch.stack(ch)[None]

            def predict():
                o, l = core(hist, f_hist, cube_hist, tape_goal[None], torch.stack(th)[None] if use_tcp else None)
                if core.last_anc is not None:
                    anc_store[0] = core.last_anc[0].detach()
                return o[0], torch.sigmoid(l[0])
            out, p_c = runner(step, predict)
            if UR:                                                 # the virtual FR3 reads the latent
                f_goal = arm.read(out, qt)
            else:
                q_goal = inverse_decode(out[None], qt, fk, emb, lo=lim[:, 0], hi=lim[:, 1])
                f_goal = fk.forward(q_goal)                        # TCP intent + own IK (decoder "tcp")
            f_now = tcp_fk(qt)
            tcp_now = f_now[0, :3].cpu().numpy()
            pos_t = f_goal[:, :3]
            if blend.on and XYZ:                                   # final approach: absolute target from the anchor head
                pos_t = blend(f_goal[0, :3], anc_store[0], tcp_now, xyz_now["cube"], xyz_now["tape"], grip_closed)[None]
            rot_t = f_goal[:, 3:9]
            if blend.on and XYZ:                                   # carrying: hold the wrist orientation of the grasp
                rot_t = blend.rot(f_goal[0, 3:9], f_now[0, 3:9], grip_closed)[None]
            if UR:
                q_goal = arm.ik(qt, pos_t, rot_t)
            else:
                q_goal = torch.max(torch.min(tcp_ik(fk, qt, pos_t, rot_t), lim[:, 1]), lim[:, 0])
            new_closed = bool(p_c.item() > 0.5)
            if (ep < args.trace or (args.only is not None and args.trace > 0)) and step % args.trace_every == 0:
                tgt = pos_t[0].cpu().numpy(); cb = d.xpos[obj]
                to_c = cb[:2] - tcp_now[:2]; mv = tgt - tcp_now
                Kc = G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam])
                uv_true = G.project(cb, d.cam_xpos[sc.cam], d.cam_xmat[sc.cam], Kc)[0]
                kp_err = cube_err()
                cosv = float(mv[:2] @ to_c / (np.linalg.norm(mv[:2]) * np.linalg.norm(to_c) + 1e-9))
                print(f"    t{step:4d}  TCP-cube xy {1000 * np.linalg.norm(to_c):6.1f}  TCP z {1000 * tcp_now[2]:6.1f}  "
                      f"cube z {1000 * cb[2]:5.1f} | asked move {1000 * np.linalg.norm(mv):5.1f} mm (xy cos to cube "
                      f"{cosv:+.2f}, dz {1000 * mv[2]:+5.1f}) | p_close {p_c.item():.2f} | cube keypoint err {kp_err:5.1f} px"
                      f" | cube-place xy {1000 * np.linalg.norm(cb[:2] - place[:2]):6.1f} (cos to place "
                      f"{float(mv[:2] @ (place[:2] - cb[:2]) / (np.linalg.norm(mv[:2]) * np.linalg.norm(place[:2] - cb[:2]) + 1e-9)):+.2f})"
                      f" | anchor {blend.last_alpha:.2f}",
                      flush=True)
            if new_closed and not grip_closed and ev["t_close"] is None:
                ev["t_close"] = step; ev["close_err"] = np.linalg.norm(tcp_now - d.xpos[obj]) * 1000
                ev["close_xy"] = np.linalg.norm(tcp_now[:2] - d.xpos[obj][:2]) * 1000
                ev["close_dz"] = (tcp_now[2] - d.xpos[obj][2]) * 1000             # + = TCP above the cube centre
                # cube keypoint the core saw in the last 10 steps before closing (worst frame): occlusion check
                ev["kp_close"] = float(np.max(ev["kp_err"][-10:])) if ev["kp_err"] else None
                ev["grip_rel_close"] = m90(tcp_yaw() - cube_yaw_now())    # gripper vs cube faces when closing
                ev["yaw_close"] = m90(cube_yaw_now() - pyaw_deg)
            if (not new_closed) and grip_closed and ev["t_open"] is None and ev["t_close"] is not None:
                ev["t_open"] = step; ev["release_xy"] = np.linalg.norm(tcp_now[:2] - place[:2]) * 1000
                ev["yaw_open"] = m90(cube_yaw_now() - pyaw_deg)
            closes += int(new_closed and not grip_closed)
            grip_closed = new_closed
            gstate[0] = grip_closed
            q_c = shaper(qt[0].cpu().numpy(), q_goal[0].cpu().numpy())
            if UR:
                arm.set_grip(grip_closed)
            elif grip_act is not None:
                d.ctrl[grip_act] = GRIPPER_CLOSED if grip_closed else GRIPPER_OPEN
            trk.step(d, q_c, SUB)
            if viewer is not None:
                if not viewer.is_running():
                    raise SystemExit("[viewer] window closed")
                viewer.sync()
                if args.realtime > 0:
                    _time.sleep(1.0 / (CONTROL_HZ * args.realtime))
            if ev["t_lift"] is None and not grip_closed:          # cube on the table: how far is the cube keypoint off?
                ev["kp_err"].append(cube_err())
            fc, ft = bump_force()
            if not grip_closed and ev["t_lift"] is None:           # approach / re-grasp: any push on the cube is a bump
                ev["bump_cube"] = max(ev["bump_cube"], fc)
            ev["bump_table"] = max(ev["bump_table"], ft)
            if args.reflex > 0 and ((not grip_closed and ev["t_lift"] is None and fc > args.reflex) or ft > args.reflex):
                ev["collision"] = True
                break
            cz = d.xpos[obj][2]
            if cz > OBJECT_HALF_SIZE + 0.05 and ev["t_lift"] is None:
                ev["t_lift"] = step; ev["yaw_lift"] = m90(cube_yaw_now() - pyaw_deg)
            if ev["t_lift"] is not None and ev["t_open"] is None:
                # dropped = the cube LEFT the gripper while it was still closed (slipped / fell out)
                if np.linalg.norm(d.xpos[obj] - tcp_now) > 0.04:      # tcp_now: FK TCP at this step
                    ev["dropped"] = True
                # set down off target = still held, but lowered onto the table > 30 mm from the place point
                elif cz < OBJECT_HALF_SIZE + 0.01 and np.linalg.norm(d.xpos[obj][:2] - place[:2]) > 0.03:
                    ev["set_down_off"] = True
            lifted |= cz > OBJECT_HALF_SIZE + 0.05
            qn = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
            zh.pop(0); zh.append(zarm(qn))
            if use_tcp:
                th.pop(0); th.append(tcp_tok(qn))
            fh.pop(0); fh.append(finger_now())
            zo, vis, rgb = see()
            ch.pop(0); ch.append(zo[0].clone())
            ev["cube_vis_min"] = min(ev["cube_vis_min"], float(vis[0]))
            if ep < args.video:
                frames.append(overlay(rgb, step, grip_closed, tape_uv0) if args.overlay else rgb.copy())
            if ev["t_open"] is not None and step >= ev["t_open"] + 45:
                break
        trk.hold(d, int(0.5 * SIM_HZ))
        c = d.xpos[obj].copy()
        err = np.linalg.norm(c[:2] - place[:2]) * 1000
        rest_ok = c[2] < OBJECT_HALF_SIZE + 0.01 and not grip_closed and not ev["collision"]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, d.xquat[obj]); Rm = Rm.reshape(3, 3)
        # the cube is symmetric: ANY face may end up on top (it often rolls 90 deg in an off-centre grasp).
        # Use the cube axis closest to vertical; tipped only if no face is flat on the table.
        up = int(np.argmax(np.abs(Rm[2, :])))
        hx = Rm[:, (up + 1) % 3]                                                                    # a horizontal cube axis
        cube_yaw = np.degrees(np.arctan2(hx[1], hx[0])) if abs(Rm[2, up]) > 0.95 else np.nan        # nan = tipped
        yerr = abs((cube_yaw - pyaw_deg + 45.0) % 90.0 - 45.0) if np.isfinite(cube_yaw) else 90.0     # vs the tape's yaw
        ev["yaw_err"] = float(yerr)
        ev["held_end"] = bool(grip_closed)              # never released: still in the hand at the end
        ev["rolled"] = bool(Rm[2, 2] < 0.95) and not ev["held_end"]   # released, but not on its original bottom face
        if ev["rolled"] and not args.allow_roll:
            rest_ok = False                             # on the real robot a rolled cube ends the task
        ok = err < args.succ_mm and rest_ok and (args.succ_yaw_deg <= 0 or yerr < args.succ_yaw_deg)
        ev["ok20"] = err < 20 and rest_ok
        ev["rest_ok"] = bool(rest_ok)
        res.append((lifted, err, ok, closes, ev))
        f = lambda v, u="": "  -  " if v is None else (f"{v:5.1f}{u}" if isinstance(v, float) else f"{v}{u}")
        print(f"  ep{ep:2d}: {'SUCCESS' if ok else 'fail   '}  grasped={'yes' if lifted else 'no '}  "
              f"cube->place {err:6.1f} mm | TCP-cube at close {f(ev['close_err'], ' mm')} (xy {f(ev['close_xy'])}, "
              f"dz {f(ev['close_dz'])}), closes={closes} | "
              f"dropped={'yes' if ev['dropped'] else 'no'} | release xy err {f(ev['release_xy'], ' mm')} | "
              f"steps close/lift/open {f(ev['t_close'])}/{f(ev['t_lift'])}/{f(ev['t_open'])} | bump cube {ev['bump_cube']:5.1f} N "
              f"table {ev['bump_table']:5.1f} N{'  COLLISION STOP' if ev['collision'] else ''} | yaw err {ev['yaw_err']:4.1f} deg (start {f(ev['yaw0'])} grip-vs-cube {f(ev['grip_rel_close'])} close {f(ev['yaw_close'])} lift {f(ev['yaw_lift'])} open {f(ev['yaw_open'])}) | cube kp err "
              f"t0 {ev['kp_err'][0] if ev['kp_err'] else float('nan'):5.1f} med {np.median(ev['kp_err']) if ev['kp_err'] else float('nan'):5.1f} "
              f"before close {f(ev['kp_close'])} px | tape kp err {ev['tape_kp']:5.1f} px",
              flush=True)
        if frames:
            import imageio
            p = os.path.join(args.video_dir or os.path.join(R, "validation_out"), f"{args.robot}_pp_vision_ep{ep}.mp4")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            imageio.mimsave(p, frames, fps=CONTROL_HZ); print(f"    [video] {p}")
    gr = np.mean([r[0] for r in res]); s = np.mean([r[2] for r in res]); e = np.array([r[1] for r in res])
    ks = int(sum(r[2] for r in res)); lo, hi = wilson(ks, len(res))
    med = lambda k: np.median([r[4][k] for r in res if r[4][k] is not None]) if any(r[4][k] is not None for r in res) \
        else float("nan")
    print(f"\n  behaviour (medians): TCP-cube at close {med('close_err'):.1f} mm (xy {med('close_xy'):.1f}, "
          f"dz {med('close_dz'):+.1f}), release xy err {med('release_xy'):.1f} mm, "
          f"dropped (slipped out) {100 * np.mean([r[4]['dropped'] for r in res]):.0f}%, set down off target "
          f"{100 * np.mean([r[4]['set_down_off'] for r in res]):.0f}%, single close "
          f"{100 * np.mean([r[3] == 1 for r in res]):.0f}%")
    print(f"  still in the hand at the end (never released): {100 * np.mean([r[4]['held_end'] for r in res]):.0f}% | "
          f"released but rolled / flipped / tipped (not on its original bottom face): "
          f"{100 * np.mean([r[4]['rolled'] for r in res]):.0f}% of episodes"
          f"{' (counted as failure)' if not args.allow_roll else ' (allowed)'}")
    gy = [r[4] for r in res if r[0]]
    if gy:
        mm = lambda k: np.nanmedian([e[k] for e in gy if e[k] is not None]) if any(e[k] is not None for e in gy) else float("nan")
        print(f"  yaw chain (grasped episodes, deg off the tape, mod 90): start {mm('yaw0'):.1f} | gripper vs cube faces at close "
              f"{mm('grip_rel_close'):.1f} | cube at close {mm('yaw_close'):.1f} -> lift {mm('yaw_lift'):.1f} -> open {mm('yaw_open'):.1f} "
              f"-> final {np.median([e['yaw_err'] for e in gy]):.1f}")
    km = np.array([np.median(r[4]["kp_err"]) if r[4]["kp_err"] else np.nan for r in res])
    print(f"  vision: cube keypoint median error per episode > 10 px in {100 * np.nanmean(km > 10):.0f}% of episodes, "
          f"> 50 px in {100 * np.nanmean(km > 50):.0f}%; success among episodes with < 10 px: "
          f"{100 * np.mean([r[2] for r, k in zip(res, km) if k < 10]) if np.any(km < 10) else float('nan'):.0f}%")
    # occlusion check: is a bad grasp position explained by a bad cube keypoint just before the close?
    kc = [(r[4]["kp_close"], r[4]["close_xy"], r[0], r[4]["dropped"]) for r in res if r[4]["kp_close"] is not None]
    if kc:
        for name, sel in (("< 5 px", lambda k: k < 5), (">= 5 px", lambda k: k >= 5)):
            g_ = [x for x in kc if sel(x[0])]
            if g_:
                print(f"  cube keypoint in the 10 steps before the close {name}: {len(g_)} episodes, TCP-cube xy at close "
                      f"median {np.median([x[1] for x in g_]):.1f} mm, grasped {100 * np.mean([x[2] for x in g_]):.0f}%, "
                      f"dropped {100 * np.mean([x[3] for x in g_]):.0f}%")
    # WHY did a grasped episode fail?  one cause per episode, checked in this order
    cause = {"slipped out": 0, "opened early (< 30 steps after lift)": 0, "set down / released off target": 0,
             "never released": 0, "success": 0, "other": 0}
    for r in res:
        e_ = r[4]
        if not r[0]:
            continue
        if r[2]:
            cause["success"] += 1
        elif e_["dropped"]:
            cause["slipped out"] += 1
        elif e_["t_open"] is not None and e_["t_lift"] is not None and e_["t_open"] - e_["t_lift"] < 30:
            cause["opened early (< 30 steps after lift)"] += 1
        elif e_["t_open"] is None:
            cause["never released"] += 1
        elif r[1] >= args.succ_mm:
            cause["set down / released off target"] += 1
        else:
            cause["other"] += 1
    ng = sum(cause.values())
    if ng:
        print(f"  grasped episodes ({ng}): " + ", ".join(f"{k} {v}" for k, v in cause.items() if v))
    # placement: is a far release explained by a wrong GOAL (tape keypoint) or by the core?
    tp = [(r[4]["tape_kp"], r[4]["release_xy"]) for r in res if r[4]["release_xy"] is not None]
    if tp:
        print(f"  goal: tape keypoint error (first frame) median {np.median([r[4]['tape_kp'] for r in res]):.1f} px, "
              f"> 5 px in {100 * np.mean([r[4]['tape_kp'] > 5 for r in res]):.0f}% of episodes")
        for name, sel in (("< 3 px", lambda k: k < 3), (">= 3 px", lambda k: k >= 3)):
            g_ = [x for x in tp if sel(x[0])]
            if g_:
                print(f"  tape keypoint {name}: {len(g_)} releases, TCP-place xy at release median {np.median([x[1] for x in g_]):.1f} mm")
    bc = np.array([r[4]["bump_cube"] for r in res]); bt = np.array([r[4]["bump_table"] for r in res])
    print(f"  contacts: open gripper / arm pushed the cube > 5 N in {100 * np.mean(bc > 5):.0f}% of episodes, "
          f"> 20 N in {100 * np.mean(bc > 20):.0f}% (median max {np.median(bc):.1f} N); table > 20 N in "
          f"{100 * np.mean(bt > 20):.0f}%" + (f"; reflex stops ({args.reflex:.0f} N) "
          f"{100 * np.mean([r[4]['collision'] for r in res]):.0f}%" if args.reflex > 0 else ""))
    th = [10, 15, 20, 30]
    print("  success vs placement threshold (cube on the table, released; yaw < "
          f"{args.succ_yaw_deg:.0f} deg / any yaw): " + "  ".join(
              f"{t} mm {100 * np.mean([r[1] < t and r[4]['rest_ok'] and r[4]['yaw_err'] < args.succ_yaw_deg for r in res]):.0f}%"
              f"/{100 * np.mean([r[1] < t and r[4]['rest_ok'] for r in res]):.0f}%" for t in th))
    print(f"[conditions] targets {'dataset/' + args.from_dataset if args.from_dataset else 'random/' + args.target_split}"
          f"{'(' + args.holdout_which + ')' if args.target_split == 'holdout' else ''}, "
          f"appearance {'clean' if args.clean else ('test scene (real arm, nominal camera)' if args.test_scene else 'randomized')}")
    s20 = 100 * np.mean([r[4]["ok20"] for r in res])
    ye = np.array([r[4]["yaw_err"] for r in res if r[1] < 20])
    if len(ye):
        print(f"  placement yaw (episodes with the cube < 20 mm from the place point): median {np.median(ye):.1f} deg, "
              f"p90 {np.percentile(ye, 90):.1f} deg, within {args.succ_yaw_deg:.0f} deg {100 * np.mean(ye < args.succ_yaw_deg):.0f}%")
    print(f"\n==> {args.robot + ' ZERO-SHOT' if UR else 'FR3'} pick-and-place VISION (MuJoCo), n={len(res)}: grasp {100 * gr:.0f}%  success@{args.succ_mm:.0f}mm "
          f"& yaw<{args.succ_yaw_deg:.0f}deg {100 * s:.0f}% ({ks}/{len(res)}, 95% CI {lo:.0f}-{hi:.0f}%)  [success@20mm, no yaw: {s20:.0f}%]  place error median "
          f"{np.median(e):.1f} mm, p90 {np.percentile(e, 90):.1f} mm")
    if UR:
        lo2, hi2 = wilson(ks, len(res) + n_skip)
        print(f"    {args.robot}: {n_skip} FR3 start poses had no elbow-up twin pose; counting them as failures: success "
              f"{100 * ks / max(1, len(res) + n_skip):.0f}% ({ks}/{len(res) + n_skip}, 95% CI {lo2:.0f}-{hi2:.0f}%)"
              f"  <- compare this one with the FR3 run (same targets, same start poses)")


if __name__ == "__main__":
    main()