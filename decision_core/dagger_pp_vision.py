"""
DAgger for the VISION pick-and-place core: expert corrections from the states the POLICY reaches.

Per episode (cube / place / start pose / posture goal of a decision-core TRAIN episode, cycled):
  1. the vision core runs closed loop exactly as in eval_pp_vision.py (d435 render -> encoder -> tokens)
  2. branch at step b = min(t_s, the step where the core first wants to CLOSE the gripper),
     t_s = --close_max for a --p_close fraction of the episodes (branch at the close decision), else
     t_s ~ U[--t_min, --t_max]; the close is NOT applied
     (the close decision is where it fails: closes 30-40 mm off the cube in xy)
  3. the scripted expert (generate_fr3_pickplace_data.plan) plans from the CURRENT arm pose and cube
     pose and executes in physics; kept only if the cube really ends at the place point
  post-grasp mode (--p_post): the core runs WITH its gripper; once it has lifted the cube, branch at
     t_lift + U[0, --post_max] or when it wants to OPEN, whichever first; the expert then carries the cube
     from where it is (lift -> above the place point -> lower -> open -> retreat)
  local fix (default on): if the branch is low and near the cube, the expert does not go back up to the
     10 cm hover; it moves over the cube at a low height (~6 cm) and descends -- a local correction
  targets: --p_random of the episodes use random cube / place / start pose (outside the hold-out box),
     the rest those of the decision-core TRAIN episodes
  4. saved as one episode: rows 0..b-1 = the policy's own frames (history only, NEVER a target),
     rows b.. = expert. dagger_meta.json at the dataset root stores b per episode file; the training
     loader only samples t >= b, so every target is expert but the history is the policy's drift.

Same parquet layout as the generator (+ object_quat), so the usual pipeline runs on it:
  precompute_latents_pp.py -> (precompute_nullspace.py) -> vision/render_episodes.py -> encode_episode_tokens.py
then train_pp_vision.py --extra_dataset <name> --init_vision <ckpt>.

  P="--pick_gen $HOME/cross_embody/fr3_pick_gen/scripts --scene_module build_scene --cam_name d435"
  A="--ee_site tcp_site --cube object --table_geom table_top --width 768 --height 432"
  MUJOCO_GL=egl python decision_core/dagger_pp_vision.py --ckpt checkpoints/decision_core_vision_kpz_K10_bi_l10_h4_s42_v2.pt \
      --episodes 400 --out_name expert_pp_v3_dag1 $P $A
  --gen_args "..."   plan-affecting flags of the generator run that made the base dataset
                     (--speed_frac / --min_seg / --cart_speed / --hold_close / --swivel_tol / --yaw_rule ...)
"""
import argparse, glob, json, os, shlex, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mujoco

R = os.path.expanduser("~/cross_embody")
for p in (R, os.path.join(R, "fr3_pick_gen", "scripts"), os.path.join(R, "vision")):
    sys.path.insert(0, p)
from decision_core.anchor_blend import AnchorBlend, add_args as add_anchor_args                          # noqa: E402
from decision_core.state_expert import StateExpert, add_args as add_se_args                              # noqa: E402

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--episodes", type=int, default=400, help="episodes to KEEP")
ap.add_argument("--out_name", default="expert_pp_v3_dag1", help="new dataset under fr3_pick_gen/datasets/")
ap.add_argument("--start_index", type=int, default=0, help="first episode index (to append to an existing run)")
ap.add_argument("--t_min", type=int, default=40)
ap.add_argument("--t_max", type=int, default=160)
ap.add_argument("--p_close", type=float, default=0.7,
                help="fraction of episodes that branch ONLY at the core's close decision (up to --close_max steps); "
                     "the rest branch at t_s ~ U[t_min, t_max] (or earlier if the core closes)")
ap.add_argument("--close_max", type=int, default=300)
ap.add_argument("--p_post", type=float, default=0.4, help="fraction of rollouts in post-grasp mode")
ap.add_argument("--post_max", type=int, default=120, help="post-grasp branch at most this many steps after the lift")
ap.add_argument("--regrasp_wait", type=int, default=40,
                help="post-grasp mode: gripper closed this long without lifting the cube = missed grasp -> branch, "
                     "and the expert opens and grasps again (a missed-grasp recovery instead of a wasted rollout)")
ap.add_argument("--p_random", type=float, default=0.5, help="fraction of rollouts with random targets / start pose")
ap.add_argument("--beta", type=float, default=0.0,
                help="state: fraction of episodes driven by the state expert from start to end (1 = generate expert "
                     "demonstrations with the state expert; the core is then not run at all)")
ap.add_argument("--rescue_stall", type=int, default=60,
                help="state: the core made no 5 mm progress for this many steps -> the expert drives --rescue_steps")
ap.add_argument("--rescue_steps", type=int, default=40)
ap.add_argument("--trigger", choices=["branch", "dev", "state"], default="branch",
                help="branch: one hand-over at the core's close decision / a timer (and --handback). dev: the core "
                     "drives the WHOLE task and the expert steps in as soon as it deviates -- no progress towards the "
                     "current goal (the cube while it lies on the table, the place point while carrying) for "
                     "--dev_stall steps, moving away from it by more than --dev_div, closing the gripper more than "
                     "--dev_tol from the cube, opening it more than --dev_tol from the place point, or closing without "
                     "lifting for --regrasp_wait steps. The expert corrects (grasp + lift, or carry + place) and, "
                     "after a grasp correction, hands control back; up to --max_iv interventions per episode. "
                     "Ground truth is used only to decide WHEN to intervene and for the expert's plan.")
ap.add_argument("--succ_mm", type=float, default=10.0,
                help="an episode is kept only if the cube ends this close to the place point (same as the eval criterion)")
ap.add_argument("--dev_div", type=float, default=0.03, help="dev: moving away from the goal by more than this [m]")
ap.add_argument("--dev_stall", type=int, default=40, help="dev: steps without getting 5 mm closer to the goal")
ap.add_argument("--dev_tol", type=float, default=0.012, help="dev: gripper close / open allowed within this xy [m]")
ap.add_argument("--push_mm", type=float, default=5.0,
                help="dev: the OPEN gripper / arm moved the resting cube more than this [mm] -> 'push', the expert "
                     "takes over before the cube tips (0 = off)")
ap.add_argument("--push_deg", type=float, default=8.0,
                help="dev: ... or rotated / tilted it more than this [deg] (0 = off)")
ap.add_argument("--straight_down", type=float, default=0.018,
                help="grasp correction: TCP within this xy [m] of the cube -> straight down to the grasp from the "
                     "current height (the open fingers clear the cube); farther -> over the cube first")
ap.add_argument("--local_fallback", action="store_true",
                help="dev: if the local grasp correction is not reachable, fall back to the full plan (via the 10 cm "
                     "hover, i.e. UP labels next to the cube); default: discard the episode")
ap.add_argument("--clear_z", type=float, default=0.005,
                help="grasp correction: the TCP crosses over to the cube at least this far above the cube top [m]")
ap.add_argument("--small_lift", type=float, default=0.02,
                help="regrasp of a cube lying next to the place point: lift only this much [m] before setting it down")
ap.add_argument("--near_place", type=float, default=0.06,
                help="carrying: within this xy [m] of the place point the expert places from the current height "
                     "(no lift first)")
ap.add_argument("--close_tol", type=float, default=0.0,
                help="dev: the core may close only within this xy of the cube centre [m] (0 = --dev_tol). An off-centre "
                     "grasp lets the cube turn in the hand and roll over at the lift / release")
add_anchor_args(ap)
add_se_args(ap)
ap.add_argument("--max_iv", type=int, default=3, help="dev: interventions per episode")
ap.add_argument("--max_steps", type=int, default=900, help="dev: episode length")
ap.add_argument("--handback", type=float, default=0.0,
                help="multi-intervention DAgger: fraction of the PRE-grasp branches where the expert only grasps and "
                     "lifts the cube, then hands control BACK to the core; the core carries it until it wants to open "
                     "or for U[0, --post_max] steps, then the expert takes over again and places it. Every such "
                     "episode gives grasp AND carry / place corrections, also in the first rounds when the core cannot "
                     "grasp by itself yet. Rows are marked expert_frame = 1 (target) / 0 (history only).")
ap.add_argument("--no_offset_fix", action="store_true",
                help="carrying: put the TCP (not the cube) on the place point, i.e. do not compensate an off-centre grasp")
ap.add_argument("--no_local_fix", action="store_true", help="always re-plan through the 10 cm hover (round-1 behaviour)")
ap.add_argument("--max_tries", type=int, default=0, help="stop after this many rollouts (0 = 3x --episodes)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--jpeg", type=int, default=92)
ap.add_argument("--clean", action="store_true")
ap.add_argument("--chunk_exec", type=int, default=1)
ap.add_argument("--ens_m", type=float, default=0.1)
ap.add_argument("--smooth", type=float, default=0.5)
ap.add_argument("--accel_steps", type=int, default=8)
ap.add_argument("--gen_args", default="", help="flags for generate_fr3_pickplace_data (plan settings)")
ap.add_argument("--gpu_mem_frac", type=float, default=0.3)
args, rest = ap.parse_known_args()

_argv = sys.argv
sys.argv = ["generate_fr3_pickplace_data.py"] + shlex.split(args.gen_args)
import generate_fr3_pickplace_data as GEN                    # noqa: E402  (parses its own flags at import)
sys.argv = _argv
from build_scene import OBJECT_HALF_SIZE                      # noqa: E402
from generate_fr3_expert_data import (resolve_joint_and_actuator_ids, CONTROL_HZ, SIM_HZ,  # noqa: E402
                                      GRIPPER_OPEN, GRIPPER_CLOSED, N_ARM_JOINTS, sample_random_init,
                                      HOME_Q, INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES)
from decision_core_dataset import T_HISTORY, PADDING_NOISE_STD, _episode_split            # noqa: E402
from decision_core.model import load_frozen_stage1, inverse_decode, tcp_ik               # noqa: E402
from decision_core.model_chunk import ChunkRunner                                          # noqa: E402
from decision_core.vision_core import load_vision_core, object_tokens, tcp_tokens, xyz_tokens          # noqa: E402
from joint_tracker import JointTracker, FR3_TAU, CommandShaper                             # noqa: E402
from object_latent import load_any_encoder                                                 # noqa: E402
from kp_conf import heatmap_conf                                                           # noqa: E402
from decision_core.vision_dataset import KeypointFilter                                    # noqa: E402
import gen_vision_data as G                                                                # noqa: E402

g = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"])
XYZ = torch.load(args.ckpt if os.path.isabs(args.ckpt) else os.path.join(R, args.ckpt), map_location="cpu",
                 weights_only=False)["config"].get("coord", "uv") == "xyz"
gB = None
if XYZ:                     # two fixed cameras, triangulated (same set-up as vision/render_encode_2cam.py)
    from render_encode_2cam import CAM_A, CAM_B, LOOKAT                                    # noqa: E402
    from triangulate import triangulate                                                     # noqa: E402
    _cam = ["--width", "960", "--height", "540", "--fovy", "42", "--cam_name=", "--cam_lookat", *map(str, LOOKAT)]
    g = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"] + _cam + ["--cam_pos", *map(str, CAM_A)])
    gB = G.parse_args(rest + ["--n_occ", "0", "--p_distractor", "0"] + _cam + ["--cam_pos", *map(str, CAM_B)])
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda" and args.gpu_mem_frac < 1:
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
VSTEP = torch.tensor([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], device=DEV) / CONTROL_HZ
SUB = max(1, round(SIM_HZ / CONTROL_HZ))


def yaw_of(quat):
    """cube yaw from its quaternion (w, x, y, z); None if the cube is tipped over or rolled onto another face
    (on the real robot that ends the episode, so it is not a state to recover from)"""
    Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, quat); Rm = Rm.reshape(3, 3)
    if Rm[2, 2] < 0.95:
        return None
    return float(np.arctan2(Rm[1, 0], Rm[0, 0]))


def plan_wps(P, q_start, wps, Rm, grip0):
    """generator-style plan (ik per waypoint, hermite segments, gripper dwell after phases 1 / 5) through an
    explicit waypoint list [(phase, xyz)] with a fixed tool orientation Rm; None if a waypoint is unreachable"""
    A = GEN.args
    qs, ns, q_prev = [], [], q_start.copy()
    for phase, p in wps:
        q_t, err = GEN.ik(P["m"], P["d"], P["site"], P["qadr"], P["vadr"], np.asarray(p, float), Rm, q_prev, P["lo"], P["hi"])
        if err > 0.003:
            return None
        ns.append(max(int(np.ceil(np.max(np.abs(q_t - q_prev) / (GEN.SPEED_FRAC * GEN.VLIM)) * CONTROL_HZ)),
                      int(A.min_seg * CONTROL_HZ)))
        qs.append(q_t); q_prev = q_t
    vel = []
    for i, (phase, _) in enumerate(wps):
        if A.stop_at_waypoints or phase in (1, 5, 7) or i + 1 >= len(wps):
            vel.append(np.zeros(7))
        else:
            vel.append((qs[i + 1] - qs[i]) / (ns[i + 1] / CONTROL_HZ))
    segs, q_prev, v_prev, grip = [], q_start.copy(), np.zeros(7), float(grip0)
    for i, (phase, _) in enumerate(wps):
        q_t, n = qs[i], ns[i]
        traj = GEN.quintic(q_prev, q_t, n) if A.stop_at_waypoints else GEN.hermite(q_prev, q_t, v_prev, vel[i], n)
        segs.append((traj, np.full(n, grip), phase))
        q_prev, v_prev = q_t, vel[i]
        if phase == 1:
            grip = 1.0; k = int(GEN.HOLD_S[2] * CONTROL_HZ); segs.append((np.repeat(q_t[None], k, 0), np.full(k, grip), 2))
        if phase == 5:
            grip = 0.0; k = int(GEN.HOLD_S[6] * CONTROL_HZ); segs.append((np.repeat(q_t[None], k, 0), np.full(k, grip), 6))
    return segs


def place_wps(place, start_xy=None, start_z=None, hold_off=None):
    """hold_off: cube xy - TCP xy while carrying (an off-centre grasp). The TCP goes to place - hold_off so the
    CUBE (not the TCP) lands on the place point -- the expert then shows how to compensate a bad grasp."""
    if hold_off is not None and not args.no_offset_fix:
        place = place.copy(); place[:2] = place[:2] - hold_off
    w = []
    if start_xy is not None and np.linalg.norm(np.asarray(start_xy) - place[:2]) < args.near_place:
        # carrying and already near the place point (the core was lowering it): finish from HERE -- slide over at the
        # current height (never up, >= 1.5 cm above the set-down height), then down and open. Going up first would
        # label a lowering state with 'lift', and the core learns to pump up and down instead of releasing.
        zt = float(np.clip(start_z, GEN.PLACE_Z + 0.015, place[2] + GEN.HOVER))
        return [(4, np.r_[place[:2], zt]), (5, np.r_[place[:2], GEN.PLACE_Z]), (7, place + [0, 0, GEN.RETREAT])]
    if start_xy is not None:                                  # carrying, far: first up (never down) where it is
        w.append((3, [start_xy[0], start_xy[1], max(start_z, OBJECT_HALF_SIZE + GEN.LIFT)]))
    return w + [(4, place + [0, 0, GEN.HOVER]), (5, np.r_[place[:2], GEN.PLACE_Z]), (7, place + [0, 0, GEN.RETREAT])]


def grasp_wps(tcp, cube, place):
    """expert grasp correction from the core's current TCP: never a clearance hop UP when the fingers already clear
    the cube, and -- for a cube lying right next to the place point (the core released it a bit off) -- only a small
    lift, slide and set down, NOT the full 10 cm lift: a big lift there labels 'held just above the place point' with
    'go up', the opposite of what the base expert does in that state"""
    off = np.linalg.norm(tcp[:2] - cube[:2])
    grasp = (1, np.r_[cube[:2], GEN.GRASP_Z])
    if off < args.straight_down:                              # the open fingers already straddle the cube
        w = [grasp]
    else:                                                     # over the cube first, at the current height if it clears
        zc = float(np.clip(tcp[2], 2 * OBJECT_HALF_SIZE + args.clear_z, OBJECT_HALF_SIZE + GEN.HOVER))
        w = [(0, np.r_[cube[:2], zc]), grasp]
    if np.linalg.norm(cube[:2] - place[:2]) < args.near_place:
        z_up = GEN.GRASP_Z + args.small_lift
        return w + [(3, np.r_[cube[:2], z_up])] + place_wps(place, cube[:2], z_up)
    return w + [(3, cube + [0, 0, GEN.LIFT])] + place_wps(place)


@torch.no_grad()
def main():
    rng = np.random.default_rng(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    core, cfg = load_vision_core(args.ckpt, DEV)
    enc, in_hw, _ = load_any_encoder(cfg["encoder_ckpt"], DEV)
    fk, emb, _, _ = load_frozen_stage1(device=DEV)
    K = int(cfg["chunk"]); hold = float(cfg.get("hold_thr", 0.0))
    TH = int(cfg.get("t_history", T_HISTORY))
    jump = float(cfg.get("jump_px", 0.0))
    kpf = KeypointFilter(thr=hold, jump_px=jump, wh=(g.width, g.height)) if jump > 0 else None
    gstate = [False]
    print(f"[dagger] history {TH}, visibility {cfg.get('vis_source', 'vis')}, hold {hold}, jump filter {jump or 'off'}")
    runner = ChunkRunner(K, args.chunk_exec, True, args.ens_m)
    blend = AnchorBlend(core, args)
    anc_store = [None]
    if blend.on:
        print(f"[anchor] the core's final approach uses its anchor head (k={blend.k})")
    zarm = lambda q: emb(fk.forward(q)[:, :18])[0]
    use_tcp = bool(cfg.get("use_tcp", False))

    # base dataset: the TRAIN split only (val stays untouched for evaluation)
    base = os.path.join(R, "fr3_pick_gen", "datasets", cfg["dataset"])
    tr_f, _ = _episode_split(sorted(glob.glob(os.path.join(base, "data", "chunk-000", "episode_*.parquet"))))
    cols = pd.read_parquet(tr_f[0]).columns
    use_psi = "psi_goal" in cols
    eps = []
    for f in tr_f:
        dfe = pd.read_parquet(f, columns=["observation.state", "object_xyz", "place_xyz"] + (["psi_goal"] if use_psi else []))
        eps.append(dict(q0=np.asarray(dfe["observation.state"].iloc[0], np.float64)[:7],
                        cube=np.asarray(dfe["object_xyz"].iloc[0], np.float64),
                        place=np.asarray(dfe["place_xyz"].iloc[0], np.float64),
                        psi=tuple(np.asarray(dfe["psi_goal"].iloc[0], np.float64)) if use_psi else None))
    order = rng.permutation(len(eps))
    print(f"[dagger] core {args.ckpt}\n         base {cfg['dataset']}: {len(eps)} train episodes, posture goal "
          f"{'yes (psi_goal)' if use_psi else 'no'}; generator flags: '{args.gen_args}'")

    # vision scene (rollout + expert execution) and a separate generator model for planning (IK writes its qpos)
    sc = G.Scene(g); m, d = sc.m, sc.d
    scB = G.Scene(gB) if XYZ else None
    xyz_now = {"cube": None, "tape": None}
    jnt_ids, qadr, arm_act, grip_act, g_qadr = resolve_joint_and_actuator_ids(m)
    qadr = np.asarray(qadr); vadr = np.array([m.jnt_dofadr[j] for j in jnt_ids])
    trk = JointTracker(m, qadr, vadr, FR3_TAU, actuator_ids=arm_act)
    lim = torch.tensor(m.jnt_range[jnt_ids], dtype=torch.float32, device=DEV)
    o_adr, obj = sc.cube.qadr, sc.cube.bid
    pm = GEN.build_model(); pd_ = mujoco.MjData(pm)
    pj, pqadr, _, _, _ = resolve_joint_and_actuator_ids(pm)
    pqadr = np.asarray(pqadr); pvadr = np.array([pm.jnt_dofadr[j] for j in pj])
    plo, phi = pm.jnt_range[pj, 0], pm.jnt_range[pj, 1]
    psite = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    P = dict(m=pm, d=pd_, site=psite, qadr=pqadr, vadr=pvadr, lo=plo, hi=phi)
    tcp_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    SE = StateExpert(GEN, P, OBJECT_HALF_SIZE, CONTROL_HZ, args, args.succ_mm)
    st = {"ok": 0, "tipped": 0, "ik": 0, "timeout": 0, "rescues": 0, "frames": 0, "core_frames": 0}
    hb = GEN.args.holdout_box
    in_box = lambda p: hb[0] < p[0] < hb[1] and hb[2] < p[1] < hb[3]

    def random_targets():
        A = GEN.args
        while True:
            c = np.array([rng.uniform(*A.cube_x), rng.uniform(*A.cube_y), OBJECT_HALF_SIZE])
            p = np.array([rng.uniform(*A.place_x), rng.uniform(*A.place_y), 0.0])
            if not in_box(c) and not in_box(p) and np.linalg.norm(p[:2] - c[:2]) >= A.min_sep:
                return c, p

    last_uv = [None]

    def tcp_tok(q):
        if XYZ:
            return tcp_tokens(fk, q, None, None, None, cfg["n_freq"], coord="xyz")[0]
        Kc = torch.tensor(G.intrinsics(g.width, g.height, m.cam_fovy[sc.cam]), dtype=torch.float32, device=DEV)
        cp = torch.tensor(d.cam_xpos[sc.cam], dtype=torch.float32, device=DEV)
        cx = torch.tensor(d.cam_xmat[sc.cam], dtype=torch.float32, device=DEV)
        return tcp_tokens(fk, q, cp[None], cx[None], Kc[None], cfg["n_freq"])[0]

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
        conf = heatmap_conf(o["prob"]).cpu().numpy()
        toks = []
        for k, key in enumerate(("cube", "tape")):
            plane = (OBJECT_HALF_SIZE if not gstate[0] else None) if k == 0 else 0.0
            pr = xyz_now[key] if (k == 0 and gstate[0] and xyz_now[key] is not None) else None   # cube in the hand
            q_, info = triangulate(uv[:, k], conf[:, k], cams, plane_z=plane, min_w=0.3, prior=pr)
            if q_ is not None:
                xyz_now[key] = q_; v = len(info["views"]) / 2.0
            else:
                v = 0.0
                if xyz_now[key] is None:
                    xyz_now[key] = np.array([0.475, 0.025, OBJECT_HALF_SIZE])
            toks.append(xyz_tokens(torch.tensor(xyz_now[key], dtype=torch.float32, device=DEV),
                                   torch.tensor(v, device=DEV), cfg["n_freq"]))
        return torch.stack(toks)

    def see():
        if XYZ:
            return see_xyz()
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
        z = o["z"].float()[0] if "z" in o else torch.zeros(2, 0, device=DEV)
        if kpf is not None:
            uv = uv.clone()
            uv[0] = torch.tensor(kpf(uv[0].cpu().numpy(), float(vis[0]), gstate[0]), device=DEV)
        elif hold > 0:
            if float(vis[0]) >= hold or last_uv[0] is None:
                last_uv[0] = uv[0].clone()
            else:
                uv = uv.clone(); uv[0] = last_uv[0]
        return object_tokens(uv, vis, z[:, :cfg["z_dim"]], cfg["n_freq"], cfg["use_z"]).to(DEV)

    def row(q_cmd, g_closed, phase, goal_xyz, place, expert=0.0):
        g_raw = GRIPPER_CLOSED if g_closed else GRIPPER_OPEN
        finger = float(d.qpos[g_qadr]) if g_qadr is not None else 0.0
        return {"observation.state": np.append(d.qpos[qadr], finger).astype(np.float32),
                "observation.velocity": np.append(d.qvel[vadr], 0.0).astype(np.float32),
                "action": np.append(q_cmd, g_raw).astype(np.float32),
                "gripper_closed": float(g_closed),
                "object_xyz": d.xpos[obj].astype(np.float32).copy(),
                "object_quat": d.xquat[obj].astype(np.float32).copy(),
                "goal_xyz": goal_xyz.astype(np.float32), "place_xyz": place.astype(np.float32),
                "phase": phase, "phase_name": GEN.PHASES[phase], "expert_frame": float(expert)}

    root = os.path.join(R, "fr3_pick_gen", "datasets", args.out_name)
    out = os.path.join(root, "data", "chunk-000"); os.makedirs(out, exist_ok=True)
    meta_p = os.path.join(root, "dagger_meta.json")
    dmeta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    emeta_p = os.path.join(root, "expert_frames.json")          # [start, end) intervals of expert-controlled rows
    emeta = json.load(open(emeta_p)) if os.path.exists(emeta_p) else {}
    kept, tried, t0 = 0, 0, time.time()
    fail = {"tipped": 0, "plan": 0, "grasp": 0, "place": 0, "no_lift": 0, "dropped": 0}
    why = {"close": 0, "t_s": 0, "open": 0, "post_t": 0, "regrasp": 0}
    xy_at_branch = {"close": [], "t_s": [], "regrasp": []}
    n_local, n_random = 0, 0
    max_tries = args.max_tries or 3 * args.episodes
    while kept < args.episodes and tried < max_tries:
        E = eps[order[tried % len(eps)]]; tried += 1
        is_rand = rng.random() < args.p_random
        post = rng.random() < args.p_post
        mujoco.mj_resetData(m, d)
        d.qpos[:] = sc.qpos_init
        sc.randomize_appearance(np.random.default_rng([args.seed, tried]), np.random.default_rng([args.seed, tried, 1000]))
        if is_rand:
            cube0, place = random_targets()
            d.qpos[o_adr:o_adr + 3] = cube0; d.qpos[o_adr + 3:o_adr + 7] = [1, 0, 0, 0]
            q0 = np.asarray(sample_random_init(m, d, HOME_Q, jnt_ids, INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES))[:7]
            psi = None
        else:
            cube0, place, q0, psi = E["cube"].copy(), E["place"], E["q0"], E["psi"]
        mujoco.mj_resetData(m, d)
        d.qpos[qadr] = q0; d.qpos[o_adr:o_adr + 3] = cube0; d.qpos[o_adr + 3:o_adr + 7] = [1, 0, 0, 0]
        sc.tape.set_pose(m, d, place[0], place[1], 0.0)
        if XYZ:                                   # same world for camera B, its own mount offset
            scB.d.qpos[:] = scB.qpos_init
            scB.randomize_appearance(np.random.default_rng([args.seed, tried]),
                                     np.random.default_rng([args.seed, tried, 1000]))
            rj = np.random.default_rng([args.seed, tried, 3001])
            scB.m.cam_pos[scB.cam] = scB.cam_pos0 + rj.normal(0, gB.cam_pos_std, 3)
            scB.m.cam_quat[scB.cam] = G.quat_mul(scB.cam_quat0, G.small_rot_quat(rj, np.deg2rad(gB.cam_rot_std)))
            scB.m.geom_rgba[scB.robot_geoms] = m.geom_rgba[sc.robot_geoms]
            scB.tape.set_pose(scB.m, scB.d, place[0], place[1], 0.0)
            xyz_now["cube"] = xyz_now["tape"] = None
        d.ctrl[arm_act] = q0
        if grip_act is not None:
            d.ctrl[grip_act] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)

        # ---- 1) policy rollout up to the branch point
        qt = torch.tensor(np.asarray(q0, np.float32)[None], device=DEV)
        z0 = zarm(qt)
        zh = [z0 + torch.randn_like(z0) * PADDING_NOISE_STD for _ in range(TH - 1)] + [z0.clone()]
        fh = [float(d.qpos[g_qadr]) if g_qadr is not None else 0.0] * TH
        last_uv[0] = None
        gstate[0] = False
        if kpf is not None:
            kpf.reset()
        zo = see(); tape_goal = zo[1].clone(); ch = [zo[0].clone() for _ in range(TH)]
        th = [tcp_tok(qt) for _ in range(TH)] if use_tcp else None
        runner.reset(); shaper = CommandShaper(VSTEP.cpu().numpy(), args.accel_steps, args.smooth)
        if args.trigger == "state":
            # ======== DAgger with a STATE expert: every visited state gets the expert's chunk as its label ========
            only_expert = args.beta >= 1.0
            drive_expert = only_expert or rng.random() < args.beta
            rows, end_why, step, rescue_left = [], "timeout", 0, 0
            grip_closed = False
            f_prev = float(d.qpos[g_qadr]) if g_qadr is not None else 0.0
            f_open = f_prev                                  # the gripper starts open
            best, best_t = np.inf, 0

            def keep_hist_s():
                qn = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
                zh.pop(0); zh.append(zarm(qn))
                if use_tcp:
                    th.pop(0); th.append(tcp_tok(qn))
                fh.pop(0); fh.append(float(d.qpos[g_qadr]) if g_qadr is not None else 0.0)
                zo_ = see(); ch.pop(0); ch.append(zo_[0].clone())

            while step < args.max_steps:
                tcp = d.site_xpos[tcp_sid].copy(); Rt = d.site_xmat[tcp_sid].reshape(3, 3).copy()
                fin = float(d.qpos[g_qadr]) if g_qadr is not None else 0.0
                f_open = max(f_open, fin)            # widest opening seen (after a reset the fingers start shut)
                lab = SE.label(d.qpos[qadr].copy(), tcp, Rt, grip_closed, fin, f_prev, d.xpos[obj].copy(),
                               d.xquat[obj].copy(), place, K, finger_open=f_open)
                if lab is None:
                    end_why = SE.fail or "ik"; break
                if lab["done"]:
                    end_why = "ok"; break
                held = grip_closed and d.xpos[obj][2] > OBJECT_HALF_SIZE + 0.02
                goal = place[:2] if held else d.xpos[obj][:2]
                dist = float(np.linalg.norm(tcp[:2] - goal) + (0.0 if held else max(0.0, tcp[2] - OBJECT_HALF_SIZE)))
                if dist < best - 0.005:
                    best, best_t = dist, step
                if not drive_expert and rescue_left == 0 and step - best_t > args.rescue_stall:
                    rescue_left = args.rescue_steps; st["rescues"] += 1
                use_exp = drive_expert or rescue_left > 0
                if use_exp:
                    q_c = lab["q"][0].astype(np.float64); want = bool(lab["grip"][0] > 0.5)
                    if rescue_left > 0:
                        rescue_left -= 1
                        if rescue_left == 0:                 # back to the core: fresh chunk state
                            best, best_t = np.inf, step
                            runner.reset(); shaper = CommandShaper(VSTEP.cpu().numpy(), args.accel_steps, args.smooth)
                else:
                    qt = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
                    hist, f_hist, cube_hist = torch.stack(zh)[None], torch.tensor([fh], dtype=torch.float32, device=DEV), torch.stack(ch)[None]

                    def predict_s():
                        o, l = core(hist, f_hist, cube_hist, tape_goal[None], torch.stack(th)[None] if use_tcp else None)
                        if core.last_anc is not None:
                            anc_store[0] = core.last_anc[0].detach()
                        return o[0], torch.sigmoid(l[0])
                    o_, p_c = runner(step, predict_s)
                    want = p_c.item() > 0.5
                    q_goal = inverse_decode(o_[None], qt, fk, emb, lo=lim[:, 0], hi=lim[:, 1])
                    f_goal = fk.forward(q_goal)
                    pos_t = f_goal[:, :3]
                    if blend.on and XYZ:
                        pos_t = blend(f_goal[0, :3], anc_store[0], tcp.copy(), xyz_now["cube"], xyz_now["tape"],
                                      grip_closed)[None]
                    rot_t = f_goal[:, 3:9]
                    if blend.on and XYZ:
                        rot_t = blend.rot(f_goal[0, 3:9], fk.forward(qt)[0, 3:9], grip_closed)[None]
                    q_goal = torch.max(torch.min(tcp_ik(fk, qt, pos_t, rot_t), lim[:, 1]), lim[:, 0])
                    q_c = shaper(qt[0].cpu().numpy(), q_goal[0].cpu().numpy())
                    st["core_frames"] += 1
                r_ = row(q_c, want, int(lab["phase"]), d.xpos[obj].copy(), place, expert=float(use_exp))
                r_["label_q"] = lab["q"].reshape(-1); r_["label_grip"] = lab["grip"]
                rows.append(r_)
                grip_closed = want; gstate[0] = want
                if grip_act is not None:
                    d.ctrl[grip_act] = GRIPPER_CLOSED if want else GRIPPER_OPEN
                f_prev = fin
                trk.step(d, q_c, SUB)
                if not only_expert:
                    keep_hist_s()
                step += 1
            st[end_why] += 1; st["frames"] += len(rows)
            if len(rows) < 2 * K:
                continue
            idx = args.start_index + kept
            df = pd.DataFrame(rows)
            df["frame_index"] = np.arange(len(df)); df["episode_index"] = idx
            df["timestamp"] = df["frame_index"] / CONTROL_HZ; df["index"] = 0; df["task_index"] = 0
            if use_psi:
                df["psi_goal"] = [np.array(psi if psi is not None else (0.0, 0.0), dtype=np.float32)] * len(df)
            name = f"episode_{idx:06d}.parquet"
            df.to_parquet(os.path.join(out, name))
            dmeta[name] = 0                                  # every frame has a label
            emeta[name] = [[0, len(df)]]
            n_random += int(is_rand); kept += 1
            if kept % 20 == 0 or kept == args.episodes:
                json.dump(dmeta, open(meta_p, "w"), indent=0); json.dump(emeta, open(emeta_p, "w"))
                print(f"[{kept}/{args.episodes}] tried {tried} | ended: done {st['ok']}, tipped {st['tipped']}, IK {st['ik']}, "
                      f"timeout {st['timeout']} | rescues {st['rescues']} | frames driven by the core "
                      f"{100 * st['core_frames'] / max(1, st['frames']):.0f}%  ({time.time() - t0:.0f}s)", flush=True)
            continue
        if args.trigger == "dev":
            # ================= deviation-triggered, multi-intervention rollout =================
            rows, ivs, n_iv, ok_ep, fail_why = [], [], 0, False, None
            grip_closed, t_shut, step = False, None, 0

            def keep_hist():
                qn = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
                zh.pop(0); zh.append(zarm(qn))
                if use_tcp:
                    th.pop(0); th.append(tcp_tok(qn))
                fh.pop(0); fh.append(float(d.qpos[g_qadr]) if g_qadr is not None else 0.0)
                zo_ = see(); ch.pop(0); ch.append(zo_[0].clone())

            def run_expert(segs_, stop_after_lift):
                """execute expert segments (expert_frame = 1); stop once lifted if asked; -> lifted"""
                lifted_ = False
                for q_seq, g_seq, phase in segs_:
                    if stop_after_lift and lifted_ and phase >= 4:
                        break
                    for q_t, gg in zip(q_seq, g_seq):
                        rows.append(row(q_t, gg > 0.5, phase, d.xpos[obj].copy(), place, expert=1.0))
                        if grip_act is not None:
                            d.ctrl[grip_act] = GRIPPER_CLOSED if gg > 0.5 else GRIPPER_OPEN
                        gstate[0] = gg > 0.5
                        trk.step(d, q_t, SUB)
                        if phase == 3 and d.xpos[obj][2] > OBJECT_HALF_SIZE + 0.01:
                            lifted_ = True
                        keep_hist()
                return lifted_

            def grasp_plan():
                q_b = d.qpos[qadr].copy(); tcp = d.site_xpos[tcp_sid].copy(); cube = d.xpos[obj].copy()
                yaw = yaw_of(d.xquat[obj].copy())
                if yaw is None or cube[2] > OBJECT_HALF_SIZE + 0.01:
                    return None, "tipped"
                cube[2] = OBJECT_HALF_SIZE
                segs_ = None
                near_pl = np.linalg.norm(cube[:2] - place[:2]) < args.near_place
                if (near_pl or tcp[2] < OBJECT_HALF_SIZE + 0.07 and np.linalg.norm(tcp[:2] - cube[:2]) < 0.08) and psi is None:
                    wps = grasp_wps(tcp, cube, place)
                    for k in sorted(range(4), key=lambda k: abs(np.angle(np.exp(1j * (yaw + k * np.pi / 2 - np.arctan2(
                            d.site_xmat[tcp_sid][3], d.site_xmat[tcp_sid][0])))))):
                        segs_ = plan_wps(P, q_b, wps, GEN.R_down(yaw + k * np.pi / 2), 0.0)
                        if segs_ is not None:
                            break
                    if segs_ is None and not args.local_fallback:
                        return None, "plan"                  # the full plan would hop up to the 10 cm hover
                if segs_ is None:
                    segs_ = GEN.plan(pm, pd_, psite, pqadr, pvadr, plo, phi, q_b, cube, yaw, place, psi)
                return segs_, None if segs_ is not None else "plan"

            def place_plan():
                q_b = d.qpos[qadr].copy(); tcp = d.site_xpos[tcp_sid].copy()
                Rm = d.site_xmat[tcp_sid].reshape(3, 3).copy()
                segs_ = plan_wps(P, q_b, place_wps(place, tcp[:2], tcp[2], d.xpos[obj][:2] - tcp[:2]), Rm, 1.0)
                return segs_, None if segs_ is not None else "plan"

            while step < args.max_steps and fail_why is None and not ok_ep:
                # ---- the core drives until it deviates
                runner.reset(); shaper = CommandShaper(VSTEP.cpu().numpy(), args.accel_steps, args.smooth)
                best, best_t, trig, was_held = np.inf, 0, None, None
                c_ref, q_ref = d.xpos[obj].copy(), d.xquat[obj].copy()    # the cube when the core takes over
                k = 0
                while step < args.max_steps:
                    held = d.xpos[obj][2] > OBJECT_HALF_SIZE + 0.02 and grip_closed
                    if held != was_held:                            # new sub-goal (cube -> place point): restart
                        best, best_t, was_held = np.inf, k, held
                    goal = place[:2] if held else d.xpos[obj][:2]
                    tcp = d.site_xpos[tcp_sid]
                    dist = float(np.linalg.norm(tcp[:2] - goal) + (0.0 if held else max(0.0, tcp[2] - OBJECT_HALF_SIZE)))
                    if dist < best - 0.005:
                        best, best_t = dist, k
                    if dist > best + args.dev_div:
                        trig = "diverge"; break
                    if k - best_t > args.dev_stall and not (grip_closed and not held):
                        # no progress: far from the goal (stall), or at it without closing / opening (hesitate);
                        # not while the gripper is shut on the cube before the lift (the regrasp test handles that)
                        trig = "stall" if dist > args.dev_tol else "hesitate"; break
                    qt = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
                    hist, f_hist, cube_hist = torch.stack(zh)[None], torch.tensor([fh], dtype=torch.float32, device=DEV), torch.stack(ch)[None]

                    def predict3():
                        o, l = core(hist, f_hist, cube_hist, tape_goal[None], torch.stack(th)[None] if use_tcp else None)
                        if core.last_anc is not None:
                            anc_store[0] = core.last_anc[0].detach()
                        return o[0], torch.sigmoid(l[0])
                    o_, p_c = runner(k, predict3)
                    want = p_c.item() > 0.5
                    xy_c = np.linalg.norm(tcp[:2] - d.xpos[obj][:2])
                    if want and not grip_closed and not held and (xy_c > (args.close_tol or args.dev_tol) or tcp[2] - d.xpos[obj][2] > 0.02):
                        trig = "bad_close"; break
                    if not want and grip_closed and held and np.linalg.norm(d.xpos[obj][:2] - place[:2]) > min(args.dev_tol, 0.8 * args.succ_mm / 1000.0):
                        trig = "bad_open"; break
                    if want and grip_closed and not held and t_shut is not None and k - t_shut > args.regrasp_wait:
                        trig = "regrasp"; break
                    q_goal = inverse_decode(o_[None], qt, fk, emb, lo=lim[:, 0], hi=lim[:, 1])
                    f_goal = fk.forward(q_goal)
                    pos_t = f_goal[:, :3]
                    if blend.on and XYZ:                    # the core's final approach: anchor head
                        pos_t = blend(f_goal[0, :3], anc_store[0], tcp.copy(), xyz_now["cube"], xyz_now["tape"],
                                      grip_closed)[None]
                    rot_t = f_goal[:, 3:9]
                    if blend.on and XYZ:
                        rot_t = blend.rot(f_goal[0, 3:9], fk.forward(qt)[0, 3:9], grip_closed)[None]
                    q_goal = torch.max(torch.min(tcp_ik(fk, qt, pos_t, rot_t), lim[:, 1]), lim[:, 0])
                    q_c = shaper(qt[0].cpu().numpy(), q_goal[0].cpu().numpy())
                    if want and not grip_closed:
                        t_shut = k
                    grip_closed = want; gstate[0] = want
                    if grip_act is not None:
                        d.ctrl[grip_act] = GRIPPER_CLOSED if want else GRIPPER_OPEN
                    rows.append(row(q_c, want, 3 if held else (2 if want else 0), d.xpos[obj].copy(), place, expert=0.0))
                    trk.step(d, q_c, SUB)
                    keep_hist(); step += 1; k += 1
                    if not grip_closed and not held and c_ref[2] < OBJECT_HALF_SIZE + 0.01:
                        # the open gripper / arm is pushing the resting cube: hand over BEFORE it tips
                        moved = np.linalg.norm(d.xpos[obj][:2] - c_ref[:2]) * 1000
                        turned = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(d.xquat[obj], q_ref))))))
                        if (args.push_mm > 0 and moved > args.push_mm) or (args.push_deg > 0 and turned > args.push_deg):
                            trig = "push"; break
                    if held and d.xpos[obj][2] < OBJECT_HALF_SIZE + 0.01 and not grip_closed:
                        break                                       # the core put it down by itself
                    if not held and grip_closed is False and d.xpos[obj][2] < OBJECT_HALF_SIZE + 0.01 and \
                            np.linalg.norm(d.xpos[obj][:2] - place[:2]) < args.succ_mm / 1000.0 and any(r["gripper_closed"] for r in rows):
                        break
                c = d.xpos[obj]
                if trig is None:                                    # time out, or the core released the cube
                    if np.linalg.norm(c[:2] - place[:2]) < args.succ_mm / 1000.0 and c[2] < OBJECT_HALF_SIZE + 0.01:
                        ok_ep = True
                    elif step >= args.max_steps:
                        fail_why = "timeout"
                    else:
                        trig = "released_off"
                if ok_ep or fail_why:
                    break
                if n_iv >= args.max_iv:
                    fail_why = "max_iv"; break
                # ---- the expert steps in
                n_iv += 1; ivs.append(trig)
                carrying = d.xpos[obj][2] > OBJECT_HALF_SIZE + 0.02 and grip_closed
                if carrying:
                    segs_, err = place_plan()
                    if segs_ is None:
                        fail_why = err; break
                    run_expert(segs_, False); grip_closed = False
                    trk.hold(d, int(0.5 * SIM_HZ))
                    c = d.xpos[obj]
                    ok_ep = np.linalg.norm(c[:2] - place[:2]) < args.succ_mm / 1000.0 and c[2] < OBJECT_HALF_SIZE + 0.01
                    if not ok_ep:
                        fail_why = "place"
                    break
                segs_, err = grasp_plan()
                if segs_ is None:
                    fail_why = err; break
                last_iv = n_iv >= args.max_iv or np.linalg.norm(d.xpos[obj][:2] - place[:2]) < args.near_place
                lifted_ = run_expert(segs_, stop_after_lift=not last_iv)
                if not lifted_:
                    fail_why = "grasp"; break
                grip_closed = bool(gstate[0]); t_shut = None
                if last_iv:                                         # expert finished the whole task
                    trk.hold(d, int(0.5 * SIM_HZ))
                    c = d.xpos[obj]
                    ok_ep = np.linalg.norm(c[:2] - place[:2]) < args.succ_mm / 1000.0 and c[2] < OBJECT_HALF_SIZE + 0.01
                    if not ok_ep:
                        fail_why = "place"
                    break
            if fail_why or not ok_ep:
                fail[fail_why or "place"] = fail.get(fail_why or "place", 0) + 1; continue
            if not any(r["expert_frame"] > 0.5 for r in rows):     # the core did it alone: nothing to learn here
                fail["no_iv"] = fail.get("no_iv", 0) + 1; continue
            idx = args.start_index + kept
            df = pd.DataFrame(rows)
            df["frame_index"] = np.arange(len(df)); df["episode_index"] = idx
            df["timestamp"] = df["frame_index"] / CONTROL_HZ; df["index"] = 0; df["task_index"] = 0
            if use_psi:
                df["psi_goal"] = [np.array(psi if psi is not None else (0.0, 0.0), dtype=np.float32)] * len(df)
            name = f"episode_{idx:06d}.parquet"
            df.to_parquet(os.path.join(out, name))
            ef = df["expert_frame"].values > 0.5
            dmeta[name] = int(np.argmax(ef))
            edges = np.flatnonzero(np.diff(np.r_[0, ef.astype(int), 0]))
            emeta[name] = [[int(edges[q]), int(edges[q + 1])] for q in range(0, len(edges), 2)]
            for t_ in ivs:
                why[t_] = why.get(t_, 0) + 1
            n_random += int(is_rand); kept += 1
            if kept % 20 == 0 or kept == args.episodes:
                json.dump(dmeta, open(meta_p, "w"), indent=0); json.dump(emeta, open(emeta_p, "w"))
                print(f"[{kept}/{args.episodes}] tried {tried}  kept {kept / tried:.0%}  failures {fail}  "
                      f"interventions {why}  ({sum(why.values()) / max(1, kept):.2f} per episode)  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            continue

        if post:
            t_s = args.close_max + args.post_max
        else:
            t_s = args.close_max if rng.random() < args.p_close else int(rng.integers(args.t_min, args.t_max + 1))
        rows, reason = [], "t_s"
        grip_closed, t_lift, t_post, bad, t_shut = False, None, None, None, None
        for step in range(t_s):
            qt = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
            hist, f_hist, cube_hist = torch.stack(zh)[None], torch.tensor([fh], dtype=torch.float32, device=DEV), torch.stack(ch)[None]

            def predict():
                o, l = core(hist, f_hist, cube_hist, tape_goal[None], torch.stack(th)[None] if use_tcp else None)
                return o[0], torch.sigmoid(l[0])
            o_, p_c = runner(step, predict)
            want_close = p_c.item() > 0.5
            if not post and want_close:                             # the core wants to close: branch HERE
                reason = "close"; break
            if post and t_lift is None and t_shut is not None and step - t_shut >= args.regrasp_wait:
                reason = "regrasp"; break
            if post and t_lift is not None:
                if not want_close:                                  # wants to open while carrying: branch HERE
                    reason = "open"; break
                if step >= t_post:
                    reason = "post_t"; break
            q_goal = inverse_decode(o_[None], qt, fk, emb, lo=lim[:, 0], hi=lim[:, 1])
            f_goal = fk.forward(q_goal)
            q_goal = torch.max(torch.min(tcp_ik(fk, qt, f_goal[:, :3], f_goal[:, 3:9]), lim[:, 1]), lim[:, 0])
            q_c = shaper(qt[0].cpu().numpy(), q_goal[0].cpu().numpy())
            if post:
                if want_close and not grip_closed:
                    t_shut = step
                elif not want_close:
                    t_shut = None
                grip_closed = want_close
                gstate[0] = grip_closed
                if grip_act is not None:
                    d.ctrl[grip_act] = GRIPPER_CLOSED if grip_closed else GRIPPER_OPEN
            rows.append(row(q_c, grip_closed, 3 if t_lift is not None else (2 if grip_closed else 0), cube0, place))
            trk.step(d, q_c, SUB)
            if post:
                cz = d.xpos[obj][2]
                if t_lift is None and cz > OBJECT_HALF_SIZE + 0.05:
                    t_lift = step; t_post = step + int(rng.integers(0, args.post_max + 1))
                if t_lift is not None and cz < OBJECT_HALF_SIZE + 0.02:
                    bad = "dropped"; break
            qn = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
            zh.pop(0); zh.append(zarm(qn))
            if use_tcp:
                th.pop(0); th.append(tcp_tok(qn))
            fh.pop(0); fh.append(float(d.qpos[g_qadr]) if g_qadr is not None else 0.0)
            zo = see(); ch.pop(0); ch.append(zo[0].clone())
        b = len(rows)
        if post and bad is None and t_lift is None and reason == "t_s":
            reason = "regrasp"                                      # never lifted: expert takes over from here
        if bad:
            fail[bad] += 1; continue

        # ---- 2) expert from the reached state
        carry = post and t_lift is not None
        q_b = d.qpos[qadr].copy()
        tcp = d.site_xpos[tcp_sid].copy()
        if carry:                                                    # carrying the cube: expert places it from here
            Rm = d.site_xmat[tcp_sid].reshape(3, 3).copy()
            off = d.xpos[obj][:2] - tcp[:2]                          # where the cube sits in the gripper
            segs = plan_wps(P, q_b, place_wps(place, tcp[:2], tcp[2], off), Rm, 1.0)
            cube = d.xpos[obj].copy()
        if carry and segs is None:
            fail["plan"] += 1; continue
        cube = d.xpos[obj].copy() if not carry else cube
        yaw = yaw_of(d.xquat[obj].copy())
        if not carry and (yaw is None or cube[2] > OBJECT_HALF_SIZE + 0.01):
            fail["tipped"] += 1; continue
        if not carry:
            cube[2] = OBJECT_HALF_SIZE
            low = tcp[2] < OBJECT_HALF_SIZE + 0.07 and np.linalg.norm(tcp[:2] - cube[:2]) < 0.08 or \
                np.linalg.norm(cube[:2] - place[:2]) < args.near_place
            segs = None
            if low and not args.no_local_fix and psi is None:
                # local correction: over the cube at ~6 cm (above the cube top), then down -- no 10 cm hover
                wps = grasp_wps(tcp, cube, place)
                for k in sorted(range(4), key=lambda k: abs(np.angle(np.exp(1j * (yaw + k * np.pi / 2 - np.arctan2(
                        d.site_xmat[tcp_sid][3], d.site_xmat[tcp_sid][0])))))):   # yaw closest to the current one
                    segs = plan_wps(P, q_b, wps, GEN.R_down(yaw + k * np.pi / 2), 0.0)
                    if segs is not None:
                        n_local += 1; break
            if segs is None:
                segs = GEN.plan(pm, pd_, psite, pqadr, pvadr, plo, phi, q_b, cube, yaw, place, psi)
            if segs is None:
                fail["plan"] += 1; continue
        lifted = False
        hb = (not carry) and rng.random() < args.handback       # multi-intervention: hand back after the lift
        handed = False

        def keep_history():                                     # the core must have a current history when it resumes
            qn = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
            zh.pop(0); zh.append(zarm(qn))
            if use_tcp:
                th.pop(0); th.append(tcp_tok(qn))
            fh.pop(0); fh.append(float(d.qpos[g_qadr]) if g_qadr is not None else 0.0)
            zo_ = see(); ch.pop(0); ch.append(zo_[0].clone())
        for q_seq, g_seq, phase in segs:
            if hb and phase >= 4 and lifted:                    # expert grasped and lifted: back to the core
                handed = True; break
            for q_t, gg in zip(q_seq, g_seq):
                rows.append(row(q_t, gg > 0.5, phase, cube, place, expert=1.0))
                if grip_act is not None:
                    d.ctrl[grip_act] = GRIPPER_CLOSED if gg > 0.5 else GRIPPER_OPEN
                gstate[0] = gg > 0.5
                trk.step(d, q_t, SUB)
                if phase == 3 and d.xpos[obj][2] > OBJECT_HALF_SIZE + 0.01:
                    lifted = True
                if hb:
                    keep_history()
        if handed:
            # ---- 3) the core carries the cube (its own drift), then 4) the expert places it from there
            runner.reset(); shaper = CommandShaper(VSTEP.cpu().numpy(), args.accel_steps, args.smooth)
            t_end = int(rng.integers(10, args.post_max + 1)); why_hb = "hb_post_t"
            grip_closed = True; gstate[0] = True
            for step2 in range(t_end):
                qt = torch.tensor(d.qpos[qadr].astype(np.float32)[None], device=DEV)
                hist, f_hist, cube_hist = torch.stack(zh)[None], torch.tensor([fh], dtype=torch.float32, device=DEV), torch.stack(ch)[None]

                def predict2():
                    o, l = core(hist, f_hist, cube_hist, tape_goal[None], torch.stack(th)[None] if use_tcp else None)
                    return o[0], torch.sigmoid(l[0])
                o_, p_c = runner(step2, predict2)
                if p_c.item() <= 0.5:                               # wants to open: the expert takes over here
                    why_hb = "hb_open"; break
                q_goal = inverse_decode(o_[None], qt, fk, emb, lo=lim[:, 0], hi=lim[:, 1])
                f_goal = fk.forward(q_goal)
                q_goal = torch.max(torch.min(tcp_ik(fk, qt, f_goal[:, :3], f_goal[:, 3:9]), lim[:, 1]), lim[:, 0])
                q_c = shaper(qt[0].cpu().numpy(), q_goal[0].cpu().numpy())
                rows.append(row(q_c, True, 3, cube, place, expert=0.0))
                trk.step(d, q_c, SUB)
                if d.xpos[obj][2] < OBJECT_HALF_SIZE + 0.02:
                    why_hb = "hb_drop"; break
                keep_history()
            if why_hb == "hb_drop":
                fail["dropped"] += 1; continue
            q_b2 = d.qpos[qadr].copy(); tcp2 = d.site_xpos[tcp_sid].copy()
            Rm2 = d.site_xmat[tcp_sid].reshape(3, 3).copy()
            segs2 = plan_wps(P, q_b2, place_wps(place, tcp2[:2], tcp2[2], d.xpos[obj][:2] - tcp2[:2]), Rm2, 1.0)
            if segs2 is None:
                fail["plan"] += 1; continue
            for q_seq, g_seq, phase in segs2:
                for q_t, gg in zip(q_seq, g_seq):
                    rows.append(row(q_t, gg > 0.5, phase, cube, place, expert=1.0))
                    if grip_act is not None:
                        d.ctrl[grip_act] = GRIPPER_CLOSED if gg > 0.5 else GRIPPER_OPEN
                    trk.step(d, q_t, SUB)
            why[why_hb] = why.get(why_hb, 0) + 1
        trk.hold(d, int(0.5 * SIM_HZ))
        c = d.xpos[obj]
        ok = np.linalg.norm(c[:2] - place[:2]) < args.succ_mm / 1000.0 and c[2] < OBJECT_HALF_SIZE + 0.01
        if not lifted:
            fail["grasp"] += 1; continue
        if not ok:
            fail["place"] += 1; continue

        idx = args.start_index + kept
        df = pd.DataFrame(rows)
        df["frame_index"] = np.arange(len(df)); df["episode_index"] = idx
        df["timestamp"] = df["frame_index"] / CONTROL_HZ; df["index"] = 0; df["task_index"] = 0
        if use_psi:
            df["psi_goal"] = [np.array(psi if psi is not None else (0.0, 0.0), dtype=np.float32)] * len(df)
        name = f"episode_{idx:06d}.parquet"
        df.to_parquet(os.path.join(out, name))
        dmeta[name] = b
        ef = df["expert_frame"].values > 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, ef.astype(int), 0]))
        emeta[name] = [[int(edges[k]), int(edges[k + 1])] for k in range(0, len(edges), 2)]
        why[reason] += 1; n_random += int(is_rand)
        if reason in xy_at_branch:
            xy_at_branch[reason].append(np.linalg.norm(tcp[:2] - cube[:2]) * 1000)
        kept += 1
        if kept % 20 == 0 or kept == args.episodes:
            json.dump(dmeta, open(meta_p, "w"), indent=0); json.dump(emeta, open(emeta_p, "w"))
            print(f"[{kept}/{args.episodes}] tried {tried}  kept {kept / tried:.0%}  failures {fail}  branch at "
                  f"{why}  random targets {n_random}  local fixes {n_local}\n           TCP-cube xy at pre-grasp branch median "
                  f"{' / '.join(f'{k} {np.median(v):.0f}' if v else f'{k} -' for k, v in xy_at_branch.items())} mm  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(dmeta, open(meta_p, "w"), indent=0); json.dump(emeta, open(emeta_p, "w"))
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    with open(os.path.join(root, "meta", "info.json"), "w") as f:
        json.dump({"source": "dagger_pp_vision.py", "core": args.ckpt, "base_dataset": cfg["dataset"],
                   "fps": CONTROL_HZ, "total_episodes": len(dmeta), "gen_args": args.gen_args,
                   "note": "rows < dagger_meta[file] are policy frames (history only)"}, f, indent=2)
    print(f"[done] {kept} episodes -> {out}\n       branch points -> {meta_p}")


if __name__ == "__main__":
    main()