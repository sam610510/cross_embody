"""
FR3 pick-and-place expert demonstrations (scripted), in the same MuJoCo scene as the
reaching data (build_scene.py: FR3 + Franka Hand + table + 5 cm cube).

Each episode:
  random start near HOME_Q, cube at the scene's fixed OBJECT_POS (or random with
  --random_cube), place point fixed (or random with --random_place; each episode stores its own
  goal_xyz / place_xyz, which the decision core gets as input)
  phases (TCP waypoints, gripper pointing down):
    0 approach  -> 10 cm above the cube
    1 descend   -> grasp height
    2 close     -> gripper closes (hold)
    3 lift      -> 15 cm up
    4 transfer  -> 10 cm above the place point
    5 lower     -> place height
    6 open      -> gripper opens (hold)
    7 retreat   -> 12 cm above the place point
  every waypoint is solved by damped-least-squares IK on the tcp_site, seeded from the
  previous waypoint (continuous elbow), with a weak pull toward HOME_Q; joint motion
  between waypoints is a quintic profile timed by the joint velocity limits.
  The episode is executed in physics with the scene's position actuators; it is kept
  only if the cube really ends at the place point (|xy| < 2 cm, resting on the table).

Recorded at CONTROL_HZ per frame, same LeRobot-v2-style layout and column conventions as
generate_fr3_expert_data.py (observation.state = arm(7) + hand_finger_joint1, action =
arm command(7) + raw gripper command 0..255), plus:
  gripper_closed (0/1), object_xyz (cube now), goal_xyz (cube start), place_xyz (fixed),
  phase (0..7) and phase_name.
Timing convention: observation.state[t] is the state BEFORE action[t] is applied, so
action[t] is the command that moves the arm from state t toward state t+1.
(generate_fr3_expert_data.py logs the state AFTER stepping, i.e. action[t] already produced
observation.state[t] -- keep this in mind when comparing the two datasets.)

Run:
  python fr3_pick_gen/scripts/generate_fr3_pickplace_data.py --episodes 300
  MUJOCO_GL=egl python fr3_pick_gen/scripts/generate_fr3_pickplace_data.py --episodes 3 --video
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_scene import build_model, OBJECT_POS, OBJECT_HALF_SIZE
from generate_fr3_expert_data import (resolve_joint_and_actuator_ids, sample_random_init, HOME_Q,
                                      CONTROL_HZ, SIM_HZ, INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES,
                                      GRIPPER_OPEN, GRIPPER_CLOSED, N_ARM_JOINTS)
from domain_rand import apply_domain_randomization
from joint_tracker import JointTracker, FR3_TAU

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--place", type=float, nargs=3, default=[0.45, 0.25, 0.0], help="fixed place point (table top)")
ap.add_argument("--cube_x", type=float, nargs=2, default=[0.30, 0.65])
ap.add_argument("--cube_y", type=float, nargs=2, default=[-0.30, 0.35])
ap.add_argument("--swivel_targets", action="store_true",
                help="arm posture as a goal: every episode draws a target swivel (elbow) angle psi1 for the pick "
                     "waypoints and psi2 for the place waypoints; each group picks the gripper yaw (cube faces, "
                     "k*90 deg) that can take it, and the joints come from IK on the FR3 null-space")
ap.add_argument("--psi1_range", type=float, nargs=2, default=[-12.0, 12.0], help="deg, pick group")
ap.add_argument("--psi2_range", type=float, nargs=2, default=[-30.0, 55.0], help="deg, place group")
ap.add_argument("--swivel_tol", type=float, default=1.5, help="deg, accepted swivel error at a waypoint")
ap.add_argument("--cart_speed", type=float, default=0.06,
                help="m/s, TCP speed that times the segments in --swivel_targets mode")
ap.add_argument("--yaw_rule", choices=["goal", "start"], default="goal",
                help="goal: the gripper yaw of each group follows from the posture goal in a fixed order (learnable "
                     "from the goal input); start: least joint motion from the start pose (old, start-dependent)")
ap.add_argument("--psi_tries", type=int, default=4,
                help="if no gripper yaw can take the drawn posture, draw new posture goals this many times")
ap.add_argument("--random_place", action="store_true",
                help="randomize the place point within --place_x/--place_y (default: fixed --place)")
ap.add_argument("--place_x", type=float, nargs=2, default=[0.30, 0.65])
ap.add_argument("--place_y", type=float, nargs=2, default=[-0.30, 0.35])
ap.add_argument("--min_sep", type=float, default=0.12, help="min distance cube start - place point (m)")
ap.add_argument("--holdout_box", type=float, nargs=4, default=[0.44, 0.54, 0.00, 0.15], metavar=("X0", "X1", "Y0", "Y1"),
                help="NO training target (cube start or place point) inside this xy box -- kept for testing "
                     "generalization to unseen target positions; pass 0 0 0 0 to disable")
ap.add_argument("--cube_yaw", type=float, default=0.0, help="max |yaw| of the cube (deg)")
ap.add_argument("--random_cube", action="store_true",
                help="randomize the cube position within --cube_x/--cube_y (default: the scene's fixed "
                     "OBJECT_POS, so no perception is needed -- only the start pose is random)")
ap.add_argument("--out", default="~/cross_embody/fr3_pick_gen/datasets/expert_pp_v2/data/chunk-000")
ap.add_argument("--video", action="store_true", help="also save an MP4 of each episode (use few episodes)")
ap.add_argument("--speed_frac", type=float, default=0.15,
                help="fraction of the joint velocity limits the expert moves at (lower = slower, more frames)")
ap.add_argument("--min_seg", type=float, default=1.0, help="minimum duration of each motion segment (s)")
ap.add_argument("--hold_close", type=float, default=0.4, help="dwell while the gripper closes (s) -- just long "
                "enough to close, so there is no long motionless stretch whose end the policy cannot time")
ap.add_argument("--hold_open", type=float, default=0.4, help="dwell while the gripper opens (s)")
ap.add_argument("--stop_at_waypoints", action="store_true",
                help="old behaviour (v1): every segment starts and ends at rest. Default: the arm passes "
                     "the via points (above the cube, top of the lift, above the place point) without stopping")
ap.add_argument("--no_domain_rand", action="store_true",
                help="skip apply_domain_randomization (gravity / friction / lighting), on by default "
                     "like generate_fr3_expert_data.py")
args = ap.parse_args()

HOVER, LIFT, RETREAT = 0.10, 0.15, 0.12
GRASP_Z = OBJECT_HALF_SIZE     # TCP (fingertip midpoint) at the cube centre
PLACE_Z = OBJECT_HALF_SIZE + 0.005
HOLD_S = {2: args.hold_close, 6: args.hold_open}     # gripper close / open dwell
PHASES = ["approach", "descend", "close", "lift", "transfer", "lower", "open", "retreat"]
TASK = "Pick the cube and place it at the fixed point"
VLIM = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
SPEED_FRAC = args.speed_frac   # fraction of the velocity limit used by the expert
SUBSTEPS = max(1, round(SIM_HZ / CONTROL_HZ))
OUT = Path(os.path.expanduser(args.out))


# ------------------------------------------------------------------ IK
def rot_err(R_t, R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (R_t @ R.T).flatten())
    v = q[1:]; n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3)
    ang = 2 * np.arctan2(n, q[0])
    if ang > np.pi:
        ang -= 2 * np.pi
    return v / n * ang


def ik(m, d, site, qadr, vadr, p_t, R_t, q0, lo, hi, iters=300, ns=0.02):
    """damped least squares on tcp_site (position + orientation), null-space pull to HOME"""
    q = q0.copy(); home = np.asarray(HOME_Q)[:7]
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[qadr] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        e = np.concatenate([p_t - d.site_xpos[site], rot_err(R_t, d.site_xmat[site].reshape(3, 3))])
        if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, vadr], jr[:, vadr]])
        JJt = J @ J.T + 1e-4 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
        dq += N @ (ns * (home - q))
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    d.qpos[qadr] = q; mujoco.mj_kinematics(m, d)
    err = np.linalg.norm(p_t - d.site_xpos[site])
    return q, err


def quintic(q0, q1, n):
    s = np.linspace(0, 1, n)
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return q0 + (q1 - q0) * s[:, None]


_SW = {}


def swivel_tools():
    """FR3 FK encoder (the same one the latent uses) + IK / swivel helpers, loaded on first use"""
    if not _SW:
        import torch
        root = os.path.expanduser("~/cross_embody"); sys.path.insert(0, root)
        from fk_encoder.fk_encoder import FKEncoder
        from decision_core.model import tcp_ik
        from decision_core.swivel import swivel, project_swivel, wrap
        _SW.update(torch=torch, enc=FKEncoder("fr3", os.path.join(root, "configs", "robots.yaml"), project_root=root),
                   tcp_ik=tcp_ik, swivel=swivel, project=project_swivel, wrap=wrap)
    return _SW


# yaw preference orders (k of k*90 deg), from check_swivel_range.py at the scene's cube / place point:
#   pick group:  +0 / +90 reach [-15, 10],  +180 reaches [-10, 15]   -> negative psi1: +0 first, else +180
#   place group: +90 reaches [-35, 65] (covers the psi2 range)          -> +90 first
PICK_ORDER_NEG, PICK_ORDER_POS, PLACE_ORDER = [0, 1, 3, 2], [2, 3, 1, 0], [1, 2, 3, 0]
FINGER_YAW0 = -np.pi / 4      # encoder-frame TCP yaw at which the FR3 fingers close along world x
                              # (finger axis [0.707, -0.707] in the encoder TCP frame, check_robot_fk.py)


def r6_down_enc(yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    Rm = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ np.diag([1.0, -1.0, -1.0])
    return np.concatenate([Rm[:, 0], Rm[:, 1]])


def solve_group(points, psi, cube_yaw, q_seed, lo, hi, order=None):
    """joint configs for consecutive TCP points (gripper down) with swivel psi at every point, using the
    gripper yaw (fingers on the cube faces, k*90) that can take psi everywhere; None if no yaw can.
    The 4 yaws are solved together as one batch."""
    T = swivel_tools(); torch = T["torch"]; enc = T["enc"]
    lo_t, hi_t = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
    tol = np.radians(args.swivel_tol)
    r6 = torch.tensor(np.stack([r6_down_enc(FINGER_YAW0 + cube_yaw + k * np.pi / 2) for k in range(4)]),
                      dtype=torch.float32)
    q = torch.tensor(q_seed, dtype=torch.float32)[None].repeat(4, 1)
    tgt = torch.full((4,), float(psi))
    ok = torch.ones(4, dtype=torch.bool); qs = []
    for p in points:
        pt = torch.tensor(p, dtype=torch.float32)[None].repeat(4, 1)
        q = T["project"](enc, q, pt, r6, tgt, lo_t, hi_t, T["tcp_ik"], outer=8)
        ok &= ((enc.forward(q)[:, :3] - pt).norm(dim=1) < 1e-3) & (T["wrap"](T["swivel"](enc, q) - tgt).abs() < tol)
        qs.append(q.numpy().astype(float))
    if not ok.any():
        return None
    if order is not None:                       # deterministic yaw: the first feasible one in a fixed order
        k = next((k for k in order if ok[k]), None)
        if k is None:
            return None
    else:                                       # least joint motion from the start (yaw depends on the start)
        cost = [np.abs(qs[0][k] - q_seed).sum() if ok[k] else np.inf for k in range(4)]
        k = int(np.argmin(cost))
    return [qq[k] for qq in qs]


def hermite(q0, q1, v0, v1, n):
    """cubic Hermite from q0 (velocity v0) to q1 (velocity v1) in n control steps; velocities in rad/s"""
    T = n / CONTROL_HZ
    s = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * q0 + h10 * T * v0 + h01 * q1 + h11 * T * v1


def R_down(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ np.diag([1.0, -1.0, -1.0])        # tool z pointing down


# ------------------------------------------------------------------ episode
def plan(m, d, site, qadr, vadr, lo, hi, q_start, cube, yaw, place, psi=None):
    """joint-space plan: list of (q_targets (n,7), gripper closed (n,), phase)"""
    # the cube is square, so the fingers may close on either pair of faces (yaw + k*90°):
    # take the reachable one needing the least joint motion from the start
    best = None
    for k in range(4):
        Rk = R_down(yaw + k * np.pi / 2)
        qk, ek = ik(m, d, site, qadr, vadr, cube + [0, 0, HOVER], Rk, q_start, lo, hi)
        if ek < 0.003 and (best is None or np.abs(qk - q_start).sum() < best[0]):
            best = (np.abs(qk - q_start).sum(), Rk)
    if best is None and psi is None:
        return None
    R = best[1] if best is not None else None
    wps = [(0, cube + [0, 0, HOVER]), (1, cube + [0, 0, 0]), (3, cube + [0, 0, LIFT]),
           (4, place + [0, 0, HOVER]), (5, place + [0, 0, 0]), (7, place + [0, 0, RETREAT])]
    z_of = {1: GRASP_Z, 5: PLACE_Z}
    # 1) joint waypoints and segment lengths
    qs, ns, q_prev = [], [], q_start.copy()
    if psi is not None:
        # posture goal: pick group (above cube, grasp, lifted) at psi[0], place group (above place,
        # place, retreat) at psi[1]; each group with its own gripper yaw
        pts = []
        for phase, p in wps:
            p = p.copy()
            if phase in z_of:
                p[2] = z_of[phase]
            pts.append(p)
        if args.yaw_rule == "goal":
            # the gripper yaw must be a function of what the policy is told (the posture goals), not of the
            # start pose -- otherwise the same goal leads to different yaws and the policy averages them
            o1 = PICK_ORDER_NEG if psi[0] < 0 else PICK_ORDER_POS
            o2 = PLACE_ORDER
        else:
            o1 = o2 = None
        g1 = solve_group(pts[:3], psi[0], yaw, q_start, lo, hi, o1)
        if g1 is None:
            return None
        g2 = solve_group(pts[3:], psi[1], yaw, g1[-1], lo, hi, o2)
        if g2 is None:
            return None
        # segment length from the TCP distance (Cartesian speed), not from the joint distance: with a
        # posture constraint the elbow may move a lot in joint space while the TCP moves 10 cm, and a
        # joint-timed segment then crawls (descent of 10 cm took ~170 frames = 5.7 s: the policy saw an
        # almost motionless hover and learned to stop there). Joint speed is still capped.
        T_ = swivel_tools(); p_prev = T_["enc"].forward(T_["torch"].tensor(q_prev, dtype=T_["torch"].float32)[None])[0, :3].numpy()
        for q_t, p_t in zip(g1 + g2, pts):
            n_cart = int(np.ceil(np.linalg.norm(np.asarray(p_t) - p_prev) / args.cart_speed * CONTROL_HZ))
            n_joint = int(np.ceil(np.max(np.abs(q_t - q_prev) / (0.5 * VLIM)) * CONTROL_HZ))
            ns.append(max(n_cart, n_joint, int(args.min_seg * CONTROL_HZ)))
            qs.append(q_t); q_prev = q_t; p_prev = np.asarray(p_t)
    for phase, p in (wps if psi is None else []):
        p = p.copy()
        if phase in z_of:
            p[2] = z_of[phase]
        q_t, err = ik(m, d, site, qadr, vadr, p, R, q_prev, lo, hi)
        if err > 0.003:
            return None
        ns.append(max(int(np.ceil(np.max(np.abs(q_t - q_prev) / (SPEED_FRAC * VLIM)) * CONTROL_HZ)),
                      int(args.min_seg * CONTROL_HZ)))
        qs.append(q_t); q_prev = q_t
    # 2) velocity at each waypoint: rest where the gripper acts (grasp, place) and at the end; at the
    #    via points (above the cube, top of the lift, above the place point) the arm keeps moving
    #    with the NEXT segment's average velocity, so it is already heading down / across there
    #    instead of stopping (a motionless instant whose 'start again' the policy cannot time)
    vel = []
    for i, (phase, _) in enumerate(wps):
        if args.stop_at_waypoints or phase in (1, 5, 7) or i + 1 >= len(wps):
            vel.append(np.zeros(7))
        else:
            vel.append((qs[i + 1] - qs[i]) / (ns[i + 1] / CONTROL_HZ))
    segs, q_prev, v_prev, grip = [], q_start.copy(), np.zeros(7), 0.0
    for i, (phase, _) in enumerate(wps):
        q_t, n = qs[i], ns[i]
        traj = quintic(q_prev, q_t, n) if args.stop_at_waypoints else hermite(q_prev, q_t, v_prev, vel[i], n)
        segs.append((traj, np.full(n, grip), phase))
        q_prev, v_prev = q_t, vel[i]
        if phase == 1:                                         # close after the descent
            grip = 1.0
            k = int(HOLD_S[2] * CONTROL_HZ)
            segs.append((np.repeat(q_t[None], k, 0), np.full(k, grip), 2))
        if phase == 5:                                         # open after lowering
            grip = 0.0
            k = int(HOLD_S[6] * CONTROL_HZ)
            segs.append((np.repeat(q_t[None], k, 0), np.full(k, grip), 6))
    if psi is not None:
        segs = [(posture_path(tr, lo, hi), g, ph) for tr, g, ph in segs]
    return segs


def posture_path(traj, lo, hi):
    """Joint-space interpolation keeps the TCP path smooth but lets the elbow wander between two
    waypoints (it dipped ~5 deg mid-descent). Keep every frame's TCP and move its elbow along the
    null-space to a swivel that goes linearly -- in TCP path length -- from the segment's start
    posture to its end posture. Frames the projection cannot fix are filled by joint interpolation
    between their fixed neighbours, so the result stays continuous."""
    n = len(traj)
    if n < 3 or np.abs(traj[-1] - traj[0]).max() < 1e-6:
        return traj
    T = swivel_tools(); torch = T["torch"]; enc = T["enc"]
    q = torch.tensor(traj, dtype=torch.float32)
    f = enc.forward(q)
    ps = T["swivel"](enc, q)
    a, b = ps[0], ps[-1]
    dist = torch.cat([torch.zeros(1), (f[1:, :3] - f[:-1, :3]).norm(dim=1).cumsum(0)])
    frac = dist / dist[-1] if dist[-1] > 1e-6 else torch.linspace(0, 1, n)
    tgt = a + T["wrap"](b - a) * frac
    lo_t, hi_t = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
    qn = T["project"](enc, q, f[:, :3], f[:, 3:9], tgt, lo_t, hi_t, T["tcp_ik"], outer=8)
    good = ((T["wrap"](T["swivel"](enc, qn) - tgt).abs() < np.radians(1.0))
            & ((enc.forward(qn)[:, :3] - f[:, :3]).norm(dim=1) < 1e-3)).numpy()
    good[0] = good[-1] = True
    qn[0], qn[-1] = q[0], q[-1]                                   # waypoints stay exactly as planned
    out = qn.numpy().astype(float)
    gi = np.nonzero(good)[0]
    for j in range(7):                                            # fill the frames that could not be fixed
        out[:, j] = np.interp(np.arange(n), gi, out[gi, j])
    step_old = np.abs(np.diff(traj, axis=0)).max()
    if np.abs(np.diff(out, axis=0)).max() > 2 * step_old + 0.02:  # never introduce a jump
        STATS["posture_kept"] += 1
        return traj
    STATS["posture_fixed"] += 1; STATS["frames_unfixable"] += int((~good).sum())
    return out


STATS = {"posture_fixed": 0, "posture_kept": 0, "frames_unfixable": 0}


def brighten(m):
    """video only (the policy never sees pixels): strong key light + ambient fill, and a soft
    headlight so the side of the arm facing the camera is not in shadow"""
    for i in range(m.nlight):
        m.light_diffuse[i] = [1.0, 1.0, 1.0]
        m.light_ambient[i] = [0.45, 0.45, 0.45]
        m.light_specular[i] = [0.3, 0.3, 0.3]
    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = [0.15, 0.15, 0.15]
    m.vis.headlight.diffuse[:] = [0.35, 0.35, 0.35]
    m.vis.headlight.specular[:] = [0.05, 0.05, 0.05]


def main():
    rng = np.random.default_rng(args.seed); np.random.seed(args.seed)
    m = build_model(); d = mujoco.MjData(m)
    jnt_ids, qadr, arm_act, grip_act, g_qadr = resolve_joint_and_actuator_ids(m)
    qadr = np.asarray(qadr); vadr = np.array([m.jnt_dofadr[j] for j in jnt_ids])
    lo, hi = m.jnt_range[jnt_ids, 0], m.jnt_range[jnt_ids, 1]
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    oj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    o_adr, o_vadr = m.jnt_qposadr[oj], m.jnt_dofadr[oj]
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    place = np.array(args.place, dtype=float)
    trk = JointTracker(m, qadr, vadr, FR3_TAU, actuator_ids=arm_act)      # same controller as every arm
    OUT.mkdir(parents=True, exist_ok=True)
    renderer = None
    if args.video:
        import imageio
        m.vis.global_.offwidth, m.vis.global_.offheight = 960, 720
        renderer = mujoco.Renderer(m, 720, 960)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.azimuth, cam.elevation, cam.lookat = 1.5, 100, -20, np.array([0.45, 0.05, 0.05])
        brighten(m)

    kept, tried, t0, n_frames, episodes_meta = 0, 0, time.time(), 0, []
    fail = {"ik": 0, "grasp": 0, "place": 0}
    while kept < args.episodes:
        tried += 1
        if not args.no_domain_rand:
            apply_domain_randomization(m)
            if args.video:
                brighten(m)        # domain randomization redraws the light every episode
        mujoco.mj_resetData(m, d)
        def held_out(p):
            x0, x1, y0, y1 = args.holdout_box
            return x0 < p[0] < x1 and y0 < p[1] < y1
        if args.random_cube:
            while True:
                cube = np.array([rng.uniform(*args.cube_x), rng.uniform(*args.cube_y), OBJECT_HALF_SIZE])
                if not held_out(cube):
                    break
        else:
            cube = np.array(OBJECT_POS, dtype=float)
        if args.random_place:
            while True:
                place = np.array([rng.uniform(*args.place_x), rng.uniform(*args.place_y), args.place[2]])
                if np.linalg.norm(place[:2] - cube[:2]) >= args.min_sep and not held_out(place):
                    break
        yaw = np.radians(rng.uniform(-args.cube_yaw, args.cube_yaw))
        d.qpos[o_adr:o_adr + 3] = cube
        d.qpos[o_adr + 3:o_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        q_start = np.asarray(sample_random_init(m, d, HOME_Q, jnt_ids, INIT_SAMPLE_FRACTION,
                                                N_COLLISION_CHECK_TRIES))[:7]
        d.qpos[o_adr:o_adr + 3] = cube                         # (the sampler may have touched qpos)
        d.qpos[o_adr + 3:o_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        psi, segs = None, None
        for _ in range(args.psi_tries if args.swivel_targets else 1):
            if args.swivel_targets:
                psi = (np.radians(rng.uniform(*args.psi1_range)), np.radians(rng.uniform(*args.psi2_range)))
            segs = plan(m, d, site, qadr, vadr, lo, hi, q_start, cube, yaw, place, psi)
            if segs is not None:
                break
        if segs is None:
            fail["ik"] += 1
            continue
        # execute in physics
        mujoco.mj_resetData(m, d)
        d.qpos[qadr] = q_start
        d.qpos[o_adr:o_adr + 3] = cube
        d.qpos[o_adr + 3:o_adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        if grip_act is not None:
            d.ctrl[grip_act] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)
        rows, frames, lifted = [], [], False
        for q_seq, g_seq, phase in segs:
            for q_t, g in zip(q_seq, g_seq):
                g_raw = GRIPPER_CLOSED if g > 0.5 else GRIPPER_OPEN
                finger = float(d.qpos[g_qadr]) if g_qadr is not None else 0.0
                rows.append({"observation.state": np.append(d.qpos[qadr], finger).astype(np.float32),
                             "observation.velocity": np.append(d.qvel[vadr], 0.0).astype(np.float32),
                             "action": np.append(q_t, g_raw).astype(np.float32),
                             "gripper_closed": float(g > 0.5),
                             "object_xyz": d.xpos[obj_body].astype(np.float32).copy(),
                             "goal_xyz": cube.astype(np.float32), "place_xyz": place.astype(np.float32),
                             "phase": phase, "phase_name": PHASES[phase]})
                if grip_act is not None:
                    d.ctrl[grip_act] = g_raw
                trk.step(d, q_t, SUBSTEPS)
                if phase == 3 and d.xpos[obj_body][2] > OBJECT_HALF_SIZE + 0.05:
                    lifted = True
                if renderer is not None:
                    renderer.update_scene(d, camera=cam); frames.append(renderer.render().copy())
        # settle, then check the placement
        trk.hold(d, int(0.5 * SIM_HZ))
        obj = d.xpos[obj_body]
        ok_place = np.linalg.norm(obj[:2] - place[:2]) < 0.02 and obj[2] < OBJECT_HALF_SIZE + 0.01
        if not lifted:
            fail["grasp"] += 1
        elif not ok_place:
            fail["place"] += 1
        if renderer is not None:
            p = OUT.parent.parent / f"videos/ep{tried:03d}_{'ok' if lifted and ok_place else 'fail'}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(p, frames, fps=CONTROL_HZ)
            print(f"  [video] {p}  ({len(frames)} frames, {len(frames) / CONTROL_HZ:.1f} s)")
        if not (lifted and ok_place):
            continue
        df = pd.DataFrame(rows)
        if args.swivel_targets:
            # posture goal of the episode + the swivel actually taken at every frame (for checking)
            T_ = swivel_tools(); tq = T_["torch"].tensor(np.stack(df["observation.state"].values)[:, :7],
                                                          dtype=T_["torch"].float32)
            df["psi_goal"] = [np.array(psi, dtype=np.float32)] * len(df)
            df["psi_now"] = T_["swivel"](T_["enc"], tq).numpy().astype(np.float32)
        df["frame_index"] = np.arange(len(df)); df["episode_index"] = kept
        df["timestamp"] = df["frame_index"] / CONTROL_HZ
        df["index"] = np.arange(n_frames, n_frames + len(df)); df["task_index"] = 0
        df.to_parquet(OUT / f"episode_{kept:06d}.parquet")
        episodes_meta.append({"episode_index": kept, "length": len(df), "tasks": [TASK]})
        n_frames += len(df)
        kept += 1
        if kept % 20 == 0 or kept == args.episodes:
            print(f"[{kept}/{args.episodes}] tried {tried}  success {kept / tried:.0%}  "
                  f"failures {fail}  ({time.time() - t0:.0f}s)  last episode {len(df)} frames")
    meta = OUT.parent.parent / "meta"; meta.mkdir(parents=True, exist_ok=True)
    with open(meta / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes_meta)
    with open(meta / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    with open(meta / "info.json", "w") as f:
        json.dump({"codebase_version": "v2.0 (hand-written, non-lerobot-lib)", "robot_type": "fr3",
                   "fps": CONTROL_HZ, "total_episodes": kept, "total_frames": n_frames,
                   "state_timing": "observation.state[t] is BEFORE action[t]",
                   "place_xyz": place.tolist(),
                   "features": {"observation.state": {"dtype": "float32", "shape": [N_ARM_JOINTS + 1]},
                                "observation.velocity": {"dtype": "float32", "shape": [N_ARM_JOINTS + 1]},
                                "action": {"dtype": "float32", "shape": [N_ARM_JOINTS + 1]}}}, f, indent=2)
    if args.swivel_targets:
        print(f"[posture] segments re-posed {STATS['posture_fixed']}, left as joint interpolation "
              f"{STATS['posture_kept']} (would have jumped), frames filled by interpolation {STATS['frames_unfixable']}")
    print(f"[done] {kept} episodes -> {OUT}   expert success {kept / tried:.0%}  failures {fail}")


if __name__ == "__main__":
    main()