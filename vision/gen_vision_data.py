#!/usr/bin/env python3
"""Vision dataset generator for the object-position encoder (vision_plan step 1).

One fixed RGB camera looks at the FR3 workspace. Each sample randomizes
  * cube / tape (place target) positions on the table
  * FR3 posture (random, near-home, hovering over the cube or the tape)
  * FR3 / table / object colours, lights, camera extrinsics (small jitter)
  * non-colliding occluders on the camera-to-object line and distractors on the table
and stores the RGB image plus labels that do NOT depend on the arm:
  world position, projected pixel (u, v), visible fraction of each object.

Only the source arm (FR3) is ever rendered here. The same script renders a UR5e
test set by pointing --scene/--robot_root/--arm_joint_regex at the UR5e scene.

Example
  python gen_vision_data.py --scene path/to/scene.xml --list          # print names
  python gen_vision_data.py --scene path/to/scene.xml --out data/vis_fr3_train \
      --n 50000 --workers 16 --seed 0
"""
import argparse
import colorsys
import json
import os
import re
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HIDE_GROUP = 5
MODES = ("random", "home", "hover_cube", "hover_tape")
SEG_BG, SEG_CUBE, SEG_TAPE, SEG_ROBOT, SEG_OCC, SEG_OTHER = 0, 1, 2, 3, 4, 5


# --------------------------------------------------------------------------- args
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="", help="MJCF of the robot scene (arm + gripper + cube + tape + table)")
    p.add_argument("--pick_gen", default="", help="build the scene with <dir>/scene.py:build_spec instead "
                   "(the fr3_pick_gen scene); then also give --menagerie and --robosuite")
    p.add_argument("--scene_module", default="build_scene",
                   help="module in --pick_gen that defines build_spec() (e.g. build_scene or scene)")
    p.add_argument("--menagerie", default="", help="mujoco_menagerie path for --pick_gen")
    p.add_argument("--robosuite", default="", help="robosuite path for --pick_gen")
    p.add_argument("--p_distractor", type=float, default=0.7,
                   help="--pick_gen: probability the bottle/can/milk distractors are placed on the table "
                        "(otherwise hidden)")
    p.add_argument("--out", default="data/vis_fr3")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--start", type=int, default=0, help="first sample index (lets you append / split)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", action="store_true", help="print body/joint/geom/site/camera names and exit")

    g = p.add_argument_group("scene names")
    g.add_argument("--robot_root", default="base", help="root body of the arm subtree (gripper included)")
    g.add_argument("--arm_joint_regex", default=r"(fr3_)?joint[1-7]")
    g.add_argument("--finger_joint_regex", default=r".*finger.*")
    g.add_argument("--ee_site", default="attachment_site", help="site used for hover IK (TCP if you have one)")
    g.add_argument("--cube", default="cube", help="body (free or static) or geom name")
    g.add_argument("--tape", default="place_target", help="body or geom name of the place target")
    g.add_argument("--table_geom", default="table")
    g.add_argument("--table_z", type=float, default=None,
                   help="table top height [m]; default 0, or TABLE_HEIGHT of the --pick_gen scene")
    g.add_argument("--keyframe", default="home", help="keyframe for initial state ('' = qpos0)")

    g = p.add_argument_group("camera (defaults ~ RealSense colour stream)")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--cam_name", default="scene_cam", help="use this existing scene camera as the base pose "
                   "(ignored if absent; then --cam_pos/--cam_lookat are used)")
    g.add_argument("--fovy", type=float, default=None, help="vertical FOV [deg]; default = scene camera's, else 42.5")
    g.add_argument("--cam_pos", type=float, nargs=3, default=[1.25, 0.45, 0.75])
    g.add_argument("--cam_lookat", type=float, nargs=3, default=[0.45, 0.05, 0.05])
    g.add_argument("--cam_pos_std", type=float, default=0.01, help="extrinsic jitter, position std [m]")
    g.add_argument("--cam_rot_std", type=float, default=1.0, help="extrinsic jitter, rotation std [deg]")
    g.add_argument("--fovy_std", type=float, default=0.5)

    g = p.add_argument_group("objects")
    g.add_argument("--cube_xy", type=float, nargs=4, default=[0.35, 0.65, -0.25, 0.25], help="xmin xmax ymin ymax")
    g.add_argument("--tape_xy", type=float, nargs=4, default=[0.35, 0.65, -0.25, 0.25])
    g.add_argument("--min_sep", type=float, default=0.10, help="min cube-tape distance [m]")
    g.add_argument("--tape_side", type=float, default=0.07, help="added tape square outer side [m] "
                   "(used only when --tape is not in the scene, e.g. markers drawn at render time)")
    g.add_argument("--tape_width", type=float, default=0.010)
    g.add_argument("--tape_rgba", type=float, nargs=4, default=[0.95, 0.80, 0.10, 1.0])
    g.add_argument("--cube_yaw_deg", type=float, default=0.0, help="+- yaw range; 0 = fixed angle (v1)")

    g = p.add_argument_group("arm posture")
    g.add_argument("--mode_probs", default="random:0.3,home:0.15,hover_cube:0.4,hover_tape:0.15")
    g.add_argument("--hover_h", type=float, nargs=2, default=[0.0, 0.20],
                   help="ee_site height above object top [m]; use a TCP/fingertip site")
    g.add_argument("--hover_xy_std", type=float, default=0.03)
    g.add_argument("--pen_tol", type=float, default=0.002, help="allowed penetration of robot geoms [m]")
    g.add_argument("--hide_robot", action="store_true", help="ablation / upper bound: render without the arm")

    g = p.add_argument_group("occluders and randomization")
    g.add_argument("--n_occ", type=int, default=6, help="occluder slots (box, sphere, cylinder, capsule, bar, ellipsoid)")
    g.add_argument("--p_occ", type=float, default=0.5, help="probability a sample has any occluder")
    g.add_argument("--p_full", type=float, default=0.08, help="probability of fully hiding one object")
    g.add_argument("--no_dr", action="store_true", help="disable all appearance randomization (clean set)")
    g.add_argument("--env_dr", default="mild", choices=["none", "mild", "strong"],
                   help="table/floor/light randomization; the arm is always randomized unless --no_dr")
    g.add_argument("--p_shadow", type=float, default=0.2,
                   help="mild: probability lights cast (faint) shadows; the lab has diffuse ceiling light")
    g.add_argument("--save_depth", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- helpers
def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera looks along its -z axis, +y is image-up."""
    pos, target, up = map(np.asarray, (pos, target, up))
    z = pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((0, 1, 0), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def small_rot_quat(rng, std_rad):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.normal() * std_rad
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def intrinsics(width, height, fovy_deg):
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def project(p_world, cam_pos, cam_xmat, K):
    """World point -> continuous pixel (u, v); pixel i spans [i, i+1). Returns uv, depth."""
    R = cam_xmat.reshape(3, 3)
    pc = R.T @ (np.asarray(p_world) - cam_pos)
    depth = -pc[2]
    u = K[0, 2] + K[0, 0] * pc[0] / depth
    v = K[1, 2] - K[1, 1] * pc[1] / depth
    return np.array([u, v]), depth


def parse_probs(s):
    probs = {}
    for item in s.split(","):
        k, v = item.split(":")
        if k not in MODES:
            raise ValueError(f"unknown posture mode {k}; choose from {MODES}")
        probs[k] = float(v)
    keys = list(probs)
    w = np.array([probs[k] for k in keys])
    return keys, w / w.sum()


def hue_of(rgb):
    return colorsys.rgb_to_hsv(*np.clip(rgb[:3], 0, 1))


def jitter_colour(rng, rgb, dh=0.02, ds=0.15, dv=0.2):
    h, s, v = colorsys.rgb_to_hsv(*np.clip(rgb[:3], 0, 1))
    h = (h + rng.uniform(-dh, dh)) % 1.0
    s = np.clip(s + rng.uniform(-ds, ds), 0, 1)
    v = np.clip(v + rng.uniform(-dv, dv), 0.05, 1)
    return np.array(colorsys.hsv_to_rgb(h, s, v))


def name(m, objtype, i):
    return mujoco.mj_id2name(m, objtype, i) or ""


# --------------------------------------------------------------------------- model
def load_pick_gen(args):
    import importlib
    sys.path.insert(0, os.path.abspath(os.path.expanduser(args.pick_gen)))
    return importlib.import_module(args.scene_module)


def add_tape(spec, args, z):
    """Hollow square of tape as a mocap body, so it can be moved, segmented and labelled."""
    half, w, t = args.tape_side / 2, args.tape_width, 0.0004
    b = spec.worldbody.add_body()
    b.name, b.mocap, b.pos = "vis_tape", True, [0.5, 0.0, z + t]
    for nm, size, pos in (("n", [half, w / 2, t], [0, half - w / 2, 0]), ("s", [half, w / 2, t], [0, -half + w / 2, 0]),
                          ("w", [w / 2, half, t], [-half + w / 2, 0, 0]), ("e", [w / 2, half, t], [half - w / 2, 0, 0])):
        g = b.add_geom()
        g.name, g.type, g.size, g.pos = f"vis_tape_{nm}", mujoco.mjtGeom.mjGEOM_BOX, size, pos
        g.rgba, g.contype, g.conaffinity = list(args.tape_rgba), 0, 0


def build_model(args):
    if args.pick_gen:
        import inspect
        pick_scene = load_pick_gen(args)
        n_par = len(inspect.signature(pick_scene.build_spec).parameters)
        spec = pick_scene.build_spec(*[x for x in (args.menagerie, args.robosuite)][:n_par]) if n_par \
            else pick_scene.build_spec()
    elif args.scene:
        spec = mujoco.MjSpec.from_file(os.path.abspath(args.scene))
    else:
        raise SystemExit("give --scene <xml> or --pick_gen <fr3_pick_gen dir>")
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.width)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.height)

    if spec.body(args.tape) is None and spec.geom(args.tape) is None:
        add_tape(spec, args, args.table_z or 0.0)
        args.tape = "vis_tape"

    cam = spec.worldbody.add_camera()
    cam.name = "vis_cam"
    cam.pos = list(args.cam_pos)
    cam.quat = list(look_at_quat(args.cam_pos, args.cam_lookat))
    cam.fovy = args.fovy or 42.5

    types = [mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER,
             mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID]
    for i in range(args.n_occ):
        b = spec.worldbody.add_body()
        b.name = f"vis_occ{i}"
        b.mocap = True
        b.pos = [0, 0, -10]
        geom = b.add_geom()
        geom.name = f"vis_occ{i}_geom"
        geom.type = types[i % len(types)]
        geom.size = [0.03, 0.03, 0.03]
        geom.contype = 0
        geom.conaffinity = 0
        geom.rgba = [0.5, 0.5, 0.5, 1]
    return spec.compile()


class Obj:
    """Cube or tape: a body (free joint or static) or a bare geom."""

    def __init__(self, m, d, key):
        self.key = key
        self.bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, key)
        self.gid = -1
        self.qadr = -1
        self.mocap = -1
        if self.bid >= 0:
            self.mocap = m.body_mocapid[self.bid]
            sub = subtree(m, self.bid)
            self.geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub]
            for j in range(m.njnt):
                if m.jnt_bodyid[j] == self.bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    self.qadr = m.jnt_qposadr[j]
            if self.qadr >= 0:
                # rest pose from the model (qpos0), not from a keyframe: a keyframe written for the bare arm
                # can leave the free body's qpos at zero, i.e. the cube centre at the table plane
                self.pos0, self.quat0 = m.qpos0[self.qadr:self.qadr + 3].copy(), m.qpos0[self.qadr + 3:self.qadr + 7].copy()
            elif self.mocap >= 0:
                self.pos0, self.quat0 = d.mocap_pos[self.mocap].copy(), d.mocap_quat[self.mocap].copy()
            else:
                self.pos0, self.quat0 = m.body_pos[self.bid].copy(), m.body_quat[self.bid].copy()
        else:
            self.gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, key)
            if self.gid < 0:
                raise SystemExit(f"'{key}' is neither a body nor a geom; run with --list")
            self.geoms = [self.gid]
            self.pos0 = m.geom_pos[self.gid].copy()
            self.quat0 = m.geom_quat[self.gid].copy()
        self.rgba0 = np.array([m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
                               for g in self.geoms])

    def set_pose(self, m, d, x, y, yaw):
        q = quat_mul(yaw_quat(yaw), self.quat0)
        if self.qadr >= 0:
            d.qpos[self.qadr:self.qadr + 3] = [x, y, self.pos0[2]]
            d.qpos[self.qadr + 3:self.qadr + 7] = q
            d.qvel[:] = 0
        elif self.mocap >= 0:
            d.mocap_pos[self.mocap] = [x, y, self.pos0[2]]
            d.mocap_quat[self.mocap] = q
        elif self.bid >= 0:
            m.body_pos[self.bid] = [x, y, self.pos0[2]]
            m.body_quat[self.bid] = q
        else:
            m.geom_pos[self.gid] = [x, y, self.pos0[2]]
            m.geom_quat[self.gid] = q

    def pos(self, d):
        return (d.xpos[self.bid] if self.bid >= 0 else d.geom_xpos[self.gid]).copy()


def subtree(m, root):
    out = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            p = m.body_parentid[p]
            if p == root:
                out.add(b)
                break
    return out


class Scene:
    def __init__(self, args):
        self.a = args
        self.pick = load_pick_gen(args) if args.pick_gen else None
        if args.table_z is None:
            args.table_z = float(getattr(self.pick, "TABLE_TOP_Z", getattr(self.pick, "TABLE_HEIGHT", 0.0))) \
                if self.pick else 0.0
        self.m = m = build_model(args)
        self.d = d = mujoco.MjData(m)
        if args.keyframe:
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
            if k >= 0:
                try:
                    mujoco.mj_resetDataKeyframe(m, d, k)
                except Exception:
                    mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        self.qpos_init = d.qpos.copy()

        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.robot_root)
        if root < 0:
            raise SystemExit(f"robot_root '{args.robot_root}' not found; run with --list")
        self.robot_bodies = subtree(m, root)
        self.robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in self.robot_bodies])

        def joints(rx):
            return [j for j in range(m.njnt) if re.fullmatch(rx, name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
                    and m.jnt_bodyid[j] in self.robot_bodies]
        self.arm_j = joints(args.arm_joint_regex)
        self.finger_j = [j for j in joints(args.finger_joint_regex) if j not in self.arm_j]
        if not self.arm_j:
            raise SystemExit("no arm joints matched --arm_joint_regex; run with --list")
        self.arm_q = m.jnt_qposadr[self.arm_j]
        self.arm_v = m.jnt_dofadr[self.arm_j]
        self.arm_lo, self.arm_hi = m.jnt_range[self.arm_j].T
        self.q_home = d.qpos[self.arm_q].copy()
        if np.allclose(self.q_home, 0) and len(self.arm_q) == 7:   # no usable keyframe: FR3 ready pose
            self.q_home = np.clip([0, -0.3, 0, -2.2, 0, 2.0, 0.79], self.arm_lo, self.arm_hi)
            d.qpos[self.arm_q] = self.q_home
            self.qpos_init = d.qpos.copy()

        # distractors from the pick_gen scene (bottle / can / milk), placed or hidden per sample
        self.distractor_geoms = []
        self.distractor_qadr = []
        try:
            import distractor_rand
            names = distractor_rand.DISTRACTOR_NAMES if self.pick is not None else []
        except ImportError:
            names = []
        if names and all(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint") >= 0 for nm in names):
            self.dist_rand = distractor_rand
            for nm in distractor_rand.DISTRACTOR_NAMES:
                self.distractor_geoms.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{nm}_geom"))
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint")
                self.distractor_qadr.append(m.jnt_qposadr[j])

        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, args.ee_site)
        if self.ee < 0:
            raise SystemExit(f"ee_site '{args.ee_site}' not found; run with --list")
        self.cube = Obj(m, d, args.cube)
        self.tape = Obj(m, d, args.tape)
        self.cube_half_h = self.cube.pos0[2] - args.table_z if self.cube.bid >= 0 else 0.025
        if not 0.005 < self.cube_half_h < 0.2:
            raise SystemExit(f"cube centre {self.cube.pos0[2]:.4f} m is not above table_z {args.table_z} m; "
                             "check --table_z / --cube")

        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam_name) if args.cam_name else -1
        self.cam = cid if cid >= 0 else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "vis_cam")
        if m.cam_bodyid[self.cam] != 0:
            raise SystemExit(f"camera '{args.cam_name}' is not fixed in the world; pick a world camera")
        self.fovy0 = args.fovy or float(m.cam_fovy[self.cam])
        self.cam_pos0 = m.cam_pos[self.cam].copy()
        self.cam_quat0 = m.cam_quat[self.cam].copy()
        self.occ_mocap = [m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"vis_occ{i}")]
                          for i in range(args.n_occ)]
        self.occ_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"vis_occ{i}_geom") for i in range(args.n_occ)]
        self.occ_type0 = m.geom_type[self.occ_geoms].copy()

        # robot colours: drop materials so geom_rgba controls the colour directly
        self.robot_rgba0 = np.array([m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
                                     for g in self.robot_geoms])
        m.geom_matid[self.robot_geoms] = -1
        for o in (self.cube, self.tape):
            m.geom_matid[o.geoms] = -1
        self.table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, args.table_geom)
        self.table_mat = m.geom_matid[self.table] if self.table >= 0 else -1
        self.table_rgba0 = (m.mat_rgba[self.table_mat].copy() if self.table_mat >= 0 else
                            (m.geom_rgba[self.table].copy() if self.table >= 0 else None))
        self.table_refl0 = m.mat_reflectance[self.table_mat] if self.table_mat >= 0 else 0.0
        self.light0 = (m.light_pos.copy(), m.light_diffuse.copy(), m.light_ambient.copy(), m.light_specular.copy())
        self.shadow0 = m.light_castshadow.copy()
        self.head0 = (np.array(m.vis.headlight.diffuse), np.array(m.vis.headlight.ambient))
        self.group0 = m.geom_group.copy()
        self.mat_rgba0 = m.mat_rgba.copy()
        self.mat_refl0 = m.mat_reflectance.copy()
        # pick_gen scene: table / floor material candidates, switched per sample (as in domain_rand.py)
        self.table_geoms, self.table_mats, self.floor_mats = [], [], []
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.pick is not None:
            self.table_geoms = [g for g in range(m.ngeom)
                                if re.fullmatch(r"table_(top|leg_\d+)", name(m, mujoco.mjtObj.mjOBJ_GEOM, g))]
            mid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, nm)
            self.table_mats = [i for i in (mid(f"{x}_mat") for x in getattr(self.pick, "TABLE_MATERIALS", [])) if i >= 0]
            self.floor_mats = [i for i in (mid(f"floor_{x}_mat") for x in getattr(self.pick, "FLOOR_MATERIALS", []))
                               if i >= 0]

        self.seg_lut = np.full(m.ngeom, SEG_OTHER, np.uint8)
        self.seg_lut[self.robot_geoms] = SEG_ROBOT
        self.seg_lut[self.occ_geoms] = SEG_OCC
        self.seg_lut[self.cube.geoms] = SEG_CUBE
        self.seg_lut[self.tape.geoms] = SEG_TAPE

        self.modes, self.mode_p = parse_probs(args.mode_probs)
        self.r_rgb = mujoco.Renderer(m, args.height, args.width)
        self.r_seg = mujoco.Renderer(m, args.height, args.width)
        self.r_seg.enable_segmentation_rendering()
        self.r_dep = None
        if args.save_depth:
            self.r_dep = mujoco.Renderer(m, args.height, args.width)
            self.r_dep.enable_depth_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[HIDE_GROUP] = 0
        self.jac_p = np.zeros((3, m.nv))
        self.jac_r = np.zeros((3, m.nv))

    # ------------------------------------------------------------------ posture
    def robot_collides(self):
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -self.a.pen_tol:
                continue
            if m.geom_bodyid[c.geom1] in self.robot_bodies or m.geom_bodyid[c.geom2] in self.robot_bodies:
                return True
        return False

    def ik(self, target, q0, down):
        m, d = self.m, self.d
        q = q0.copy()
        for _ in range(200):
            d.qpos[self.arm_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = target - d.site_xpos[self.ee]
            mujoco.mj_jacSite(m, d, self.jac_p, self.jac_r, self.ee)
            J = self.jac_p[:, self.arm_v]
            e = err_p
            if down:
                z = d.site_xmat[self.ee].reshape(3, 3)[:, 2]
                err_r = np.cross(z, np.array([0, 0, -1.0]))
                J = np.vstack([J, 0.3 * self.jac_r[:, self.arm_v]])
                e = np.concatenate([err_p, 0.3 * err_r])
            if np.linalg.norm(err_p) < 2e-3 and (not down or np.linalg.norm(e[3:]) < 0.02):
                return q, True
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(len(e)), e)
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, self.arm_lo, self.arm_hi)
        return q, False

    def sample_posture(self, rng, mode, cube_p, tape_p):
        m, d, a = self.m, self.d, self.a
        lo, hi = self.arm_lo, self.arm_hi
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        for _ in range(40):
            if mode == "random":
                q = mid + 0.9 * half * rng.uniform(-1, 1, len(lo))
            elif mode == "home":
                q = np.clip(self.q_home + rng.normal(0, 0.15, len(lo)), lo, hi)
            else:
                obj = cube_p if mode == "hover_cube" else tape_p
                top = obj[2] + (self.cube_half_h if mode == "hover_cube" else 0.0)
                tgt = np.array([obj[0] + rng.normal(0, a.hover_xy_std), obj[1] + rng.normal(0, a.hover_xy_std),
                                top + rng.uniform(*a.hover_h)])
                q0 = np.clip(self.q_home + rng.normal(0, 0.3, len(lo)), lo, hi)
                q, ok = self.ik(tgt, q0, down=rng.random() < 0.7)
                if not ok:
                    continue
            d.qpos[self.arm_q] = q
            for j in self.finger_j:
                d.qpos[m.jnt_qposadr[j]] = rng.uniform(*m.jnt_range[j])
            mujoco.mj_forward(m, d)
            if d.site_xpos[self.ee][2] < a.table_z + 0.01 or self.robot_collides():
                continue
            return mode
        d.qpos[self.arm_q] = self.q_home
        mujoco.mj_forward(m, d)
        return "home"

    # ------------------------------------------------------------------ appearance
    def randomize_appearance(self, rng):
        m, a = self.m, self.a
        m.geom_rgba[self.robot_geoms] = self.robot_rgba0
        m.geom_rgba[self.cube.geoms] = self.cube.rgba0
        m.geom_rgba[self.tape.geoms] = self.tape.rgba0
        m.light_pos[:], m.light_diffuse[:], m.light_ambient[:], m.light_specular[:] = (x.copy() for x in self.light0)
        m.light_castshadow[:] = self.shadow0
        m.vis.headlight.diffuse, m.vis.headlight.ambient = self.head0
        m.cam_pos[self.cam] = self.cam_pos0
        m.cam_quat[self.cam] = self.cam_quat0
        m.cam_fovy[self.cam] = self.fovy0
        m.mat_rgba[:] = self.mat_rgba0
        m.mat_reflectance[:] = self.mat_refl0
        if self.table_mats:
            self.table_mat = self.table_mats[0]
            m.geom_matid[self.table_geoms] = self.table_mat
        if self.floor_mats and self.floor >= 0:
            m.geom_matid[self.floor] = self.floor_mats[0]
        if a.no_dr:
            return
        if self.table_mats:
            self.table_mat = self.table_mats[rng.integers(len(self.table_mats))]
            m.geom_matid[self.table_geoms] = self.table_mat
        if self.floor_mats and self.floor >= 0:
            m.geom_matid[self.floor] = self.floor_mats[rng.integers(len(self.floor_mats))]

        # arm: original 30%, global tint 30%, per-geom random 40%
        r = rng.random()
        base = self.robot_rgba0[:, :3]
        if 0.3 <= r < 0.6:
            col = rng.uniform(0, 1, 3)
            new = (1 - (w := rng.uniform(0.2, 0.8))) * base + w * col
            new *= rng.uniform(0.6, 1.2)
        elif r >= 0.6:
            col = rng.uniform(0, 1, (len(base), 3))
            new = (1 - (w := rng.uniform(0.4, 1.0, (len(base), 1)))) * base + w * col
        else:
            new = base
        m.geom_rgba[self.robot_geoms, :3] = np.clip(new, 0, 1)

        for o in (self.cube, self.tape):
            for k, g in enumerate(o.geoms):
                m.geom_rgba[g, :3] = jitter_colour(rng, o.rgba0[k])

        self.randomize_environment(rng)

        m.cam_pos[self.cam] = self.cam_pos0 + rng.normal(0, a.cam_pos_std, 3)
        m.cam_quat[self.cam] = quat_mul(self.cam_quat0, small_rot_quat(rng, np.deg2rad(a.cam_rot_std)))
        m.cam_fovy[self.cam] = self.fovy0 + rng.normal(0, a.fovy_std)

    def randomize_environment(self, rng):
        """Table, floor and lights. The arm is randomized separately (always on unless --no_dr).

        none   : scene as built
        mild   : matte table (PVC board, no reflection) with small colour/brightness jitter,
                 neutral white lights with brightness, position and slight warm/cool variation
        strong : any table colour, random reflectance, per-channel light colour noise (Tobin-style)
        """
        m, a = self.m, self.a
        lvl = a.env_dr
        if lvl == "none":
            return
        if self.table_mat >= 0:
            c = self.mat_rgba0[self.table_mat, :3]
            if lvl == "mild":
                m.mat_rgba[self.table_mat, :3] = jitter_colour(rng, c, dh=0.02, ds=0.1, dv=0.15)
                m.mat_reflectance[self.table_mat] = 0.0
            else:
                tgt = rng.uniform(0, 1, 3) if rng.random() < 0.4 else np.full(3, rng.uniform(0.2, 0.95))
                m.mat_rgba[self.table_mat, :3] = (1 - (w := rng.uniform(0, 0.7))) * c + w * tgt
                m.mat_reflectance[self.table_mat] = rng.uniform(0.0, 0.35)
        elif self.table >= 0:
            c = m.geom_rgba[self.table, :3].copy()
            m.geom_rgba[self.table, :3] = jitter_colour(rng, c, 0.02, 0.1, 0.15) if lvl == "mild" \
                else rng.uniform(0.2, 0.95, 3)

        shadow = lvl == "strong" or rng.random() < a.p_shadow
        m.light_castshadow[:] = self.shadow0 & shadow
        for i in range(m.nlight):
            m.light_pos[i] = self.light0[0][i] + rng.normal(0, 0.3 if lvl == "mild" else 0.4, 3)
            if lvl == "mild":
                t = rng.uniform(-1, 1)                       # warm (+) / cool (-) white
                tint = np.array([1 + 0.06 * t, 1.0, 1 - 0.06 * t])
                m.light_diffuse[i] = np.clip(self.light0[1][i].mean() * rng.uniform(0.6, 1.3) * tint, 0, 1)
                # with shadows on, more ambient so the shadow stays faint (diffuse room light)
                m.light_ambient[i] = np.full(3, rng.uniform(0.25, 0.4) if shadow else rng.uniform(0.0, 0.15))
                m.light_specular[i] = np.full(3, rng.uniform(0.0, 0.3))
            else:
                m.light_diffuse[i] = np.clip(self.light0[1][i] * rng.uniform(0.5, 1.4) + rng.normal(0, 0.05, 3), 0, 1)
                m.light_ambient[i] = np.clip(rng.uniform(0, 0.2, 3), 0, 1)
                m.light_specular[i] = rng.uniform(0, 0.5, 3)
        m.vis.headlight.diffuse = np.full(3, rng.uniform(0.25, 0.7) if lvl == "mild" else rng.uniform(0.2, 0.8))
        m.vis.headlight.ambient = np.full(3, rng.uniform(0.1, 0.3) if lvl == "mild" else rng.uniform(0.05, 0.4))

    def place_occluders(self, rng, cam_pos, cube_p, tape_p):
        m, d, a = self.m, self.d, self.a
        m.geom_type[self.occ_geoms] = self.occ_type0
        for k in range(a.n_occ):
            d.mocap_pos[self.occ_mocap[k]] = [0, 0, -10]
        if a.n_occ == 0 or rng.random() > a.p_occ:
            return -1
        n_act = min(a.n_occ, 1 + rng.poisson(1.0))
        slots = rng.permutation(a.n_occ)[:n_act]
        forbid = [hue_of(self.cube.rgba0[0]), hue_of(self.tape.rgba0[0])]
        full_target = -1
        if rng.random() < a.p_full:
            full_target = int(rng.integers(0, 2))
        for i, k in enumerate(slots):
            g, mid = self.occ_geoms[k], self.occ_mocap[k]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                size = [rng.uniform(0.01, 0.05), 0, 0]
            elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                size = [rng.uniform(0.008, 0.04), rng.uniform(0.02, 0.12), 0]
            elif k % 6 == 4:  # long bar
                size = list(rng.permutation([rng.uniform(0.06, 0.2), rng.uniform(0.005, 0.02), rng.uniform(0.005, 0.02)]))
            else:
                size = list(rng.uniform(0.01, 0.05, 3))
            # colour far (in hue) from the cube and the tape, unless desaturated
            for _ in range(20):
                col = rng.uniform(0, 1, 3)
                h, s, v = hue_of(col)
                if s < 0.3 or all(min(abs(h - f[0]), 1 - abs(h - f[0])) > 0.08 for f in forbid):
                    break
            m.geom_rgba[g] = [*col, 1]
            obj = cube_p if rng.random() < 0.6 else tape_p
            if i == 0 and full_target >= 0:
                obj = cube_p if full_target == 0 else tape_p
                tt = rng.uniform(0.2, 0.35)
                pos = obj + tt * (cam_pos - obj)
                m.geom_type[g] = mujoco.mjtGeom.mjGEOM_BOX
                size = [0.07, 0.07, 0.07]
                quat = random_quat(rng)
            elif rng.random() < 0.7:  # on the camera-object line
                tt = rng.uniform(0.08, 0.6)
                pos = obj + tt * (cam_pos - obj) + rng.normal(0, 0.03 + max(size), 3)
                quat = random_quat(rng)
            else:  # distractor resting on the table, away from the objects
                for _ in range(20):
                    xy = np.array([rng.uniform(a.cube_xy[0] - 0.1, a.cube_xy[1] + 0.1),
                                   rng.uniform(a.cube_xy[2] - 0.1, a.cube_xy[3] + 0.1)])
                    if min(np.linalg.norm(xy - cube_p[:2]), np.linalg.norm(xy - tape_p[:2])) > 0.08:
                        break
                quat = yaw_quat(rng.uniform(0, 2 * np.pi))
                if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                    hz = size[0]
                elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    hz = size[0] + size[1]
                elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    hz = size[1]
                else:
                    hz = size[2]
                pos = np.array([*xy, a.table_z + hz])
            m.geom_size[g] = size
            d.mocap_pos[mid] = pos
            d.mocap_quat[mid] = quat
        return full_target

    # ------------------------------------------------------------------ render
    def seg_counts(self, seg):
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        lab = np.where(geom >= 0, self.seg_lut[np.clip(geom, 0, None)], SEG_BG).astype(np.uint8)
        return lab

    def isolated_count(self, obj):
        m = self.m
        keep = np.zeros(m.ngeom, bool)
        keep[obj.geoms] = True
        m.geom_group[~keep] = HIDE_GROUP
        self.r_seg.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        lab = self.seg_counts(self.r_seg.render())
        m.geom_group[:] = self.group0
        return int((lab == (SEG_CUBE if obj is self.cube else SEG_TAPE)).sum())

    def sample(self, idx, seed):
        m, d, a = self.m, self.d, self.a
        rng = np.random.default_rng([seed, idx])
        d.qpos[:] = self.qpos_init
        d.qvel[:] = 0
        self.randomize_appearance(rng)

        for _ in range(100):
            c = [rng.uniform(*a.cube_xy[:2]), rng.uniform(*a.cube_xy[2:])]
            t = [rng.uniform(*a.tape_xy[:2]), rng.uniform(*a.tape_xy[2:])]
            if np.hypot(c[0] - t[0], c[1] - t[1]) >= a.min_sep:
                break
        yaw = np.deg2rad(rng.uniform(-a.cube_yaw_deg, a.cube_yaw_deg))
        self.cube.set_pose(m, d, *c, yaw)
        self.tape.set_pose(m, d, *t, 0.0)
        if self.distractor_geoms:
            if rng.random() < a.p_distractor:
                self.dist_rand.randomize_distractors(m, d, rng, np.array([*c, 0]), a.cube_xy[:2], a.cube_xy[2:],
                                                     a.table_z, target_pos=np.array([*t, a.table_z]))
            else:
                for q in self.distractor_qadr:
                    d.qpos[q:q + 3] = [0, 0, -5]
        mujoco.mj_forward(m, d)
        cube_p, tape_p = self.cube.pos(d), self.tape.pos(d)

        mode = self.modes[rng.choice(len(self.modes), p=self.mode_p)]
        mode = self.sample_posture(rng, mode, cube_p, tape_p)

        mujoco.mj_forward(m, d)
        cam_pos = d.cam_xpos[self.cam].copy()
        full = self.place_occluders(rng, cam_pos, cube_p, tape_p)
        mujoco.mj_forward(m, d)
        cam_pos, cam_xmat = d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].copy()
        K = intrinsics(a.width, a.height, m.cam_fovy[self.cam])

        if a.hide_robot:
            m.geom_group[self.robot_geoms] = HIDE_GROUP
        self.r_rgb.update_scene(d, camera=self.cam, scene_option=self.opt)
        rgb = self.r_rgb.render().copy()
        self.r_seg.update_scene(d, camera=self.cam, scene_option=self.opt)
        lab = self.seg_counts(self.r_seg.render())
        depth = None
        if self.r_dep is not None:
            self.r_dep.update_scene(d, camera=self.cam, scene_option=self.opt)
            depth = self.r_dep.render().astype(np.float16)
        m.geom_group[:] = self.group0

        out = {"rgb": rgb, "seg": lab, "depth": depth}
        lbl = {"idx": idx, "mode": MODES.index(mode), "full_occ": full,
               "q": d.qpos[self.arm_q].copy(), "ee_pos": d.site_xpos[self.ee].copy(),
               "cam_pos": cam_pos, "cam_xmat": cam_xmat, "K": K}
        for key, obj, sid in (("cube", self.cube, SEG_CUBE), ("tape", self.tape, SEG_TAPE)):
            p = obj.pos(d)
            uv, dep = project(p, cam_pos, cam_xmat, K)
            n_vis = int((lab == sid).sum())
            n_iso = self.isolated_count(obj)
            lbl[f"{key}_pos"] = p
            lbl[f"{key}_uv"] = uv
            lbl[f"{key}_depth"] = dep
            lbl[f"{key}_npix"] = n_iso
            lbl[f"{key}_vis"] = n_vis / n_iso if n_iso > 0 else 0.0
        lbl["cube_yaw"] = yaw
        lbl["ee_cube_dist"] = float(np.linalg.norm(lbl["ee_pos"] - lbl["cube_pos"]))
        return out, lbl


# --------------------------------------------------------------------------- io
def run_worker(job):
    args, idxs, wid = job
    sc = Scene(args)
    labels = []
    t0 = time.time()
    for n, idx in enumerate(idxs):
        out, lbl = sc.sample(idx, args.seed)
        Image.fromarray(out["rgb"]).save(os.path.join(args.out, "rgb", f"{idx:06d}.png"), compress_level=1)
        Image.fromarray(out["seg"]).save(os.path.join(args.out, "seg", f"{idx:06d}.png"), compress_level=1)
        if out["depth"] is not None:
            np.save(os.path.join(args.out, "depth", f"{idx:06d}.npy"), out["depth"])
        labels.append(lbl)
        if wid == 0 and (n + 1) % 200 == 0:
            rate = (n + 1) / (time.time() - t0)
            print(f"  worker0 {n + 1}/{len(idxs)}  {rate:.1f} img/s/worker", flush=True)
    return labels


def list_names(args):
    m = build_model(args)
    for t, tag in ((mujoco.mjtObj.mjOBJ_BODY, "body"), (mujoco.mjtObj.mjOBJ_JOINT, "joint"),
                   (mujoco.mjtObj.mjOBJ_SITE, "site"), (mujoco.mjtObj.mjOBJ_CAMERA, "camera"),
                   (mujoco.mjtObj.mjOBJ_KEY, "key")):
        n = {"body": m.nbody, "joint": m.njnt, "site": m.nsite, "camera": m.ncam, "key": m.nkey}[tag]
        print(f"{tag}s:", [name(m, t, i) for i in range(n)])
    print("named geoms:", [nm for i in range(m.ngeom) if (nm := name(m, mujoco.mjtObj.mjOBJ_GEOM, i))])


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        list_names(args)
        return
    for sub in ("rgb", "seg") + (("depth",) if args.save_depth else ()):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    idx = np.arange(args.start, args.start + args.n)
    chunks = [c for c in np.array_split(idx, max(1, args.workers)) if len(c)]
    jobs = [(args, c.tolist(), w) for w, c in enumerate(chunks)]
    t0 = time.time()
    if len(jobs) == 1:
        results = [run_worker(jobs[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(run_worker, jobs)
    labels = [l for r in results for l in r]
    labels.sort(key=lambda l: l["idx"])
    arr = {k: np.stack([np.asarray(l[k]) for l in labels]) for k in labels[0]}

    lbl_path = os.path.join(args.out, f"labels_{args.start:06d}.npz")
    np.savez(lbl_path, **arr)
    meta = {"args": vars(args), "modes": MODES, "width": args.width, "height": args.height,
            "seg_ids": {"bg": SEG_BG, "cube": SEG_CUBE, "tape": SEG_TAPE, "robot": SEG_ROBOT,
                        "occluder": SEG_OCC, "other": SEG_OTHER},
            "uv_convention": "continuous pixels, pixel i spans [i, i+1), v down",
            "cube_center_z": float(arr["cube_pos"][:, 2].mean()), "tape_center_z": float(arr["tape_pos"][:, 2].mean())}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    dt = time.time() - t0
    modes = np.bincount(arr["mode"], minlength=len(MODES))
    print(f"wrote {len(labels)} samples to {args.out} in {dt:.0f}s ({len(labels) / dt:.1f} img/s)")
    print("  posture modes:", dict(zip(MODES, modes.tolist())))
    for k in ("cube", "tape"):
        v = arr[f"{k}_vis"]
        print(f"  {k}: visible>0.9 {np.mean(v > 0.9):.2f}  partly {np.mean((v > 0.1) & (v <= 0.9)):.2f}  "
              f"hidden<=0.1 {np.mean(v <= 0.1):.2f}  out-of-frame {np.mean(arr[f'{k}_npix'] == 0):.2f}")


if __name__ == "__main__":
    sys.exit(main())
