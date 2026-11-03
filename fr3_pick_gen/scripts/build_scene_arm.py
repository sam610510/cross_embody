"""
The FR3 pick scene (build_scene.py: table, floor, cube, lights, d435 mount) with ANOTHER ARM instead of the FR3 --
for the cross-arm (zero-shot) evaluation. Everything except the arm is identical to the FR3 scene, so the image
encoder sees the same table / cube / tape / cameras and only the arm differs.

The arm is the robots.yaml entry named by the environment variable NEW_ARM (e.g. ur5e_menagerie), built exactly as
the FK encoder builds it (fk_encoder.build_mjcf_robot_spec): arm MJCF + gripper + TCP body. Nothing arm-specific is
written here:
  - the arm hangs under one static body 'arm_mount' at the FR3 base position (world origin); all its names get the
    prefix 'arm_' (gen_vision_data.Scene: --robot_root arm_mount);
  - a site 'arm_tcp_site' is added on the TCP body (<prefix><gripper prefix>tcp);
  - arm + gripper geoms collide with the table and the cube, not with each other;
  - the FR3 (+ hand) stays in the model (its keyframe / actuators), but is moved 5 m under the floor, invisible and
    without collisions.

Use with gen_vision_data / eval_pp_vision.py:  --scene_module build_scene_arm   (eval_pp_vision.py --robot <entry>
sets this up itself)
"""
import os
import sys

import mujoco
import yaml

from build_scene import *                       # noqa: F401,F403  (OBJECT_POS, TABLE_TOP_Z, ... re-exported)
from build_scene import build_spec as _build_fr3_spec

R = os.path.expanduser("~/cross_embody")
for _p in (R, os.path.join(R, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FR3_ROOT = "base"                                          # root body of the FR3 in fr3.xml
PREFIX = "arm_"
MOUNT = "arm_mount"
TCP_SITE = "arm_tcp_site"


def robot_name():
    n = os.environ.get("NEW_ARM", "")
    if not n:
        raise SystemExit("[build_scene_arm] set NEW_ARM=<robots.yaml entry>")
    return n


def robot_cfg(name=None):
    import train_stage2_align as T
    return yaml.safe_load(open(T.ROBOTS_YAML))[name or robot_name()]


def gripper_prefix(cfg):
    return (cfg.get("gripper") or {}).get("prefix", "grip_")


def build_spec() -> mujoco.MjSpec:
    from fk_encoder.fk_encoder import build_mjcf_robot_spec
    cfg = robot_cfg()
    spec = _build_fr3_spec()

    # FR3: out of the way, invisible, no contacts
    fr3 = spec.body(FR3_ROOT)
    fr3.pos = [0.0, 0.0, -5.0]
    for g in fr3.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        g.rgba = [g.rgba[0], g.rgba[1], g.rgba[2], 0.0]
        g.material = ""
        g.contype = 0
        g.conaffinity = 0
        g.group = 5

    # the new arm, as the FK encoder builds it, under one static mount body at the FR3 base position
    robot = build_mjcf_robot_spec(cfg, R)
    mount = spec.worldbody.add_body(name=MOUNT, pos=[0.0, 0.0, 0.0])
    spec.attach(robot, prefix=PREFIX, frame=mount.add_frame(pos=[0.0, 0.0, 0.0]))
    for g in mount.find_all(mujoco.mjtObj.mjOBJ_GEOM):   # vs table / cube, not vs itself
        if g.contype or g.conaffinity:
            g.contype = 2
            g.conaffinity = 1

    tcp_name = PREFIX + gripper_prefix(cfg) + "tcp"
    tcp = spec.body(tcp_name)
    if tcp is None:
        raise SystemExit(f"[build_scene_arm] TCP body '{tcp_name}' not found")
    tcp.add_site(name=TCP_SITE, pos=[0.0, 0.0, 0.0], size=[0.004, 0.004, 0.004])
    return spec


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    m = build_model()
    print(f"[{robot_name()}] nq={m.nq} nu={m.nu} nbody={m.nbody}")
    for i in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith(PREFIX):
            print("  joint", n)
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if n.startswith(PREFIX):
            print("  actuator", n)