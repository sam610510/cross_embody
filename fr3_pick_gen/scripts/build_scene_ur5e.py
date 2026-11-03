"""
The FR3 pick scene (build_scene.py: table, floor, cube, lights, d435 mount) with the UR5e + Robotiq 2F-85 instead of
the FR3 -- for the cross-arm (zero-shot) vision evaluation. Everything except the arm is IDENTICAL to the FR3 scene,
so the image encoder sees the same table / cube / tape / cameras and only the arm differs.

  - the UR5e + 2F-85 is built exactly as the FK encoder / posture dictionary build it
    (robots.yaml entry, fk_encoder.build_mjcf_robot_spec), as in scripts/eval_pp_ur5e.py;
  - it hangs under one static body 'ur_mount' at the FR3 base position (world origin), so
    gen_vision_data.Scene can use --robot_root ur_mount;
  - a site 'ur_tcp_site' is added on the TCP body (Scene needs an ee site);
  - the FR3 (+ hand) stays in the model (its keyframe / actuators), but is moved 5 m under the floor,
    invisible and without collisions.

Use with gen_vision_data / eval_pp_vision.py:  --scene_module build_scene_ur5e
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

ROBOT = os.environ.get("UR_ROBOT", "ur5e_menagerie")      # robots.yaml entry
FR3_ROOT = "base"                                          # root body of the FR3 in fr3.xml
UR_PREFIX = "ur_"


def _subtree_geoms(body):
    return body.find_all(mujoco.mjtObj.mjOBJ_GEOM)


def build_spec() -> mujoco.MjSpec:
    import train_stage2_align as T
    from fk_encoder.fk_encoder import build_mjcf_robot_spec
    cfg = yaml.safe_load(open(T.ROBOTS_YAML))[ROBOT]
    spec = _build_fr3_spec()

    # FR3: out of the way, invisible, no contacts
    fr3 = spec.body(FR3_ROOT)
    fr3.pos = [0.0, 0.0, -5.0]
    for g in _subtree_geoms(fr3):
        g.rgba = [g.rgba[0], g.rgba[1], g.rgba[2], 0.0]
        g.material = ""
        g.contype = 0
        g.conaffinity = 0
        g.group = 5                                # not in the default render groups either

    # UR5e + 2F-85, as the FK encoder builds it, under one static mount body at the FR3 base position
    robot = build_mjcf_robot_spec(cfg, R)
    mount = spec.worldbody.add_body(name="ur_mount", pos=[0.0, 0.0, 0.0])
    spec.attach(robot, prefix=UR_PREFIX, frame=mount.add_frame(pos=[0.0, 0.0, 0.0]))

    # robot geoms (arm + gripper): collide with the table and the cube, not with each other (as eval_pp_ur5e.py)
    for g in _subtree_geoms(mount):
        if g.contype or g.conaffinity:
            g.contype = 2
            g.conaffinity = 1

    # ee site on the TCP body (same body the FK encoder calls the TCP)
    tcp_name = UR_PREFIX + (cfg.get("gripper") or {}).get("prefix", "grip_") + "tcp"
    tcp = spec.body(tcp_name)
    if tcp is None:
        raise SystemExit(f"[build_scene_ur5e] TCP body '{tcp_name}' not found")
    tcp.add_site(name="ur_tcp_site", pos=[0.0, 0.0, 0.0], size=[0.004, 0.004, 0.004])
    return spec


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    m = build_model()
    print(f"nq={m.nq} nu={m.nu} nbody={m.nbody} ncam={m.ncam}")
    for i in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith(UR_PREFIX):
            print("  joint", n)
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if n.startswith(UR_PREFIX):
            print("  actuator", n)
