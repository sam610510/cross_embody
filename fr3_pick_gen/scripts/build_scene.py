"""
Builds the FR3 + Franka Hand + D435i + table + object scene entirely in
Python via mujoco.MjSpec, instead of hand-writing a merged MJCF file.

Why MjSpec.attach() instead of <include>/<attach> in raw XML:
  - fr3.xml ends at a bare <site name="attachment_site"/> with no gripper;
    the hand has to be composed onto it.
  - hand.xml and d435i.xml are each a full standalone <mujoco> document
    (their own <asset>, <default>, <worldbody> root) -- XML <include> can't
    drop a whole such document inside a <body> in a different file, but
    MjSpec.attach() is the officially supported way to compose exactly
    this (arm + gripper + sensor) from separate menagerie models.
  - Verified against this environment's installed mujoco: attach() returns
    an MjsFrame; joints/actuators/tendons from the attached model are
    automatically renamed with the given prefix (checked: hand.xml's
    "actuator8" -> "hand_actuator8", "finger_joint1" -> "hand_finger_joint1").

Run this file directly to write out the assembled scene as XML (for
inspection / version control) to OUT_XML_PATH. generate_fr3_expert_data.py
imports build_model() directly instead of loading that XML back in, so the
XML file is a debug artifact, not the source of truth.
"""

import os
from pathlib import Path

import numpy as np
import mujoco

from camera_mount import (camera_world_pos, fixed_tilt_yaw_quat, LOOK_AT, D435_FOVY_DEG,
                           CAM_TILT_DEG, CAM_YAW_DEG,
                           TABLE_LENGTH, TABLE_WIDTH, ROBOT_EDGE_X)

# ============================== CONFIG ====================================

# EDIT ME: your mujoco_menagerie clone location.
MENAGERIE_DIR = os.path.expanduser("~/cross_embody/mujoco_menagerie")
FR3_XML = f"{MENAGERIE_DIR}/franka_fr3/fr3.xml"
HAND_XML = f"{MENAGERIE_DIR}/franka_emika_panda/hand.xml"
D435I_XML = f"{MENAGERIE_DIR}/realsense_d435i/d435i.xml"

OUT_XML_PATH = Path(os.path.expanduser(
    "~/cross_embody/fr3_pick_gen/assets/fr3_pick_scene.xml"))

# Real Franka Hand: hand frame -> fingertip contact point is ~0.1034m along
# local +Z (standard Franka TCP offset). Only used for the tcp_site below.
TCP_OFFSET = 0.1034

# Table: derived from camera_mount.py's TABLE_LENGTH/TABLE_WIDTH/ROBOT_EDGE_X
# so the two files can't disagree about where the table actually is. The
# robot base (at X=ROBOT_EDGE_X, Y=0, Z=0) must sit INSIDE the table
# footprint, not at its lip, so the table extends ROBOT_BASE_MARGIN behind
# the robot's mounting edge as well as forward across TABLE_LENGTH.
ROBOT_BASE_MARGIN = 0.10       # m, table clearance behind the robot base
TABLE_TOP_Z = 0.0
TABLE_THICKNESS = 0.75          # m, how far the table body extends below Z=0

_table_x_min = ROBOT_EDGE_X - ROBOT_BASE_MARGIN
_table_x_max = ROBOT_EDGE_X + TABLE_LENGTH
TABLE_HALF_SIZE = np.array([(_table_x_max - _table_x_min) / 2,
                            TABLE_WIDTH / 2,
                            TABLE_THICKNESS / 2])
TABLE_POS = np.array([(_table_x_max + _table_x_min) / 2, 0.0,
                       TABLE_TOP_Z - TABLE_HALF_SIZE[2]])
FLOOR_Z = TABLE_POS[2] - TABLE_HALF_SIZE[2]  # m, floor level = table's bottom (legs stand here)

# Fixed object: 5cm cube resting on the table surface, centered under the
# camera's look-at point (matches the Milestone-1 pick target).
OBJECT_HALF_SIZE = 0.025
OBJECT_POS = np.array([LOOK_AT[0], LOOK_AT[1], TABLE_TOP_Z + OBJECT_HALF_SIZE])
OBJECT_RGBA = [0.1, 0.8, 0.1, 1.0]
OBJECT_MASS = 0.05

# ============================================================================


def build_spec() -> mujoco.MjSpec:
    fr3 = mujoco.MjSpec.from_file(FR3_XML)
    hand = mujoco.MjSpec.from_file(HAND_XML)
    d435i = mujoco.MjSpec.from_file(D435I_XML)
    for light in d435i.lights:
        light.active = 0  # d435i.xml ships its own preview light -- disable it,
                           # or it becomes a second, uncontrolled light source
                           # once attached (this was causing the double-shadow).

    wb = fr3.worldbody

    # --- attach gripper at the arm's attachment_site (no IK, just a rigid weld) ---
    site = fr3.site("attachment_site")
    fr3.attach(hand, prefix="hand_", site=site)

    # TCP (tool-center-point) site at the fingertip midpoint, used only by
    # solve_fixed_poses.py's one-time offline IK -- never referenced by the
    # actual data-generation loop, which stays pure joint-space playback.
    hand_body = fr3.body("hand_hand")
    hand_body.add_site(name="tcp_site", pos=[0, 0, TCP_OFFSET])

    # --- optional finger-pad contact model, chosen by the environment variable FR3_PAD ---
    #   default : menagerie hand.xml as is (friction 1, soft default contact, no priority)
    #   robotiq : the SAME pad contact as the Robotiq 2F-85 of the UR5e (friction 0.7, solref 0.004,
    #             solimp 0.95/0.99, priority 1) -> both grippers differ only in geometry and force
    #   high    : robotiq stiffness + friction 1.5 and torsional friction 0.02 (condim 4)
    pad = os.environ.get("FR3_PAD", "default")
    if pad != "default":
        n_pad = 0
        for bn in ("hand_left_finger", "hand_right_finger"):
            for g in fr3.body(bn).find_all(mujoco.mjtObj.mjOBJ_GEOM):
                if g.type != mujoco.mjtGeom.mjGEOM_BOX:          # the 5 fingertip pad boxes per finger
                    continue
                g.priority = 1
                g.solref = [0.004, 1.0]
                g.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
                if pad == "robotiq":
                    g.friction = [0.7, 0.005, 0.0001]
                elif pad == "high":
                    g.friction = [1.5, 0.02, 0.0001]
                    g.condim = 4
                else:
                    raise SystemExit(f"FR3_PAD={pad}: use default / robotiq / high")
                n_pad += 1
        print(f"[build_scene] FR3_PAD={pad}: {n_pad} finger pad geoms changed")

    # --- static D435i visual mesh + camera sensor, both at the computed mount pose ---
    cam_pos = camera_world_pos()
    # Explicit tilt/yaw (not look-at): honors CAM_TILT_DEG exactly as set,
    # rather than deriving whatever angle the table geometry happens to imply.
    cam_quat = fixed_tilt_yaw_quat(CAM_TILT_DEG, CAM_YAW_DEG)
    cam_frame = wb.add_frame(pos=cam_pos.tolist(), quat=cam_quat.tolist())
    fr3.attach(d435i, prefix="cam_", frame=cam_frame)
    wb.add_camera(name="d435", pos=cam_pos.tolist(), quat=cam_quat.tolist(),
                   fovy=D435_FOVY_DEG)

    # --- table (static, wood-look, matte finish) ---
    wood_mat = fr3.add_material(
        name="wood_table", rgba=[0.65, 0.50, 0.32, 1.0],
        specular=0.1, shininess=0.1, reflectance=0.02,
    )
    table = wb.add_body(name="table", pos=TABLE_POS.tolist())
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=TABLE_HALF_SIZE.tolist(), material="wood_table")

    # --- floor: cream/beige ceramic tile (checker pattern, matte) ---
    floor_tex = fr3.add_texture(
        name="floor_tile", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.93, 0.88, 0.78], rgb2=[0.85, 0.79, 0.66],
        mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.6, 0.55, 0.45],
        width=300, height=300,
    )
    floor_mat = fr3.add_material(
        name="floor_tile_mat", textures=["", "floor_tile"],
        texuniform=True, texrepeat=[8, 8], reflectance=0.03,
    )
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[TABLE_POS[0], 0.0, FLOOR_Z], size=[3.0, 3.0, 0.02],
                material="floor_tile_mat")

    # --- object (free body, the pick target) ---
    obj = wb.add_body(name="object", pos=OBJECT_POS.tolist())
    obj.add_freejoint(name="object_freejoint")
    obj.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[OBJECT_HALF_SIZE] * 3, rgba=OBJECT_RGBA, mass=OBJECT_MASS, condim=4)

    # --- lighting (randomized per-episode by domain_rand.py) ---
    # MuJoCo's default headlight is on unless disabled -- combined with the
    # key_light below that's TWO light sources (hence two shadows). Turn it
    # off so only our own controllable light remains.
    fr3.visual.headlight.active = 0
    wb.add_light(name="key_light", pos=[0.5, 0.0, 2.0], dir=[0, 0, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    return fr3


def build_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    spec = build_spec()
    model = spec.compile()

    OUT_XML_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_XML_PATH.write_text(spec.to_xml())
    print(f"Wrote assembled scene to {OUT_XML_PATH}")

    print(f"nq={model.nq} nu={model.nu} nbody={model.nbody} ncam={model.ncam}")
    print("\nJoints:")
    for i in range(model.njnt):
        print(" ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    print("\nActuators:")
    for i in range(model.nu):
        print(" ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
    print("\nCameras:")
    for i in range(model.ncam):
        print(" ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i))

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        print("\nfr3.xml 'home' keyframe qpos (padded with 0s for the added "
              "gripper DOFs):", model.key_qpos[key_id])