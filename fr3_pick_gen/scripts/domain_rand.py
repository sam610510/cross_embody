"""
Per-episode domain randomization: gravity, friction, lighting.
Call apply_domain_randomization(model) once at the start of each episode,
before mj_forward / the first control step.
"""

import numpy as np
import mujoco

# ----------------------------- CONFIG ------------------------------------
GRAVITY_Z_RANGE = (-9.91, -9.71)          # m/s^2, nominal -9.81
FRICTION_SLIDE_RANGE = (0.4, 1.2)          # tangential friction coefficient
FRICTION_TORSION_SCALE_RANGE = (0.8, 1.2)  # multiplier on the model's default torsional friction
FRICTION_ROLL_SCALE_RANGE = (0.8, 1.2)     # multiplier on the model's default rolling friction
LIGHT_DIFFUSE_RANGE = (0.10, 0.65)   # was (0.3, 1.0) -- dimmer overall, wider spread
LIGHT_AMBIENT_RANGE = (0.05, 0.40)   # was (0.1, 0.5)
# Simulates the sun/window light changing angle through the day: jitter the
# key light's direction and position around its nominal mount.
LIGHT_DIR_JITTER_DEG = 30.0   # was 25.0 -- bigger angle swing per episode
LIGHT_POS_JITTER_M = 0.6      # was 0.4

# EDIT ME: names of geoms whose friction should be randomized (object + table
# top -- these are the two surfaces that actually matter for a pick task;
# randomizing the arm/gripper geoms too is usually unnecessary noise).
RANDOMIZED_FRICTION_GEOMS = ["object_geom", "table_top"]
# ---------------------------------------------------------------------------


def randomize_gravity(model):
    g = np.random.uniform(*GRAVITY_Z_RANGE)
    model.opt.gravity[:] = [0.0, 0.0, g]


def randomize_friction(model, geom_names=RANDOMIZED_FRICTION_GEOMS):
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        slide = np.random.uniform(*FRICTION_SLIDE_RANGE)
        torsion_scale = np.random.uniform(*FRICTION_TORSION_SCALE_RANGE)
        roll_scale = np.random.uniform(*FRICTION_ROLL_SCALE_RANGE)
        base_torsion = model.geom_friction[gid, 1]
        base_roll = model.geom_friction[gid, 2]
        model.geom_friction[gid, 0] = slide
        model.geom_friction[gid, 1] = base_torsion * torsion_scale
        model.geom_friction[gid, 2] = base_roll * roll_scale


def randomize_lighting(model):
    for lid in range(model.nlight):
        diffuse = np.random.uniform(*LIGHT_DIFFUSE_RANGE)
        ambient = np.random.uniform(*LIGHT_AMBIENT_RANGE)
        model.light_diffuse[lid] = [diffuse, diffuse, diffuse]
        model.light_ambient[lid] = [ambient, ambient, ambient]


_nominal_light_cache = {}  # id(model) -> (light_pos0, light_dir0), so repeated
                            # randomization jitters around the *original* mount
                            # instead of random-walking from the last draw.


def randomize_light_pose(model):
    """
    Jitters each light's position/direction around its nominal mount, to
    simulate outdoor/window light changing angle through the day. Always
    perturbs from the model's ORIGINAL light pose (cached on first call),
    not from whatever the previous episode randomized it to.
    """
    key = id(model)
    if key not in _nominal_light_cache:
        _nominal_light_cache[key] = (model.light_pos.copy(), model.light_dir.copy())
    pos0, dir0 = _nominal_light_cache[key]

    for lid in range(model.nlight):
        jitter_xy = np.random.uniform(-LIGHT_POS_JITTER_M, LIGHT_POS_JITTER_M, size=2)
        model.light_pos[lid, 0] = pos0[lid, 0] + jitter_xy[0]
        model.light_pos[lid, 1] = pos0[lid, 1] + jitter_xy[1]
        model.light_pos[lid, 2] = pos0[lid, 2]

        d = dir0[lid].copy()
        # tilt the light off straight-down by a random azimuth + a random
        # horizontal offset (a purely vertical dir=[0,0,-1] is unaffected by
        # a pure Z-rotation, so this also perturbs the tilt, not just azimuth)
        theta = np.random.uniform(0, 2 * np.pi)
        tilt = np.radians(np.random.uniform(0, LIGHT_DIR_JITTER_DEG))
        horiz = np.sin(tilt) * np.array([np.cos(theta), np.sin(theta), 0.0])
        d_new = d * np.cos(tilt) + horiz * np.linalg.norm(d)
        model.light_dir[lid] = d_new / np.linalg.norm(d_new)


def apply_domain_randomization(model):
    """Call once per episode, before the first mj_forward/mj_step."""
    randomize_gravity(model)
    randomize_friction(model)
    randomize_lighting(model)
    randomize_light_pose(model)