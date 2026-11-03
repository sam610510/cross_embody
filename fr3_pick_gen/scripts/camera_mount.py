"""
RealSense D435 external-camera mount for the FR3 MuJoCo scene.

Setup assumed (edit TABLE_* / CAM_* below to match your actual table):
  - World origin at the robot base, robot facing +X across the table.
  - Table surface at world Z = TABLE_HEIGHT, extends from the robot's edge
    (X = ROBOT_EDGE_X) to the far edge (X = ROBOT_EDGE_X + TABLE_LENGTH).
  - Camera is mounted on the far edge, centered on table width (Y = 0),
    at CAM_HEIGHT above the world floor (Z=0), tilted CAM_TILT_DEG downward
    from horizontal, looking back across the table toward the workspace
    center (so it also needs a 180 deg yaw relative to the robot's facing).

Only the numbers in the CONFIG block should need editing for your bench.
"""

import numpy as np

# ----------------------------- CONFIG -----------------------------------
# ----------------------------- CONFIG -----------------------------------
# The robot base (fr3_link0) sits at world Z=0 -- that plane IS the table
# surface / robot mounting plane in this scene, not the physical floor.
# (fr3.xml's own body tree can't easily be re-rooted at a different height,
# so instead the table is built downward from Z=0 in build_scene.py.)
TABLE_LENGTH = 0.80          # m, robot edge -> far edge, EDIT to your table
TABLE_WIDTH = 0.80           # m, EDIT to your table
ROBOT_EDGE_X = 0.0           # m, X coord of the table edge the robot sits at
TABLE_HEIGHT = 0.0           # m, table/robot-mount plane -- keep at 0, do not edit

# EDIT ME: vertical height of the camera. The value below assumes you meant
# height above the TABLE surface (i.e. above TABLE_HEIGHT=0), matching how
# camera mounts are usually specified ("桌邊立柱高90cm"). If you actually
# measured 90cm from the physical floor to the lens, subtract the table's
# real floor-to-tabletop height (often ~0.72-0.76m) before putting a number
# here.
CAM_HEIGHT = 0.90            # m, above the table surface (Z=0)
CAM_TILT_DEG = 60.0          # deg from horizontal (= 30 deg off vertical/nadir)
CAM_YAW_DEG = 180.0          # deg, camera faces back toward the robot (opposite side)

# Fixed workspace/object point -- table center, independent of camera tilt.
# Used for object placement and as the overview-camera aim point. NOT used
# for the D435's own orientation (that comes from CAM_TILT_DEG/CAM_YAW_DEG
# directly, see fixed_tilt_yaw_quat -- a look-at-derived quat would silently
# override whatever tilt you set here to whatever angle the geometry works
# out to, which is exactly what caused the 45->~66deg mismatch).
LOOK_AT = np.array([ROBOT_EDGE_X + TABLE_LENGTH * 0.5, 0.0, TABLE_HEIGHT])

# D435 RGB stream: ~69 deg HFOV / ~42 deg VFOV at 16:9. MuJoCo <camera fovy>
# is the *vertical* FOV in degrees.
D435_FOVY_DEG = 42.0
D435_RESOLUTION = (640, 480)   # (width, height) passed to the renderer
# --------------------------------------------------------------------------


def look_at_quat(eye: np.ndarray, target: np.ndarray, world_up=(0, 0, 1)) -> np.ndarray:
    """
    Build a MuJoCo camera quaternion (w, x, y, z) for a camera at `eye`
    looking at `target`.

    MuJoCo camera convention: the camera looks down its local -Z axis,
    and local +Y is "up" in the rendered image, local +X is "right".
    """
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)

    up = np.asarray(world_up, dtype=float)

    z_cam = -forward  # camera looks down -Z
    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-8:
        # forward nearly parallel to world_up; pick an arbitrary perpendicular
        x_cam = np.cross(np.array([1.0, 0.0, 0.0]), z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    rot = np.stack([x_cam, y_cam, z_cam], axis=1)  # columns = cam axes in world frame

    return _mat_to_quat(rot)


def fixed_tilt_yaw_quat(tilt_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Alternative to look_at_quat: build the quaternion directly from the
    stated height/tilt spec (45 deg downward tilt, facing back across the
    table) instead of computing a look-at target. Useful if you'd rather
    hard-set the mount angle than depend on TABLE_* geometry being exact.
    """
    tilt = np.radians(tilt_deg)
    yaw = np.radians(yaw_deg)

    # Start pointing camera -Z along world -X (i.e. yaw=0 looks toward -X),
    # then yaw about world Z, then pitch down about the camera's local X.
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1],
    ])
    # base orientation: -Z forward = world -X, +Y up = world +Z, +X right = world -Y
    R_base = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ])  # columns = cam axes (x,y,z) expressed in world frame -- NOT transposed

    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt),  np.sin(tilt)],
        [0, -np.sin(tilt), np.cos(tilt)],
    ])  # pitch down about local X axis

    R = R_yaw @ R_base @ R_tilt
    return _mat_to_quat(R)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def camera_world_pos() -> np.ndarray:
    return np.array([ROBOT_EDGE_X + TABLE_LENGTH, 0.0, CAM_HEIGHT])


def camera_table_intersection(tilt_deg: float = CAM_TILT_DEG) -> np.ndarray:
    """
    Where the D435's actual viewing axis (at the given tilt) hits the table
    plane (Z=TABLE_HEIGHT) -- i.e. what the camera is really centered on.
    Useful for sanity-checking a tilt choice: if this lands outside the
    table's X range [ROBOT_EDGE_X, ROBOT_EDGE_X+TABLE_LENGTH], the camera's
    center of view is off the table (a shallower tilt look further out).
    """
    cam_pos = camera_world_pos()
    tilt = np.radians(tilt_deg)
    if tilt <= 0:
        raise ValueError("tilt_deg must be > 0 (camera must point at least "
                          "somewhat downward to ever hit the table plane)")
    horizontal_reach = (cam_pos[2] - TABLE_HEIGHT) / np.tan(tilt)
    return np.array([cam_pos[0] - horizontal_reach, cam_pos[1], TABLE_HEIGHT])


def mjcf_camera_snippet(name: str = "d435") -> str:
    """
    Print/return an MJCF <camera> element you can paste into a <body> at
    the world root of your scene XML (add it as a static camera, it does
    not need to be attached to any moving body).
    """
    pos = camera_world_pos()
    # Explicit tilt/yaw, NOT look_at_quat -- this is the one that actually
    # honors CAM_TILT_DEG as you set it (see the note above LOOK_AT).
    quat = fixed_tilt_yaw_quat(CAM_TILT_DEG, CAM_YAW_DEG)
    return (
        f'<camera name="{name}" mode="fixed" '
        f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
        f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
        f'fovy="{D435_FOVY_DEG}"/>'
    )


if __name__ == "__main__":
    print("Camera world position:", camera_world_pos())
    print(f"At {CAM_TILT_DEG} deg tilt, camera axis hits the table at:",
          camera_table_intersection())
    print("(table X range is", ROBOT_EDGE_X, "to", ROBOT_EDGE_X + TABLE_LENGTH,
          "-- if the point above falls outside that, the camera is looking "
          "past the table, not too shallow to make sense but worth knowing)")
    print()
    print("MJCF snippet:")
    print(mjcf_camera_snippet())