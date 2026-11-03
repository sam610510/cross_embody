"""
Expert demonstration generator for FR3 + gripper in MuJoCo.

Design (per your spec):
  - NO inverse kinematics anywhere in this script. Every waypoint (home,
    reach target) is a fixed, hand-picked JOINT configuration. Only
    the *initial* configuration is randomized (also in joint space).
    Segments between waypoints are joint-space quintic (minimum-jerk)
    interpolations, so there is never an IK multi-solution problem and
    the arm never takes a weird detour to satisfy a Cartesian target.
  - Reach task: random init -> fixed TARGET_Q, then HOLD there (does NOT
    return to HOME_Q -- the task is "go to the target and stay put", so
    FINAL-step distance-to-target is now a fair, directly meaningful
    metric). No grasp, gripper stays open the whole episode (Milestone 1:
    goal-conditioned reaching).
  - Camera: static RealSense D435 mounted opposite the arm (see
    camera_mount.py) -> recorded to mp4 per episode.
  - Per-step logging: joint positions/velocities, gripper state, action,
    timestamps -> written in a LeRobot-v2-style dataset layout
    (data/chunk-000/episode_XXXXXX.parquet + videos/chunk-000/<cam>/episode_XXXXXX.mp4
    + meta/{info.json,episodes.jsonl,tasks.jsonl}).

TARGET_Q is filled in below from solve_fixed_poses.py's PREGRASP_Q output
(a hover pose above the fixed object -- used here purely as a reach target,
nothing is grasped).
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
import pandas as pd

from camera_mount import D435_RESOLUTION
from build_scene import build_model, OBJECT_POS
from domain_rand import apply_domain_randomization

# ============================== CONFIG ====================================

# Scene is built programmatically (see build_scene.py) via MjSpec.attach(),
# not loaded from a static XML file -- no MODEL_XML_PATH needed.

CAMERA_NAME = "d435"
RENDER_W, RENDER_H = D435_RESOLUTION

N_ARM_JOINTS = 7
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
# hand.xml's tendon actuator "actuator8" is renamed "hand_actuator8" by the
# attach(prefix="hand_", ...) call in build_scene.py.
GRIPPER_ACT_NAME = "hand_actuator8"

# fr3.xml's own built-in <keyframe name="home"> -- the single shared neutral
# pose used at the start/end of every episode.
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# TARGET_Q: fixed reach target (hover pose above the object), from
# solve_fixed_poses.py's printed PREGRASP_Q. Does NOT change per episode.
TARGET_Q = np.array([-0.000000, -0.016620, 0.000000, -2.729232, 0.000000,
                      2.712608, 0.785398])

# hand_actuator8 takes a 0-255 command (remapped from the real Franka gripper
# driver's protocol): gripper just stays open the whole episode (reach task,
# nothing grasped).
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

# Random-initialization sampling range, as a fraction of each joint's
# [lower, upper] limit range, centered on HOME_Q, so random starts stay
# "reasonable" instead of spanning the full physical joint range. The table
# sits flush with the arm's own mounting plane and covers most of the
# reachable workspace, so keep this modest -- 0.5 was rejecting almost every
# draw (arm dipping into the table) and silently falling back to HOME_Q
# every episode, which is why every episode looked identical.
INIT_SAMPLE_FRACTION = 0.15   # 0 = always HOME_Q, 1 = full joint range
N_COLLISION_CHECK_TRIES = 200

CONTROL_HZ = 30           # matches D435 stream rate; also the logging/video rate
SIM_HZ = 500              # physics substeps per second (set to your model's timestep)
SEGMENT_DURATION_S = {    # quintic segment durations, tune to taste
    "init_to_target": 2.0,
    "hold_at_target": 2.0,   # stay at target_q (not return HOME_Q). Without
                              # this, the arm only ever PASSES THROUGH the
                              # target at speed (quintic hits zero velocity
                              # there for an instant, then the old design
                              # immediately left again) -- the model never
                              # saw "arrived and staying put" as a thing to
                              # do, so at inference there was no training
                              # signal telling it to settle rather than keep
                              # moving. This segment holds target_q constant,
                              # giving the model real "stay here" examples.
}

N_EPISODES = 300
OUT_DIR = Path(os.path.expanduser("~/cross_embody/fr3_pick_gen/datasets/expert_v1"))
TASK_DESCRIPTION = "Reach a fixed target point with the FR3"

# ============================================================================


def quintic_interp(q0: np.ndarray, q1: np.ndarray, duration_s: float, hz: float) -> np.ndarray:
    """
    Minimum-jerk (quintic) joint-space interpolation from q0 to q1.
    Zero velocity/acceleration at both endpoints -> smooth, no IK involved.
    Returns an array of shape (n_steps, len(q0)).
    """
    n_steps = max(1, int(round(duration_s * hz)))
    t = np.linspace(0.0, 1.0, n_steps)
    # s(t) = 10t^3 - 15t^4 + 6t^5
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    return q0[None, :] + s[:, None] * (q1 - q0)[None, :]


def sample_random_init(model, data, home_q: np.ndarray, jnt_ids: list,
                        fraction: float, n_tries: int) -> np.ndarray:
    """
    Sample a random initial arm configuration in joint space, centered on
    home_q, scaled by each joint's own range, rejecting configs that put
    the ARM/GRIPPER into self- or table-collision. The object resting on
    the table is a permanent, expected contact (table_top <-> object_geom)
    and must NOT count as a rejection, or every single draw gets rejected
    and sample_random_init always falls back to home_q -- silently making
    every episode's "random" start identical.
    """
    lo = model.jnt_range[jnt_ids, 0]
    hi = model.jnt_range[jnt_ids, 1]
    half_span = fraction * (hi - lo) / 2.0

    def bad_contacts(d):
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if {g1, g2} == {"table_top", "object_geom"}:
                continue  # object resting on the table -- always present, not a rejection
            out.append((g1, g2))
        return out

    for _ in range(n_tries):
        candidate = home_q + np.random.uniform(-half_span, half_span)
        candidate = np.clip(candidate, lo, hi)

        data.qpos[jnt_ids] = candidate
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        if not bad_contacts(data):
            return candidate

    print(f"WARNING: no collision-free random init found in {n_tries} tries "
          f"(fraction={fraction}); falling back to home_q. Last try's contacts:")
    for g1, g2 in bad_contacts(data):
        print(f"  {g1} <-> {g2}")
    return home_q.copy()


@dataclass
class EpisodeLog:
    qpos: list = field(default_factory=list)      # arm(7) + gripper(1)
    qvel: list = field(default_factory=list)
    action: list = field(default_factory=list)     # commanded target, same shape as qpos
    phase: list = field(default_factory=list)      # str label per step, for debugging/filtering
    timestamp: list = field(default_factory=list)
    goal_xyz: list = field(default_factory=list)   # 3d world target, repeated per row --
                                                     # same value every row within one episode,
                                                     # but MUST vary across episodes for a
                                                     # goal-conditioning experiment to mean anything


def build_waypoint_sequence(init_q: np.ndarray, target_q: np.ndarray) -> list:
    """
    Returns an ordered list of (segment_name, arm_target, gripper_target, duration_s)
    describing the whole fixed trajectory shape: init -> target_q, then HOLD
    at target_q (does NOT return to HOME_Q -- the task is "go to the target
    and stay", not "visit the target on your way back home"). Every
    arm_target here is a FIXED, pre-defined joint vector (init_q is the
    only randomized one) -- there is no per-episode IK solve anywhere in
    this list. Gripper stays open throughout (reach task, nothing grasped).
    """
    return [
        ("init_to_target", target_q, GRIPPER_OPEN, SEGMENT_DURATION_S["init_to_target"]),
        ("hold_at_target", target_q, GRIPPER_OPEN, SEGMENT_DURATION_S["hold_at_target"]),
    ]


def run_episode(model, data, renderer, jnt_ids, qpos_adr, arm_act_ids,
                 gripper_act_id, gripper_qpos_adr, episode_idx: int,
                 video_path: Path, target_q: np.ndarray, goal_xyz) -> EpisodeLog:
    log = EpisodeLog()
    writer = imageio.get_writer(str(video_path), fps=CONTROL_HZ, codec="libx264",
                                 quality=8, macro_block_size=1)

    # sim-to-real domain randomization: gravity, friction, lighting -- fresh
    # draw every episode, applied before any stepping so it's baked into the
    # whole trajectory, not just the render.
    apply_domain_randomization(model)

    home_arm_q = HOME_Q
    init_q = sample_random_init(model, data, home_arm_q, jnt_ids,
                                 INIT_SAMPLE_FRACTION, N_COLLISION_CHECK_TRIES)

    # place the arm at the (randomized) initial pose, gripper open
    data.qpos[jnt_ids] = init_q
    data.qvel[:] = 0.0
    if gripper_act_id is not None:
        data.ctrl[gripper_act_id] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    segments = build_waypoint_sequence(init_q, target_q)
    prev_arm_q = init_q.copy()
    prev_gripper = GRIPPER_OPEN
    t0 = time.time()
    substeps_per_ctrl = max(1, int(round(SIM_HZ / CONTROL_HZ)))

    for seg_name, arm_target, gripper_target, duration in segments:
        arm_wp = quintic_interp(prev_arm_q, arm_target, duration, CONTROL_HZ)
        grip_wp = np.linspace(prev_gripper, gripper_target, arm_wp.shape[0])

        for step_i in range(arm_wp.shape[0]):
            q_cmd = arm_wp[step_i]
            g_cmd = grip_wp[step_i]

            data.ctrl[arm_act_ids] = q_cmd
            if gripper_act_id is not None:
                data.ctrl[gripper_act_id] = g_cmd

            for _ in range(substeps_per_ctrl):
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=CAMERA_NAME)
            frame = renderer.render()
            writer.append_data(frame)

            gripper_qpos = data.qpos[gripper_qpos_adr] if gripper_qpos_adr is not None else 0.0
            full_qpos = np.concatenate([data.qpos[qpos_adr], [gripper_qpos]])
            full_qvel = np.concatenate([data.qvel[qpos_adr], [0.0]])
            action = np.concatenate([q_cmd, [g_cmd]])

            log.qpos.append(full_qpos.tolist())
            log.qvel.append(full_qvel.tolist())
            log.action.append(action.tolist())
            log.phase.append(seg_name)
            log.timestamp.append(time.time() - t0)
            log.goal_xyz.append(list(goal_xyz))

        prev_arm_q = arm_target
        prev_gripper = gripper_target

    writer.close()
    return log


def resolve_joint_and_actuator_ids(model):
    """
    Resolves every id by NAME, never by assuming joint/actuator/qpos indices
    line up numerically (they coincidentally do for the arm in the current
    scene layout, but that's fragile -- a body added earlier in
    build_scene.py, or a joint type change, would silently break index-based
    lookups without erroring).
    """
    jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    if any(i < 0 for i in jnt_ids):
        raise ValueError(f"Could not find one or more joints in {ARM_JOINT_NAMES}; "
                          f"check ARM_JOINT_NAMES against your MJCF.")
    qpos_adr = model.jnt_qposadr[jnt_ids]

    arm_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINT_NAMES]
    if any(i < 0 for i in arm_act_ids):
        raise ValueError(f"Could not find one or more position actuators named "
                          f"{ARM_JOINT_NAMES}; check your MJCF's <actuator> names.")

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT_NAME)
    gripper_qpos_adr = None
    if gripper_act_id < 0:
        print(f"WARNING: gripper actuator '{GRIPPER_ACT_NAME}' not found; "
              f"episodes will run arm-only.")
        gripper_act_id = None
    else:
        finger_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_finger_joint1")
        if finger_jnt_id >= 0:
            gripper_qpos_adr = model.jnt_qposadr[finger_jnt_id]

    return jnt_ids, qpos_adr, arm_act_ids, gripper_act_id, gripper_qpos_adr


# ------------------------- LeRobot-style dataset writer --------------------

def write_lerobot_layout(out_dir: Path, episode_logs: list, video_paths: list,
                          fps: int, task_description: str):
    """
    Minimal LeRobot v2-style layout, written without depending on the
    `lerobot` package (so this runs even if it's not installed on the
    training server). If you DO have `lerobot` installed, prefer building
    the dataset with lerobot.common.datasets.lerobot_dataset.LeRobotDataset
    instead -- it guarantees exact compatibility with downstream loaders.
    See the commented block at the bottom of this function.
    """
    data_dir = out_dir / "data" / "chunk-000"
    video_dir = out_dir / "videos" / "chunk-000" / f"observation.images.{CAMERA_NAME}"
    meta_dir = out_dir / "meta"
    for d in (data_dir, video_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    global_index = 0
    for ep_idx, (log, video_path) in enumerate(zip(episode_logs, video_paths)):
        n = len(log.qpos)
        df = pd.DataFrame({
            "observation.state": log.qpos,
            "observation.velocity": log.qvel,
            "action": log.action,
            "phase": log.phase,
            "timestamp": log.timestamp,
            "goal_xyz": log.goal_xyz,
            "frame_index": np.arange(n),
            "episode_index": ep_idx,
            "index": np.arange(global_index, global_index + n),
            "task_index": 0,
        })
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet")

        final_video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
        if video_path != final_video_path:
            os.replace(video_path, final_video_path)

        episodes_meta.append({
            "episode_index": ep_idx,
            "length": n,
            "tasks": [task_description],
        })
        global_index += n

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    info = {
        "codebase_version": "v2.0 (hand-written, non-lerobot-lib)",
        "robot_type": "fr3",
        "fps": fps,
        "total_episodes": len(episode_logs),
        "total_frames": global_index,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [N_ARM_JOINTS + 1]},
            "observation.velocity": {"dtype": "float32", "shape": [N_ARM_JOINTS + 1]},
            "action": {"dtype": "float32", "shape": [N_ARM_JOINTS + 1]},
            f"observation.images.{CAMERA_NAME}": {"dtype": "video", "shape": [RENDER_H, RENDER_W, 3]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- Alternative, if `lerobot` is installed on your server: -----------
    # from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    # dataset = LeRobotDataset.create(
    #     repo_id="local/fr3_expert_v1",
    #     fps=fps,
    #     features={...},   # same feature dict as above, lerobot's own schema
    #     root=out_dir,
    # )
    # for ep in episode_logs: dataset.add_frame(...); dataset.save_episode()
    # ------------------------------------------------------------------------


def main():
    np.random.seed(0)  # EDIT ME / remove for non-deterministic runs
    model = build_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    jnt_ids, qpos_adr, arm_act_ids, gripper_act_id, gripper_qpos_adr = \
        resolve_joint_and_actuator_ids(model)

    tmp_dir = OUT_DIR / "_tmp_videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    episode_logs, video_paths = [], []
    for ep in range(N_EPISODES):
        vpath = tmp_dir / f"episode_{ep:06d}.mp4"
        log = run_episode(model, data, renderer, jnt_ids, qpos_adr, arm_act_ids,
                           gripper_act_id, gripper_qpos_adr, ep, vpath,
                           target_q=TARGET_Q, goal_xyz=tuple(OBJECT_POS))
        episode_logs.append(log)
        video_paths.append(vpath)
        print(f"episode {ep+1}/{N_EPISODES} done, {len(log.qpos)} frames")

    write_lerobot_layout(OUT_DIR, episode_logs, video_paths, CONTROL_HZ, TASK_DESCRIPTION)
    tmp_dir.rmdir()
    print(f"Wrote {N_EPISODES} episodes to {OUT_DIR}")


if __name__ == "__main__":
    main()