"""
Milestone 1用: 生成純reaching demo(不抓不放),重用跟pick-and-place版本同一套場景/
domain randomization/干擾物基礎設施,只是動作序列換成reach_task.py那套簡化版本。

跟generate_dataset.py（pick-and-place版本)刻意保持資料格式一致(q/tcp_pos/link_pos/
goal_pos這些欄位命名相同),這樣你的Transformer/dataloader架構之後要從reach升級到
pick-and-place時,大部分程式碼可以直接重用,不用重寫。

主要差異:
    - action是7維(只有Δq,沒有Δgripper),不是8維
    - 沒有place_target_pos這個欄位(reach任務的目標就是goal_pos本身)
    - 成功判定是「TCP最終有沒有夠靠近goal_pos」,不是看方塊位置
"""
import os
import argparse
import numpy as np
import mujoco

from scene import build_model, TABLE_MATERIALS, FLOOR_MATERIALS
from reach_task import generate_reach_trajectory
from pick_task import GRIPPER_OPEN
from ik import N_ARM_JOINTS, solve_ik
from domain_rand import apply_domain_randomization, capture_nominal
from distractor_rand import randomize_distractors
from generate_dataset import (
    Q_HOME, WORKSPACE_X, WORKSPACE_Y, TABLE_TOP_Z, CUBE_HALF, SETTLE_SUBSTEPS,
    IMG_SIZE_DEFAULT,
)
from storage import save_video, save_episode_data, save_episode_meta

REACH_SUCCESS_THRESHOLD = 0.03  # TCP離目標點在3cm以內算成功
START_POSE_JITTER = 0.12        # 高處起始姿態的關節角隨機擾動範圍(rad)
HOLD_DURATION_SEC = 1.0         # 到達目標後停留的實際時間(秒),不是隨便抓的步數

# 「高處」起始姿態: 用IK反解出桌面正上方40cm、朝下的TCP姿態當基準,不是隨便猜關節角。
# x座標選0.3(離手臂基座較近)是因為實測過:純朝下姿態能達到的最大高度會隨水平距離增加而
# 遞減(關節4活動範圍限制手肘伸展程度),x=0.5時最多只能到0.74m(差0.4m目標6cm),
# x=0.3時可以精確達到0.8m(誤差為0),兩者都測過。
HIGH_TARGET_TCP = np.array([0.3, 0.0, TABLE_TOP_Z + 0.4])  # 桌面上方40cm,精確可達
HIGH_TARGET_QUAT = np.array([0, 1, 0, 0])  # 朝下


def _compute_high_pose(model, data):
    """反解出「高處」起始姿態的關節角,而不是手動猜——實測過幾個Franka常見的
    home/ready姿態,TCP離桌面高度都只有6~9cm,不夠「高」,所以改用IK直接對準
    一個明確偏高的座標點(桌面上方40cm)。"""
    return solve_ik(model, data, HIGH_TARGET_TCP, HIGH_TARGET_QUAT, Q_HOME)


def sample_high_start_pose(rng, model, high_pose):
    """每條trajectory的起始關節角,在HIGH_POSE附近隨機擾動(不是Q_HOME附近),
    確保手臂一開始就在桌面上方偏高的位置,而不是貼著桌面。"""
    jitter = rng.uniform(-START_POSE_JITTER, START_POSE_JITTER, size=7)
    q = high_pose + jitter
    margin = 0.05
    lo = model.jnt_range[:7, 0] + margin
    hi = model.jnt_range[:7, 1] - margin
    return np.clip(q, lo, hi)


def sample_reach_target(rng):
    """reach目標點,直接沿用跟cube抓取同一個workspace範圍,方便之後升級成pick-and-place
    時goal_pos的語意可以無縫接軌。z是方塊貼著桌面的高度(方塊被實際擺在這裡當視覺標記)。"""
    x = rng.uniform(*WORKSPACE_X)
    y = rng.uniform(*WORKSPACE_Y)
    z = TABLE_TOP_Z + CUBE_HALF
    return np.array([x, y, z])


def _cam_key(cam_name):
    return cam_name.replace("_cam", "")


def run_and_record(model, data, renderer, target_pos, q_start, q_home_end, rng,
                    cameras=("scene_cam",), capture_depth=True, use_distractors=True,
                    hold_steps=None):
    tcp_id = model.site("tcp").id
    grip_id = model.actuator("hand_actuator8").id
    link_ids = [model.body(f"fr3_link{i}").id for i in range(1, 8)]
    cube_qpos_addr = model.jnt_qposadr[model.joint("cube_joint").id]
    target_mocap_id = model.body("target_marker").mocapid[0]

    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM_JOINTS] = q_start
    # 夾爪不主動控制(還沒到抓取階段),但設成open狀態,一開始就打開——這樣手臂才能真的
    # 壓到目標高度而不會撞進方塊裡(closed狀態的話,指尖沒有跨開,壓到方塊中心高度會穿模)
    finger1_qpos_addr = int(model.joint("hand_finger_joint1").qposadr[0])
    finger2_qpos_addr = int(model.joint("hand_finger_joint2").qposadr[0])
    data.ctrl[grip_id] = GRIPPER_OPEN
    data.qpos[finger1_qpos_addr] = 0.04
    data.qpos[finger2_qpos_addr] = 0.04
    # 把方塊直接擺在reach目標位置當作視覺標記(不用真的去夾它,純粹讓人眼/影像有東西可以
    # 核對「手臂有沒有真的伸到目標」)。黑色方框標記這裡沒有語意用途,挪到很遠的地方避免
    # 誤導畫面(reach任務用不到place target,但scene.py的場景結構本來就有這個mocap body)
    data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = target_pos
    data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = [1, 0, 0, 0]
    data.mocap_pos[target_mocap_id] = np.array([2.0, 2.0, 0.0])  # 挪出視野外
    if use_distractors:
        randomize_distractors(model, data, rng, target_pos, WORKSPACE_X, WORKSPACE_Y, TABLE_TOP_Z)
    mujoco.mj_forward(model, data)

    waypoints = generate_reach_trajectory(model, data, target_pos, q_start, q_home_end, rng,
                                           hold_steps=hold_steps)

    # 跟pick-and-place版本同樣的原因:規劃階段的IK暫存運算會讓data.qpos停在殘留狀態,
    # 正式replay前要重設回q_start(夾爪的qpos/ctrl不用重設,上面設過的open狀態還在)
    data.qpos[:N_ARM_JOINTS] = q_start
    data.qvel[:N_ARM_JOINTS] = 0
    data.qvel[finger1_qpos_addr - 1:finger1_qpos_addr + 1] = 0  # 順便清finger的殘留速度
    mujoco.mj_forward(model, data)

    record = {"q": [], "tcp_pos": [], "tcp_mat": [], "link_pos": [], "link_mat": []}
    for cam in cameras:
        record[f"img_{_cam_key(cam)}"] = []
        if capture_depth:
            record[f"depth_{_cam_key(cam)}"] = []

    reach_tcp = None  # hold階段結束那一刻的TCP位置,才是真正判定「有沒有搆到目標」的時間點
                       # (整條trajectory結束時手臂已經走return_home離開了,不能拿那個判定成功)
    for wp in waypoints:
        data.ctrl[:N_ARM_JOINTS] = wp["q"]
        for _ in range(SETTLE_SUBSTEPS):
            mujoco.mj_step(model, data)

        record["q"].append(data.qpos[:N_ARM_JOINTS].copy())
        record["tcp_pos"].append(data.site_xpos[tcp_id].copy())
        record["tcp_mat"].append(data.site_xmat[tcp_id].copy())
        record["link_pos"].append(data.xpos[link_ids].copy())
        record["link_mat"].append(data.xmat[link_ids].copy())
        if wp["phase"] == "hold":
            reach_tcp = data.site_xpos[tcp_id].copy()

        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            record[f"img_{_cam_key(cam)}"].append(renderer.render().copy())
            if capture_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=cam)
                record[f"depth_{_cam_key(cam)}"].append(
                    renderer.render().astype(np.float16).copy())
                renderer.disable_depth_rendering()

    success = np.linalg.norm(reach_tcp - target_pos) <= REACH_SUCCESS_THRESHOLD
    return record, success, reach_tcp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie_path", default="/home/claude/mujoco_menagerie")
    ap.add_argument("--robosuite_path", default="/home/claude/robosuite")
    ap.add_argument("--out_dir", default="./dataset_reach")
    ap.add_argument("--n_traj", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cameras", nargs="+", default=["scene_cam"],
                     choices=["scene_cam", "wrist_cam"])
    ap.add_argument("--no_depth", action="store_true")
    ap.add_argument("--no_dr", action="store_true")
    ap.add_argument("--no_dynamics_dr", action="store_true")
    ap.add_argument("--no_distractors", action="store_true")
    ap.add_argument("--img_height", type=int, default=IMG_SIZE_DEFAULT[0])
    ap.add_argument("--img_width", type=int, default=IMG_SIZE_DEFAULT[1])
    ap.add_argument("--video_quality", type=int, default=8,
                     help="imageio mp4品質0~10,預設8畫質沒有肉眼可見劣化")
    args = ap.parse_args()
    IMG_SIZE = (args.img_height, args.img_width)

    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ("videos", "data", "meta"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    model, data = build_model(args.menagerie_path, args.robosuite_path)
    renderer = mujoco.Renderer(model, height=IMG_SIZE[0], width=IMG_SIZE[1])
    control_fps = 1.0 / (SETTLE_SUBSTEPS * model.opt.timestep)
    print(f"影片frame rate: {control_fps:.2f}fps "
          f"(SETTLE_SUBSTEPS={SETTLE_SUBSTEPS} × timestep={model.opt.timestep}s)")
    capture_depth = not args.no_depth
    nominal = capture_nominal(model, table_materials=TABLE_MATERIALS,
                               floor_materials=FLOOR_MATERIALS)

    high_pose = _compute_high_pose(model, data)
    hold_steps_1sec = round(HOLD_DURATION_SEC / (SETTLE_SUBSTEPS * model.opt.timestep))
    print(f"高處起始姿態(TCP離桌面{HIGH_TARGET_TCP[2]-TABLE_TOP_Z:.2f}m): {high_pose.round(3)}")
    print(f"停留{HOLD_DURATION_SEC}秒 = {hold_steps_1sec}個waypoint "
          f"(SETTLE_SUBSTEPS={SETTLE_SUBSTEPS} × timestep={model.opt.timestep}s)")

    n_ok, n_fail = 0, 0
    for i in range(args.n_traj):
        if not args.no_dr:
            dr_params = apply_domain_randomization(model, rng, nominal,
                                                     enable_dynamics_dr=not args.no_dynamics_dr)
        else:
            dr_params = {}

        target_pos = sample_reach_target(rng)
        q_start = sample_high_start_pose(rng, model, high_pose)
        record, success, reach_tcp = run_and_record(
            model, data, renderer, target_pos, q_start, high_pose, rng,
            cameras=args.cameras, capture_depth=capture_depth,
            use_distractors=not args.no_distractors, hold_steps=hold_steps_1sec)

        if not success:
            n_fail += 1
            print(f"[{i}] target={target_pos.round(3)} FAILED "
                  f"(reach_tcp={reach_tcp.round(3)}, dist={np.linalg.norm(reach_tcp-target_pos):.4f}) - 略過不存")
            continue

        n_ok += 1
        qs = np.stack(record["q"])
        actions = np.diff(qs, axis=0)  # (T-1, 7),沒有gripper維度

        traj_name = f"traj_{i:04d}"
        for cam_key, frames in record.items():
            if not cam_key.startswith("img_"):
                continue
            cam_name = cam_key[len("img_"):]
            save_video(np.stack(frames).astype(np.uint8),
                       os.path.join(args.out_dir, "videos", f"{traj_name}_{cam_name}.mp4"),
                       fps=control_fps, quality=args.video_quality)
        depth_arrays = {
            k: np.stack(v).astype(np.float16) for k, v in record.items()
            if k.startswith("depth_")
        }
        if depth_arrays:
            np.savez_compressed(
                os.path.join(args.out_dir, "videos", f"{traj_name}_depth.npz"), **depth_arrays)

        save_episode_data({
            "q": qs.astype(np.float32),
            "tcp_pos": np.stack(record["tcp_pos"]).astype(np.float32),
            "tcp_mat": np.stack(record["tcp_mat"]).astype(np.float32),
            "link_pos": np.stack(record["link_pos"]).astype(np.float32),
            "link_mat": np.stack(record["link_mat"]).astype(np.float32),
            "action": actions.astype(np.float32),
        }, os.path.join(args.out_dir, "data", f"{traj_name}.parquet"), fps=control_fps)

        save_episode_meta({
            "goal_pos": target_pos.astype(np.float32),
            "fps": control_fps,
            **{f"dr_{k}": v for k, v in dr_params.items()},
        }, os.path.join(args.out_dir, "meta", f"{traj_name}.json"))

        print(f"[{i}] target={target_pos.round(3)} OK, T={len(qs)}, "
              f"final_dist={np.linalg.norm(reach_tcp-target_pos):.4f}")

    print(f"\n完成: {n_ok} 成功 / {n_fail} 失敗 (總共{args.n_traj}條)")


if __name__ == "__main__":
    main()