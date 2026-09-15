"""
主程式:隨機生成N條FR3「抓取5cm綠色方塊、搬到黑色膠帶標記的7cm方形內放下」的demo,
存成每條trajectory一個.npz。

每個timestep存:
    q                (7,)     手臂關節角 (rad)
    gripper_qpos     (1,)     夾爪開合量 (m, 0=closed 0.04=open)
    img_<cam>        (H,W,3)  uint8    RGB影像,每個--cameras指定的相機各一個key
    depth_<cam>      (H,W)    float16  深度圖(公尺),同上
    goal_pos         (3,)     方塊初始位置(pick target,privileged,整條trajectory固定不變)
    goal_quat        (4,)     方塊初始朝向(wxyz),隨機yaw
    place_target_pos (3,)     目標方形中心點世界座標(place target,privileged,整條trajectory固定不變)

action label(給training時用Huber loss回歸):
    action[t] = concat( q[t+1]-q[t],  gripper_qpos[t+1]-gripper_qpos[t] )   shape (8,)
最後一個timestep沒有下一步,不產生action(training時該幀只當context不當label)。

成功判定: 方塊最終中心點(x,y)有沒有落在目標方形範圍內,不用對到正中心,
碰到方形邊界內都算(不是像之前pick-only版本那樣只看有沒有被舉起來)。
"""
import os
import argparse
import io
import numpy as np
import mujoco
from PIL import Image

from scene import build_model, TABLE_MATERIALS, FLOOR_MATERIALS, TARGET_SQUARE_SIZE
from pick_task import generate_pick_trajectory, N_ARM_JOINTS
from domain_rand import apply_domain_randomization, capture_nominal
from distractor_rand import randomize_distractors
from storage import save_video, save_episode_data, save_episode_meta

Q_HOME = np.array([0, -0.3, 0, -2.2, 0, 2.0, 0.79])  # 固定完成點,每條trajectory結尾都回到這裡
START_POSE_JITTER = 0.15  # 起始姿態每個關節角的隨機擾動範圍(rad),約±8.6度
WORKSPACE_X = (0.40, 0.60)
WORKSPACE_Y = (-0.18, 0.18)
TABLE_TOP_Z = 0.4
CUBE_HALF = 0.025
TARGET_AREA_SIZE = 0.40   # 目標方形中心點取樣範圍: 桌子中心40x40cm內
MIN_CUBE_TARGET_CLEARANCE = 0.12  # cube起始位置跟目標方形中心至少要距離這麼遠,
                                   # 不然pick跟place位置太近,搬運距離太短失去任務意義
SETTLE_SUBSTEPS = 40  # 每個waypoint之間實際physics step數,讓position servo追上目標。
# 手臂掛載到桌面高度後,往下伸的姿態對抗重力的方式跟原本floor-mounted時完全不同,
# 實測過SETTLE_SUBSTEPS=4時descend階段TCP追蹤誤差高達7.6cm(遠超夾爪能容忍的範圍,
# 導致抓取幾乎全部失敗),40步時誤差降到0.44cm才安全。這個值比原本(4)貴10倍運算,
# 是手臂掛載位置改變後必要的代價,不是隨便調大。
IMG_SIZE_DEFAULT = (1080, 1920)  # D415 RGB鏡頭原生解析度(H, W),16:9


def sample_cube_pose(rng):
    """位置在桌面workspace內均勻取樣;朝向只隨機yaw(方塊平放,不會翻滾),
    因為是top-down grasp,平躺方塊繞z軸轉不影響可抓性(對角線7.07cm仍小於夾爪最大開合8cm)。"""
    x = rng.uniform(*WORKSPACE_X)
    y = rng.uniform(*WORKSPACE_Y)
    z = TABLE_TOP_Z + CUBE_HALF
    pos = np.array([x, y, z])

    yaw = rng.uniform(0, 2 * np.pi)
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # wxyz, 繞z軸
    return pos, quat


def sample_target_pos(rng, cube_xy, table_center=(0.5, 0.0), max_tries=50):
    """目標方形中心點: 桌子中心40x40cm範圍內取樣,跟cube起始位置保持最小間距,
    避免抓取跟放置位置太近讓搬運距離短到失去任務意義。"""
    half = TARGET_AREA_SIZE / 2
    for _ in range(max_tries):
        x = rng.uniform(table_center[0] - half, table_center[0] + half)
        y = rng.uniform(table_center[1] - half, table_center[1] + half)
        if np.linalg.norm([x, y] - np.asarray(cube_xy)) >= MIN_CUBE_TARGET_CLEARANCE:
            return np.array([x, y, TABLE_TOP_Z])
    return np.array([x, y, TABLE_TOP_Z])  # 找不到夠遠的位置就將就用最後一次抽到的


def _cam_key(cam_name: str) -> str:
    return cam_name.replace("_cam", "")


def encode_jpeg_frames(frames, quality=90):
    """把一段(T,H,W,3) uint8的RGB影像序列,每一幀編碼成JPEG bytes,存成object array。
    比raw array小8倍左右,比zlib壓縮的raw array讀取解碼快約25倍(JPEG解碼器是硬體/軟體
    高度優化過的,zlib對pixel data這種內容的壓縮/解壓縮效率遠不如JPEG的DCT編碼)。
    讀取時用decode_jpeg_frames()解回(T,H,W,3) uint8 array。"""
    out = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        out.append(buf.getvalue())
    return np.array(out, dtype=object)


def decode_jpeg_frames(jpeg_array):
    """把encode_jpeg_frames()存的object array解回(T,H,W,3) uint8 array。"""
    return np.stack([np.array(Image.open(io.BytesIO(b))) for b in jpeg_array])


def sample_start_pose(rng, model):
    """每條trajectory的起始關節角,在Q_HOME附近隨機擾動,不是每次都從完全一樣的固定姿態
    開始——避免模型學到「畫面長這樣才開始動作」這種跟起始姿態綁死的捷徑。
    用model.jnt_range夾住,確保擾動後還在關節限制內(留一點safety margin避免貼著極限)。"""
    jitter = rng.uniform(-START_POSE_JITTER, START_POSE_JITTER, size=7)
    q = Q_HOME + jitter
    margin = 0.05
    lo = model.jnt_range[:7, 0] + margin
    hi = model.jnt_range[:7, 1] - margin
    return np.clip(q, lo, hi)


def run_and_record(model, data, renderer, cube_pos, cube_quat, target_pos, q_start, q_home_end, rng,
                    cameras=("scene_cam",), capture_depth=True, use_distractors=True):
    grip_id = model.actuator("hand_actuator8").id
    tcp_id = model.site("tcp").id
    cube_qpos_addr = model.jnt_qposadr[model.joint("cube_joint").id]
    target_mocap_id = model.body("target_marker").mocapid[0]
    link_ids = [model.body(f"fr3_link{i}").id for i in range(1, 8)]  # 7個關節對應的link

    # reset
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM_JOINTS] = q_start
    data.qpos[cube_qpos_addr:cube_qpos_addr + 3] = cube_pos
    data.qpos[cube_qpos_addr + 3:cube_qpos_addr + 7] = cube_quat
    data.mocap_pos[target_mocap_id] = target_pos + np.array([0, 0, 0.001])  # 貼著桌面一點點厚度
    if use_distractors:
        randomize_distractors(model, data, rng, cube_pos, WORKSPACE_X, WORKSPACE_Y, TABLE_TOP_Z,
                               target_pos=target_pos)
    mujoco.mj_forward(model, data)

    waypoints = generate_pick_trajectory(model, data, cube_pos, cube_quat, target_pos,
                                          q_start, q_home_end, rng)

    # generate_pick_trajectory內部用data做IK規劃時的暫存運算,規劃完之後data.qpos[:7]
    # (手臂關節角)會停在最後一次IK求解的殘留狀態,不是q_start——實際物理模擬開始前要
    # 把手臂關節角重新設回q_start,不然第一步的起點就跟規劃的q_start對不上,誤差會一路
    # 帶到trajectory結束(實測過:沒有這個reset,結尾q跟q_home_end的差距可以到1.6 rad,
    # 完全沒收斂)。只重置手臂這一段,不用整個mj_resetData——那樣會連同cube/干擾物/mocap
    # 標記的位置也一起洗掉,而這些在IK規劃過程中只被mj_forward(純運動學,不含時間積分)
    # 讀取過,沒有被物理模擬移動過,本來就不需要重設
    data.qpos[:N_ARM_JOINTS] = q_start
    data.qvel[:N_ARM_JOINTS] = 0
    mujoco.mj_forward(model, data)

    record = {"q": [], "gripper_qpos": [], "tcp_pos": [], "tcp_mat": [],
              "link_pos": [], "link_mat": []}
    for cam in cameras:
        record[f"img_{_cam_key(cam)}"] = []
        if capture_depth:
            record[f"depth_{_cam_key(cam)}"] = []
    finger_qpos_id = int(model.joint("hand_finger_joint1").qposadr[0])

    for wp in waypoints:
        data.ctrl[:N_ARM_JOINTS] = wp["q"]
        data.ctrl[grip_id] = wp["gripper_ctrl"]
        for _ in range(SETTLE_SUBSTEPS):
            mujoco.mj_step(model, data)

        record["q"].append(data.qpos[:N_ARM_JOINTS].copy())
        record["gripper_qpos"].append(float(data.qpos[finger_qpos_id]))
        # TCP(末端點)跟每個link的世界座標位置+旋轉矩陣,MuJoCo物理模擬本來就會算出來
        # (data.site_xpos/xmat, data.xpos/xmat),這裡只是讀出來存,不用另外算FK。
        # 存旋轉矩陣(3x3攤平成9個數字)而不是quaternion,如果你要用6D continuity
        # representation(Zhou et al. CVPR 2019那篇),直接取矩陣前兩欄就好,不用
        # 自己再把quaternion轉一次矩陣。
        record["tcp_pos"].append(data.site_xpos[tcp_id].copy())
        record["tcp_mat"].append(data.site_xmat[tcp_id].copy())
        record["link_pos"].append(data.xpos[link_ids].copy())
        record["link_mat"].append(data.xmat[link_ids].copy())

        for cam in cameras:
            renderer.update_scene(data, camera=cam)
            record[f"img_{_cam_key(cam)}"].append(renderer.render().copy())

            if capture_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=cam)
                record[f"depth_{_cam_key(cam)}"].append(
                    renderer.render().astype(np.float16).copy())
                renderer.disable_depth_rendering()

    final_cube_pos = data.qpos[cube_qpos_addr:cube_qpos_addr + 3].copy()
    half = TARGET_SQUARE_SIZE / 2
    in_square = (abs(final_cube_pos[0] - target_pos[0]) <= half and
                 abs(final_cube_pos[1] - target_pos[1]) <= half)
    resting = final_cube_pos[2] < (TABLE_TOP_Z + CUBE_HALF + 0.02)  # 有沒有真的放下來(不是還懸空被夾著)
    place_success = in_square and resting
    return record, place_success, final_cube_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie_path", default="/home/claude/mujoco_menagerie")
    ap.add_argument("--robosuite_path", default="/home/claude/robosuite",
                     help="robosuite repo路徑,用來取得瓶罐干擾物的mesh資產")
    ap.add_argument("--out_dir", default="./dataset")
    ap.add_argument("--n_traj", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cameras", nargs="+", default=["scene_cam"],
                     choices=["scene_cam", "wrist_cam"],
                     help="先只用scene_cam;之後想比較wrist_cam效果時加 --cameras scene_cam wrist_cam")
    ap.add_argument("--no_depth", action="store_true",
                     help="不需要深度圖時加這個flag,省渲染時間跟硬碟空間")
    ap.add_argument("--no_dr", action="store_true",
                     help="完全關掉domain randomization(debug時方便對照)")
    ap.add_argument("--no_dynamics_dr", action="store_true",
                     help="只關掉動力學DR(質量/摩擦力/gain),保留視覺DR。動力學DR會讓成功率\n"
                          "略降(物理變化更大),如果成功率掉太多可以先關這個")
    ap.add_argument("--no_distractors", action="store_true",
                     help="不要干擾物,場景只有目標方塊")
    ap.add_argument("--img_height", type=int, default=IMG_SIZE_DEFAULT[0],
                     help=f"渲染畫面高度,預設{IMG_SIZE_DEFAULT[0]}(D415原生解析度)。"
                          "每條~110幀的trajectory在原生解析度下單條npz大約數十MB,"
                          "500條會是數十GB量級,硬碟空間/生成時間吃緊的話可以調小,"
                          "例如--img_height 360 --img_width 640")
    ap.add_argument("--img_width", type=int, default=IMG_SIZE_DEFAULT[1],
                     help=f"渲染畫面寬度,預設{IMG_SIZE_DEFAULT[1]}(D415原生解析度)")
    ap.add_argument("--video_quality", type=int, default=8,
                     help="RGB影像用mp4(h264)編碼存檔的品質(0~10,對應imageio的quality參數)。"
                          "實測quality=8同一條250幀trajectory只要0.92MB,比之前JPEG-in-npz的"
                          "12.2MB小13倍(影片編碼利用了逐幀之間的時間序列冗餘,不是JPEG那種"
                          "單張獨立壓縮)。depth資料類型不適合有損video編碼,維持npz float16存檔")
    args = ap.parse_args()
    IMG_SIZE = (args.img_height, args.img_width)

    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ("videos", "data", "meta"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    model, data = build_model(args.menagerie_path, args.robosuite_path)
    renderer = mujoco.Renderer(model, height=IMG_SIZE[0], width=IMG_SIZE[1])
    capture_depth = not args.no_depth
    nominal = capture_nominal(model, table_materials=TABLE_MATERIALS,
                               floor_materials=FLOOR_MATERIALS)

    # 影片實際的frame rate要對應SETTLE_SUBSTEPS*timestep算出來的真實控制頻率,
    # 不能沿用storage.py裡寫死的30fps預設值——不然影片播放速度跟parquet裡
    # 每個timestep的物理意義對不上
    control_fps = 1.0 / (SETTLE_SUBSTEPS * model.opt.timestep)
    print(f"影片frame rate: {control_fps:.2f}fps "
          f"(SETTLE_SUBSTEPS={SETTLE_SUBSTEPS} × timestep={model.opt.timestep}s)")

    n_ok, n_fail = 0, 0
    for i in range(args.n_traj):
        if not args.no_dr:
            dr_params = apply_domain_randomization(model, rng, nominal,
                                                     enable_dynamics_dr=not args.no_dynamics_dr)
        else:
            dr_params = {}

        cube_pos, cube_quat = sample_cube_pose(rng)
        target_pos = sample_target_pos(rng, cube_pos[:2])
        q_start = sample_start_pose(rng, model)
        record, success, final_pos = run_and_record(
            model, data, renderer, cube_pos, cube_quat, target_pos, q_start, Q_HOME, rng,
            cameras=args.cameras, capture_depth=capture_depth,
            use_distractors=not args.no_distractors)

        if not success:
            n_fail += 1
            print(f"[{i}] cube={cube_pos[:2].round(3)} target={target_pos[:2].round(3)} "
                  f"FAILED (final_xy={final_pos[:2].round(3)}) - 略過不存")
            continue

        n_ok += 1
        qs = np.stack(record["q"])
        gripper_qpos = np.array(record["gripper_qpos"])
        actions = np.concatenate([
            np.diff(qs, axis=0),
            np.diff(gripper_qpos, axis=0)[:, None],
        ], axis=1)  # (T-1, 8)

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
            # depth是float資料,不適合用影片編碼(有損、且不是給人眼看的),存法沿用npz
            np.savez_compressed(
                os.path.join(args.out_dir, "videos", f"{traj_name}_depth.npz"), **depth_arrays)

        save_episode_data({
            "q": qs.astype(np.float32),
            "gripper_qpos": gripper_qpos.astype(np.float32),
            "tcp_pos": np.stack(record["tcp_pos"]).astype(np.float32),
            "tcp_mat": np.stack(record["tcp_mat"]).astype(np.float32),
            "link_pos": np.stack(record["link_pos"]).astype(np.float32),
            "link_mat": np.stack(record["link_mat"]).astype(np.float32),
            "action": actions.astype(np.float32),
        }, os.path.join(args.out_dir, "data", f"{traj_name}.parquet"), fps=control_fps)

        save_episode_meta({
            "goal_pos": cube_pos.astype(np.float32),
            "goal_quat": cube_quat.astype(np.float32),
            "place_target_pos": target_pos.astype(np.float32),
            "fps": control_fps,
            **{f"dr_{k}": v for k, v in dr_params.items()},
        }, os.path.join(args.out_dir, "meta", f"{traj_name}.json"))

        print(f"[{i}] cube={cube_pos[:2].round(3)} target={target_pos[:2].round(3)} "
              f"OK, T={len(qs)}, final_xy={final_pos[:2].round(3)}")

    print(f"\n完成: {n_ok} 成功 / {n_fail} 失敗 (總共{args.n_traj}條)")


if __name__ == "__main__":
    main()