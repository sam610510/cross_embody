"""
視覺化工具,確認generate_dataset.py/generate_reach_dataset.py產生的trajectory是否合理。
讀的是storage.py那套mp4+parquet+json格式(videos/<traj>_<cam>.mp4, data/<traj>.parquet,
meta/<traj>.json),不是舊版的單一npz檔。

用法1: 看單一trajectory的動畫 + 關節角/夾爪/動作曲線
    python3 visualize_dataset.py single --data_dir dataset --traj traj_0000 --out_dir ./viz

用法2: 快速掃一批trajectory,產生「開始 vs 結束」縮圖網格,一眼看出哪些抓取失敗/怪異
    python3 visualize_dataset.py grid --data_dir dataset --out_dir ./viz --n 20
"""
import argparse
import os
import glob
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from storage import load_episode_data, load_episode_meta


def _list_trajs(data_dir, cam="scene"):
    """從videos/資料夾掃出有哪些trajectory,回傳排序過的traj name清單(不含副檔名/相機後綴)。"""
    pattern = os.path.join(data_dir, "videos", f"*_{cam}.mp4")
    names = [os.path.basename(p)[:-len(f"_{cam}.mp4")] for p in glob.glob(pattern)]
    return sorted(names)


def visualize_single(data_dir, traj_name, out_dir, cam="scene"):
    os.makedirs(out_dir, exist_ok=True)

    # -- 1. GIF動畫 --
    video_path = os.path.join(data_dir, "videos", f"{traj_name}_{cam}.mp4")
    reader = imageio.get_reader(video_path)
    frames = [f for f in reader]
    reader.close()
    gif_path = os.path.join(out_dir, f"{traj_name}.gif")
    imageio.mimsave(gif_path, frames, fps=15, loop=0)

    # -- 2. 關節角 / (夾爪開合,如果有的話) / 動作幅度 --
    # reach-only版本(generate_reach_dataset.py)沒有gripper_qpos跟8維action(只有7維Δq),
    # pick-and-place版本(generate_dataset.py)才有。這裡自動偵測,兩種格式都能畫。
    df = load_episode_data(os.path.join(data_dir, "data", f"{traj_name}.parquet"))
    q = np.stack(df["q"].to_list())          # (T,7)
    action = np.stack(df["action"].to_list())  # 最後一行是補齊的重複值(見storage.py說明)
    has_gripper = "gripper_qpos" in df.columns

    n_rows = 3 if has_gripper else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True)

    axes[0].plot(q)
    axes[0].set_title(f"{traj_name} — Arm joint angles q (rad)")
    axes[0].legend([f"j{i+1}" for i in range(7)], loc="upper right", fontsize=7, ncol=4)
    axes[0].set_ylabel("rad")

    action_row = 1
    if has_gripper:
        grip = df["gripper_qpos"].to_numpy()
        axes[1].plot(grip, color="green")
        axes[1].set_title("Gripper opening (0=closed, 0.04=open)")
        axes[1].set_ylabel("m")
        action_row = 2

    # 最後一行action是補齊值(跟前一行重複),畫圖時去掉避免曲線尾端出現假的0
    action_valid = action[:-1]
    axes[action_row].plot(np.linalg.norm(action_valid[:, :7], axis=1), label="|Δq| (arm)")
    if has_gripper:
        axes[action_row].plot(np.abs(action_valid[:, 7]), label="|Δgripper|")
    axes[action_row].set_title("Per-step action magnitude (check for sudden spikes)")
    axes[action_row].set_xlabel("timestep")
    axes[action_row].legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{traj_name}_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()

    meta_path = os.path.join(data_dir, "meta", f"{traj_name}.json")
    meta = load_episode_meta(meta_path) if os.path.exists(meta_path) else {}

    print(f"GIF: {gif_path}")
    print(f"曲線圖: {plot_path}")
    print(f"|Δq|最大值: {np.linalg.norm(action_valid[:,:7],axis=1).max():.4f} rad/step "
          f"(如果遠大於其他步數,可能代表IK在某一步跳解)")
    if meta:
        print(f"meta: fps={meta.get('fps')}, goal_pos={meta.get('goal_pos')}")


def visualize_grid(data_dir, out_dir, n, cam="scene"):
    os.makedirs(out_dir, exist_ok=True)
    traj_names = _list_trajs(data_dir, cam=cam)[:n]
    if not traj_names:
        print(f"{data_dir}/videos 底下沒有找到 *_{cam}.mp4")
        return

    # 每條trajectory顯示3欄: start / mid(通常是手臂舉最高,最容易被攝影機裁到的時刻) / end
    cell_size = 3.2  # 每張圖的邊長(inch),加大到看得清楚細節
    fig, axes = plt.subplots(len(traj_names), 3,
                              figsize=(cell_size * 3, cell_size * len(traj_names)))
    if len(traj_names) == 1:
        axes = axes[None, :]

    for i, traj_name in enumerate(traj_names):
        video_path = os.path.join(data_dir, "videos", f"{traj_name}_{cam}.mp4")
        reader = imageio.get_reader(video_path)
        n_frames = reader.count_frames()
        mid_idx = n_frames // 2
        # mp4是time-series編碼,跟JPEG不一樣沒辦法真正「只解一幀」不碰其他幀,
        # 但get_data()還是比整個mimsave讀進記憶體省事,實務上夠用
        start_frame = reader.get_data(0)
        mid_frame = reader.get_data(mid_idx)
        end_frame = reader.get_data(n_frames - 1)
        reader.close()

        for col, (frame, label) in enumerate([
            (start_frame, "start"),
            (mid_frame, f"mid (t={mid_idx})"),
            (end_frame, "end"),
        ]):
            axes[i, col].imshow(frame, interpolation="antialiased")
            title = f"{traj_name}\n{label}" if col == 0 else label
            axes[i, col].set_title(title, fontsize=9)
            axes[i, col].axis("off")

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "grid_start_mid_end.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"總覽圖: {grid_path} (共{len(traj_names)}條trajectory,建議一次看不超過10條,太多張會很難逐一檢查)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("--data_dir", required=True)
    p1.add_argument("--traj", required=True, help="例如 traj_0000(不含副檔名/相機後綴)")
    p1.add_argument("--cam", default="scene")
    p1.add_argument("--out_dir", default="./viz")

    p2 = sub.add_parser("grid")
    p2.add_argument("--data_dir", required=True)
    p2.add_argument("--cam", default="scene")
    p2.add_argument("--out_dir", default="./viz")
    p2.add_argument("--n", type=int, default=20)

    args = ap.parse_args()
    if args.cmd == "single":
        visualize_single(args.data_dir, args.traj, args.out_dir, cam=args.cam)
    else:
        visualize_grid(args.data_dir, args.out_dir, args.n, cam=args.cam)