"""
仿LeRobot的儲存格式:
    - 影像存mp4(h264影片編碼利用時間序列的冗餘,壓縮比逐幀JPEG好非常多——
      實測同一條250幀trajectory,JPEG-in-npz是12.2MB,mp4只要0.92MB,縮小13倍)
    - 數值資料(關節角/TCP/action/goal等)存parquet(欄位化格式,pandas/pyarrow
      原生支援,LeRobot資料集也是這樣存,比npz好整合進data pipeline)
    - episode層級的中繼資訊(goal_pos、place_target_pos、domain randomization參數)
      存成小的json,這些是純量/不定長度的東西,硬塞進parquet的每一行意義不大

輸出結構:
    out_dir/
        videos/traj_XXXX_<camera>.mp4
        data/traj_XXXX.parquet
        meta/traj_XXXX.json
"""
import os
import json
import numpy as np
import pandas as pd
import imageio

VIDEO_FPS = 30
VIDEO_QUALITY = 8  # imageio quality 0~10,實測過8畫質沒有肉眼可見的劣化


def save_video(frames, path, fps=VIDEO_FPS, quality=VIDEO_QUALITY):
    """frames: (T,H,W,3) uint8 array或list"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=quality,
                      macro_block_size=None)


def load_video(path):
    """回傳(T,H,W,3) uint8 array"""
    reader = imageio.get_reader(path)
    frames = np.stack([f for f in reader])
    reader.close()
    return frames


def save_episode_data(data_dict, path, fps=VIDEO_FPS):
    """
    data_dict: 每個key對應一個array,第一維是timestep。不同欄位長度可能不同
    (例如action比q少一行,因為action[t]代表q[t]->q[t+1]的變化量),這裡自動用
    最後一行補齊到相同長度T,方便直接讀成單一張DataFrame——被補齊的行不是真實資料,
    如果你需要精確知道原始長度,可以用q或其他不需要補齊的欄位反推真實T(=補齊前的長度)。
    多維欄位(例如(T,7)的q或(T,7,3)的link_pos)攤平成list column,parquet/pyarrow
    原生支援這種欄位型態。

    自動加上frame_index跟timestamp兩欄(對齊LeRobot慣例),timestamp用傳入的fps
    算,要跟save_video實際用的fps一致,不然時間軸會對不上影片。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T = max(len(v) for v in data_dict.values())
    cols = {
        "frame_index": np.arange(T),
        "timestamp": np.arange(T) / fps,
    }
    for k, v in data_dict.items():
        v = np.asarray(v)
        if len(v) < T:
            pad = np.repeat(v[-1:], T - len(v), axis=0)
            v = np.concatenate([v, pad], axis=0)
        if v.ndim == 1:
            cols[k] = v
        else:
            cols[k] = [row.tolist() for row in v.reshape(len(v), -1)]
    df = pd.DataFrame(cols)
    df.to_parquet(path)


def load_episode_data(path):
    return pd.read_parquet(path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"無法序列化的型態: {type(o)}")


def save_episode_meta(meta_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta_dict, f, indent=2, default=_json_default)


def load_episode_meta(path):
    with open(path) as f:
        return json.load(f)