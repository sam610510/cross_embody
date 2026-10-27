# 影像物體位置編碼器（vision_plan 步驟 1、3）

放在 `~/cross_embody/vision/`。三支程式：

| 檔案 | 用途 |
|---|---|
| `gen_vision_data.py` | 產生資料：固定相機、方塊／膠帶隨機位置、FR3 隨機姿態、遮擋物、外觀隨機化 |
| `check_vision_data.py` | 檢查資料：標籤投影 vs 分割重心、可見度統計、疊圖 montage |
| `train_vision_encoder.py` + `vision_encoder.py` | 訓練／評估：ResNet18 到 layer3 + layer2 融合（stride 8 熱圖）→ soft-argmax → (u,v) → 已知高度反投影成 (x,y,z) |

## 0. 對齊場景名稱（只要做一次）

```bash
cd ~/cross_embody/vision
export MUJOCO_GL=egl
python gen_vision_data.py --scene <你的FR3場景.xml> --list
```
依輸出設定 `--robot_root`（手臂＋夾爪的根 body）、`--arm_joint_regex`、`--finger_joint_regex`、
`--ee_site`（**用指尖中心 TCP**，hover 高度是以它計）、`--cube`、`--tape`、`--table_geom`、`--table_z`、`--keyframe`。

相機：`--cam_pos`、`--cam_lookat` 設成實驗室相機的實際位置；`--fovy` 實機是 D415 時約 40（D435 為 42.5）。

## 1. 產生資料

```bash
ARGS="--scene <scene.xml> --robot_root ... --ee_site ... --cube ... --tape ..."
python gen_vision_data.py $ARGS --out data/vis_fr3_train --n 50000 --workers 16 --seed 0
python gen_vision_data.py $ARGS --out data/vis_fr3_val   --n 2000  --workers 16 --seed 1 --start 1000000
python check_vision_data.py --data data/vis_fr3_train     # 看 montage.png 與投影誤差（應 < 1 px）
```
UR5e 零樣本測試集：同一支程式，`--scene` 換 UR5e 場景、其餘名稱參數換成 UR5e 的，`--seed 2`。
（UR5e 影像**只用來測試**，不進訓練。）

消融：`--hide_robot`（不畫手臂，上限）、`--mode_probs home:1`（只有示範附近姿態，不做姿態隨機化）、`--no_dr`。

## 2. 訓練

```bash
python train_vision_encoder.py --train data/vis_fr3_train --val data/vis_fr3_val --out runs/vis_v1
```
預設：GroupNorm、AdamW lr 2e-4 / wd 0.05、2000 步 warmup + cosine、bf16、early stopping patience 30
（指標 = 方塊＋膠帶世界座標誤差中位數）。解析度不夠時用 `--roi x0 y0 w h` 裁出桌面區域。

## 3. 評估（含 UR5e 零樣本）

```bash
python train_vision_encoder.py --eval_only --ckpt runs/vis_v1/best.pt --val data/vis_ur5e_test --out runs/vis_v1
```
輸出：mm 誤差（中位數／平均／p95）、依姿態模式與「EE 到方塊距離」分組、部分遮擋時的誤差、可見度準確率。

## 標籤（labels_*.npz）

`cube_pos`/`tape_pos`（世界座標 m）、`cube_uv`/`tape_uv`（連續像素，pixel i 涵蓋 [i,i+1)）、
`cube_vis`/`tape_vis`（可見比例 = 全場景可見像素 / 單獨渲染像素）、`mode`、`q`、`ee_pos`、`ee_cube_dist`、
`cam_pos`、`cam_xmat`、`K`（每張的相機參數，含外參抖動）。`seg/` 內 0 背景 1 方塊 2 膠帶 3 手臂 4 遮擋物 5 其他。

## 已知限制

- 標籤只涵蓋「夾取前」：方塊永遠在桌上。夾起後的位置請凍結或由夾爪狀態推得。
- 方塊角度固定（`--cube_yaw_deg` 可開，但編碼器目前不估角度）。
