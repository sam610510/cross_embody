# FR3 Pick-and-Place Dataset Generator

抓取5cm綠色方塊,搬到黑色膠帶貼成的7cm方形標記內放下(不用對到正中心,落在方形範圍內就算成功)。

## 前置需求
```bash
pip install mujoco numpy pandas pyarrow imageio imageio-ffmpeg
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set franka_fr3 franka_emika_panda
cd ..
git clone --depth 1 --filter=blob:none --sparse https://github.com/ARISE-Initiative/robosuite.git
cd robosuite && git sparse-checkout set robosuite/models/assets/objects robosuite/models/assets/textures
```
`pandas`/`pyarrow`是存parquet要用的,`imageio-ffmpeg`是存mp4要用的(內建靜態ffmpeg執行檔,
不用你自己另外裝系統的ffmpeg)。

`scene.py` 用 MJCF 的 `MjSpec.attach()` API 把 `franka_fr3/fr3.xml`(手臂)跟
`franka_emika_panda/hand.xml`(夾爪)接在一起,而不是手動改XML合併——menagerie更新版本時比較不會壞掉。
TCP site偏移量(0.1034m)是Franka官方flange-to-fingertip標準值,已經在`build_spec()`裡驗證過跟
fingertip pad collision geom的實際位置吻合。

干擾物(瓶/罐/牛奶盒)用的是`robosuite`(MIT License, ARISE Initiative/Stanford)的物體mesh資產,
不是自己組合的primitive幾何,所以除了`mujoco_menagerie`還要多clone這個repo,`--robosuite_path`指到它。

## 相機規格(對應Intel RealSense D415)
`scene_cam`的`fovy`跟畫面長寬比是照D415官方RGB鏡頭規格算的:水平69.4°/垂直42.5°/對角77°,16:9。
因為這個FOV比想像中窄,如果照最初設想的「桌邊、離地110cm」貼近架設,手臂舉高時上半部會被裁掉
(實測過)。目前的位置(`[1.6, 0, 1.5]`,距工作區約1.6m、離地1.5m)是優先滿足「整隻手臂要完整入鏡」
調出來的,不是嚴格對應「桌邊110cm」這個原始假設——如果你實際上有其他固定點(牆角、天花板吊掛之類),
把`scene.py`裡`scene_cam`的`pos`/`xyaxes`換掉即可,`xyaxes`的算法見下方「相機視角怎麼算」。

`--img_height`/`--img_width`預設是D415原生解析度1920×1080,可以直接用、之後進模型再自行downsample。
實測檔案大小(sandbox軟體渲染跑出來的,大小是準的,但渲染*時間*在你4080上會快很多,這裡的時間
不能拿來參考):
- 純RGB、無depth: 單條trajectory約9.8MB,500條約5GB
- 含depth(預設開啟): 單條約14MB,500條約7GB

硬碟空間/生成時間吃緊的話可以調小,例如`--img_height 360 --img_width 640`。

**重要**: 預設存檔用`np.savez`(不壓縮)。實測過壓縮版本(`np.savez_compressed`)讀取一條
270×480解析度、174幀的trajectory要19秒,不壓縮版本只要0.3秒,快70倍,檔案只大了約1.7倍
(不是2倍)。這個差異不只影響你用`visualize_dataset.py`檢查資料時的等待時間,**training時
dataloader每個epoch都要重複付出這個解壓縮成本**,所以預設關閉壓縮換取讀取速度。如果你的硬碟
空間真的吃緊,想要壓縮版本,加`--compress`。

### 相機視角怎麼算(xyaxes)
`add_camera`的`xyaxes`參數是攝影機local x軸(right)+local y軸(up)在世界座標下的方向,不是直接給
朝向角度。算法是給定攝影機位置`cam_pos`跟想看的目標點`target`,算出look-at方向:
```python
forward = (target - cam_pos); forward /= norm(forward)
right = cross(forward, [0,0,1]); right /= norm(right)
up = cross(right, forward)
xyaxes = list(right) + list(up)
```
`fovy`則是D415規格直接查表得出的垂直視角(42.5度),不是算出來的。


```bash
MUJOCO_GL=egl python3 generate_dataset.py \
    --menagerie_path /path/to/mujoco_menagerie \
    --robosuite_path /path/to/robosuite \
    --out_dir ./dataset \
    --n_traj 500 \
    --seed 0
    # 預設只渲染scene_cam。之後想比較wrist_cam效果時:
    # --cameras scene_cam wrist_cam
    # 預設開啟domain randomization(視覺+動力學)。debug或想先確認baseline行為時:
    # --no_dr            完全關掉DR
    # --no_dynamics_dr   只關掉動力學DR(質量/摩擦力/gain),保留視覺DR(通常這個比較常用,
    #                    因為動力學DR會讓抓取成功率降低,你可以先用視覺DR-only跑一批,
    #                    確認vision encoder的sim-to-real穩健性沒問題,再視需要加回動力學DR)
```

`MUJOCO_GL=egl` 是headless渲染用的,如果你在有螢幕的環境跑(例如直接在4080那台桌機用GUI登入),
可以拿掉這個環境變數改用預設的GLFW backend。

## Pick-and-Place任務說明
- **目標方形**: 黑色膠帶貼成的空心正方形(外框邊長7cm,膠帶寬6mm),用mocap body承載,
  每條trajectory直接設定`data.mocap_pos`移動位置,不用重新compile。實作在`scene.py`的
  `_add_target_marker()`
- **目標方形位置**: 桌子中心40×40cm範圍內隨機取樣(`sample_target_pos()`),跟cube起始位置
  至少保持12cm間距,避免抓取跟放置位置太近讓搬運距離短到失去任務意義
- **動作序列**(`pick_task.py`): home→pregrasp→descend→close→lift→**transport→place_descend→
  release→retreat**,後面4個phase是這次新加的,取代原本pick-only版本裡沒有的搬運/放置行為
- **成功判定**: 方塊最終中心點(x,y)有沒有落在目標方形範圍內(不用對到正中心,`abs(final_xy -
  target_xy) <= 方形半邊長`即可),而且要確認方塊真的放下來了(z高度接近桌面,不是還被夾著懸空)。
  這跟原本pick-only版本的判定邏輯(方塊有沒有被舉起5cm)完全不同,重寫在`run_and_record()`裡
- 干擾物現在也會避開目標方形的位置(不只避開cube起始位置),避免卡在放置區裡

`goal_pos`/`goal_quat`(cube起始pose)跟`place_target_pos`(目標方形中心點)都存進npz,
兩個都是privileged資訊,對應你trunk架構裡FiLM要用的goal conditioning——如果你的M2/M3設計裡
goal只需要一個,可以只取其中一個用,兩個都留著是給你彈性之後決定怎麼組合(例如分階段conditioning:
搬運前用goal_pos,搬運後切換成place_target_pos)。

## 檔案說明
- `scene.py`: 組裝FR3(掛載在桌面邊緣)+夾爪+桌子(薄桌面+4桌腳)+方塊+3個干擾物+雙相機
- `ik.py`: Damped least-squares differential IK,用`tcp` site解算,warm-start前一步的q避免IK跳解
- `pick_task.py`: 產生 start→observe→pregrasp→descend→close→lift→transport→place→release→
  retreat→return_home 的waypoint序列,含動作形狀隨機化跟朝向slerp
- `reach_task.py`: pick_task.py的簡化版,只做reaching(不抓不放),重用同一套`_interp_ik`/`_slerp`
- `domain_rand.py`: 每條trajectory套用一次的domain randomization(燈光/材質/相機pose/質量/摩擦力/servo gain)
- `distractor_rand.py`: 每條trajectory隨機擺放3個干擾物(真實瓶罐mesh)的位置/朝向/顏色
- `storage.py`: 仿LeRobot的儲存格式(mp4影片+parquet數值資料+json中繼資料),取代原本的npz
- `generate_dataset.py`: pick-and-place主程式
- `generate_reach_dataset.py`: reach-only主程式(Milestone 1用)
- `visualize_dataset.py`: 確認生成結果用,讀新的mp4+parquet+json格式

## 資料格式(仿LeRobot: mp4+parquet+json,取代原本的npz)
```
out_dir/
    videos/traj_XXXX_<camera>.mp4      每個camera各自的影片(H.264,fps對應實際控制頻率)
    data/traj_XXXX.parquet             每個timestep的數值資料
    meta/traj_XXXX.json                整條trajectory固定不變的中繼資料
    videos/traj_XXXX_depth.npz         深度圖(如果有開depth的話,不適合用影片編碼,維持npz)
```
**跟真正的LeRobotDataset格式不完全一樣**——真正的格式還需要`meta/info.json`、
`episodes.jsonl`、`stats.json`這些全域中繼檔案,`lerobot`套件的`LeRobotDataset` class才能
直接讀。這裡只做了核心的「影像用mp4、數值用parquet」分工,你自己寫PyTorch Dataset讀取很直接,
但不能直接丟給`lerobot`套件讀。

**`data/traj_XXXX.parquet`欄位**:
| key | shape | 說明 |
|---|---|---|
| `frame_index` | 純量 | 第幾幀(對齊LeRobot慣例) |
| `timestamp` | 純量 | 秒,`frame_index / fps`算出來的 |
| `q` | (7,) | 手臂關節角(rad),來自實際物理模擬後的qpos,不是IK target |
| `gripper_qpos` | 純量 | 夾爪開合量(m), 0=closed 0.04=open。只有pick-and-place版本才有 |
| `tcp_pos` | (3,) | 末端點(TCP)世界座標(x,y,z),MuJoCo物理模擬直接讀出來的,不用自己算FK |
| `tcp_mat` | (9,) | 末端點旋轉矩陣(3×3攤平成9個數字)。如果你要用6D continuity representation
  (Zhou et al. CVPR 2019),直接取前兩欄(6個數字)就好,不用先轉成quaternion再轉回矩陣 |
| `link_pos` | (7,3)攤平成(21,) | 7個手臂連桿(fr3_link1~7)各自的世界座標,對應`q`的7個關節角 |
| `link_mat` | (7,9)攤平成(63,) | 同上,每個連桿的旋轉矩陣(攤平) |
| `action` | (8,)或(7,) | pick-and-place是`[Δq(7), Δgripper_qpos(1)]`,reach是`Δq(7)`。
  最後一行是補齊值(跟倒數第二行重複),不是真實資料,`storage.py`裡有寫這件事,
  訓練時記得去掉最後一行 |

**`meta/traj_XXXX.json`欄位**(整條trajectory固定,不隨timestep變化):
`goal_pos`、`goal_quat`(pick-and-place)、`place_target_pos`(pick-and-place)、
`fps`(存影片時用的真實frame rate,讀取時對照`timestamp`要用這個,不要假設固定30fps)、
`dr_*`(這條trajectory實際套用的domain randomization參數)

## Domain Randomization說明
每條trajectory開始前,`domain_rand.py`會直接改MuJoCo model的runtime array(不用重新compile,
500+條資料生成才不會慢),隨機化以下維度:

**視覺**
- 燈光位置(±0.5m)、diffuse/ambient強度
- 桌面顏色(大範圍,反正只是背景,不影響任務語意)
- 方塊顏色(維持綠色系,只jitter色相深淺——任務定義還是「抓綠色方塊」,不能隨機到任意顏色)
- 相機pose jitter(位置±2cm、角度±3度,模擬真實安裝校準誤差)、視角fovy jitter(±5%)

**動力學**(`--no_dynamics_dr`可以關掉,只留視覺DR)
- 方塊質量(0.7~1.4倍,模擬不同材質/配重的方塊)
- 方塊摩擦係數(0.6~1.3倍)
- 手臂position servo gain(0.85~1.15倍,模擬真實機器人低階控制器不是完美剛性)

**重要實作細節**:因為同一個`model`物件在整個生成過程中重複使用,DR套用前一定要先回到`capture_nominal()`
存下的基準值再疊加新的隨機值,不然像燈光位置這種用疊加方式改的欄位,會隨著trajectory數量增加而越漂越遠。

啟用動力學DR後,抓取成功率會比純視覺DR略降(這批測試從~90%掉到~78%,屬於預期內,因為摩擦力/質量/gain
的變化會讓原本抓得剛好的姿態變得不夠穩)。如果你發現成功率掉太多,先檢查是不是動力學DR的範圍設太寬,
可以在`domain_rand.py`裡把`mass_scale`、`friction_scale`、`gain_scale`的uniform range縮小。

## 干擾物(distractor objects)
`scene.py`裡加了3個真實瓶罐mesh物體(來自robosuite,MIT License):`distractor_bottle`(玻璃瓶)、
`distractor_can`(汽水罐)、`distractor_milk`(牛奶盒/罐)。不是自己組合的primitive幾何——用MjSpec的
`add_mesh`/`add_texture`/`add_material` API把robosuite的mesh+texture資產接進場景,跟真實掃描/建模的
物體外觀一致,比primitive組合更接近真實clutter場景。

因為是mesh物體,形狀/大小在compile時就固定,`distractor_rand.py`每條trajectory重新隨機化:
- **位置**: 跟目標方塊至少距離7cm、跟彼此至少距離6cm(rejection sampling,瓶罐比方塊大所以間距抓寬一點)
- **朝向**: 只繞z軸隨機yaw,物體保持直立站在桌上(用`bottom_site`偏移量確保正確貼著桌面,不懸空不穿模)
- **顏色調性**: 對`geom_rgba`做非綠色隨機色調(乘法疊加在原本的texture material上)。這個是實測後加的
  修正——bottle原本的glass材質偏綠色,容易跟目標方塊搞混,驗證過`geom_rgba`確實會正確套用在mesh material
  上(不會被材質定義蓋掉),所以直接用跟`domain_rand.py`一樣的HSV排除綠色系做法處理

用`--no_distractors`可以關掉,場景回到只有目標方塊。干擾物是static clutter(不會被特別去抓),
訓練時模型要學會忽略它們、只鎖定`goal_pos`指到的目標方塊。

## 隨機化說明
- **位置**:每條trajectory在`WORKSPACE_X/Y`範圍內均勻取樣(x,y),z固定貼著桌面
- **朝向**:繞z軸隨機yaw(0~2π)。**修正**:我原本以為固定世界朝向的夾爪(不管方塊怎麼轉都用同一個垂直
  朝下姿態去夾)就夠用,理由是方塊對角線7.07cm小於夾爪最大開合8cm。實測發現這個假設太樂觀——方塊轉到
  接近45度時,有效寬度逼近8cm,餘裕不到3cm,稍微加入路徑jitter/側向弧度就會讓指尖先撞到方塊邊角把它
  撞飛,抓取成功率從100%掉到不到30%。修正後`pick_task.py`會讓夾爪閉合方向對齊方塊的yaw(mod 90度,
  因為方塊四面對稱),確保任何朝向下都維持~1.5cm的夾持餘裕。
- **動作形狀**:不只是cube位置不同,同一個位置也可能產生不同的動作(呼應「trajectory不要固定」這個考量):
  - pregrasp高度、lift高度隨機(10~20cm / 15~25cm)
  - 各phase步數隨機(18~32步),等於每條demo的速度剖面不一樣
  - 閉合夾爪停留時間隨機(10~20步)
  - 進場(home→pregrasp)路徑加隨機側向弧度(0~4cm),不是死板直線,比較接近人類teleop的路徑
  - 下降階段加極小jitter(≤3mm,精準度優先,不能太大)
  - 舉起階段也加隨機弧度但幅度更小(≤1.5cm)——這個數值是實測調出來的,太大(原本設到3cm)會在
    上升過程中因為側向加速度超過摩擦力上限讓方塊滑出去,細節見下面debug記錄

## Depth渲染的注意事項
- 用MuJoCo內建`Renderer.enable_depth_rendering()`,回傳的是公尺為單位的metric depth(不是normalize過的視差圖)
- 每個timestep每個camera要切換RGB/depth模式各render一次,渲染時間會比純RGB多,`--no_depth`可以關掉
- 存成`float16`省空間(深度精度到mm等級绰绰有餘,不需要float32)

## 已知限制 / 你可能需要調整的地方
- **抓取成功判定**目前只檢查「方塊最終z是否比初始高5cm以上」,沒有檢查抓取過程中是否曾經滑落又
  重新接觸——如果之後發現某些trajectory的動作看起來詭異(例如中途掉落又被檯面反彈頂住),
  建議額外檢查`img`序列目視confirm,或加上「z單調遞增」這個更嚴格條件。
- `WORKSPACE_X/Y`是我隨手抓的範圍(0.40~0.60, -0.18~0.18),沒有根據FR3實際可達工作空間校準,
  你需要根據自己畫的workspace bounds調整,避免生成IK解不出來或造成自碰撞的姿態。
- `steps_per_phase`不再是固定值,改成每條trajectory隨機抽(18~32步),trajectory總長度T會落在
  70~100左右浮動,不是固定91步。`SETTLE_SUBSTEPS=4`(每個waypoint之間實際physics step數)還是固定的。
- 目前每個waypoint都用高gain position servo直接track,沒有加入視覺domain randomization
  (光照、桌面顏色、camera pose擾動),等你要進入Milestone 3(sim-to-real)時才需要加。
- 抓取用的`grasp_quat`只對齊方塊yaw(mod 90度),沒有處理方塊萬一被物理引擎轉出非平躺姿態的情況
  (目前free joint只給z軸初始擾動,重力下應該會穩定平躺,但長時間模擬理論上還是有機率翻動,
  真的出現異常可以在`sample_cube_pose`裡額外檢查最終擺放姿態)。