"""
build_scene(): 組裝 FR3 手臂 + Panda夾爪 + 桌子 + 綠色方塊 + 真實瓶罐mesh干擾物 的 MuJoCo 場景
用 MjSpec API 把手臂跟夾爪接在一起,而不是手動改XML,好處是menagerie更新時比較不會壞掉。

干擾物用 robosuite (MIT License, ARISE Initiative/Stanford) 的物體mesh資產,
不是自己組合primitive幾何——bottle/can/milk都是真實掃描/建模的瓶罐,比primitive組合更接近真實場景。
"""
import mujoco
import numpy as np
import os

CUBE_SIZE = 0.025  # half-size, 邊長5cm = 0.05m, MuJoCo box size是半邊長
TABLE_HEIGHT = 0.4       # 桌面高度(離地)
TABLE_TOP_THICKNESS = 0.04   # 桌面板厚度4cm
TABLE_LEG_HALF = 0.03        # 桌腳半邊長(方形截面,6cm x 6cm)
TABLE_HALF_X = 0.5       # 桌面板x方向半邊長(footprint 1.0m)。比原本(0.4)加大,
                          # 讓基座能離workspace保持足夠距離(IK收斂需要)同時還在桌面範圍內
                          # ——這兩個需求原本用原尺寸桌子調不出兩全的位置,細節見debug紀錄
TABLE_HALF_Y = 0.4       # 桌面板y方向半邊長
TABLE_LEG_INSET = 0.05   # 桌腳從桌面板邊緣往內縮多少(避免桌腳突出桌面邊緣)
ROBOT_MOUNT_POS = [0.22, 0.0, TABLE_HEIGHT]  # 手臂基座掛載位置: 實測掃描過0.20~0.28範圍,
                                               # 0.22是IK在整個workspace收斂最好的位置
                                               # (四個角落+中心點誤差都在1e-4以下)
# (實測過base的碰撞幾何體本身有寬度,x=0.15時最遠處會伸出桌緣(0.1)快8公分,
#  x=0.25時最近邊界還留2.3cm餘裕,不會懸空在桌子外)
TCP_OFFSET_Z = 0.1034  # panda hand flange到指尖中心的官方偏移量(m),見上面驗證過程

TARGET_SQUARE_SIZE = 0.07   # 目標方形外框邊長7cm
TAPE_WIDTH = 0.006          # 膠帶寬度6mm
TAPE_THICKNESS = 0.001      # 膠帶厚度1mm,貼在桌面上基本上是平的

# 桌面材質候選(來自robosuite texture資產,MIT License),只挑寫實的木紋/灰白色系,
# 不用它們家帶顏色的木紋(blue/green/pink/yellow/red-wood)避免不寫實。
# 每條trajectory從這裡面離散挑一種材質(不是連續調色),在domain_rand.py裡實作。
# 注意: 跟FLOOR_MATERIALS完全不重疊(原本混進了light-gray-floor-tile,跟地板材質重複,
# 萬一兩邊剛好抽到同一種材質,桌面跟地板會分不出邊界,已經移除)。
TABLE_MATERIALS = [
    "light-wood", "dark-wood", "wood-varnished-panels",
    "gray-woodgrain", "gray-plaster", "white-plaster",
    "light-gray-plaster", "cream-plaster",
]

# 地板材質候選(同樣來自robosuite texture資產),跟桌面材質分開隨機,
# 避免地板跟桌面永遠是同一種材質看起來太巧合
FLOOR_MATERIALS = ["light-gray-floor-tile", "white-bricks", "wood-tiles", "dirt"]

# 干擾物: 從robosuite借來的真實瓶罐mesh(MIT License)。每個物體是
# (物體名, mesh檔相對路徑, texture檔相對路徑, material反射率, mesh scale)
# 路徑相對於 robosuite_path/robosuite/models/assets/
DISTRACTOR_OBJECTS = [
    {"name": "distractor_bottle", "mesh": "objects/meshes/bottle.stl",
     "texture": "textures/glass.png", "reflectance": 0.5, "scale": 1.0},
    {"name": "distractor_can", "mesh": "objects/meshes/can.stl",
     "texture": "textures/soda.png", "reflectance": 0.7, "scale": 1.0},
    {"name": "distractor_milk", "mesh": "objects/meshes/milk.stl",
     "texture": "textures/ceramic.png", "reflectance": 0.5, "scale": 0.9},
]


def _add_mesh_distractor(spec, world, obj, robosuite_path, pos):
    """把一個robosuite物體(mesh+texture)加進場景,包成free-floating body。"""
    name = obj["name"]
    assets_dir = f"{robosuite_path}/robosuite/models/assets"

    spec.add_mesh(name=f"{name}_mesh", file=f"{assets_dir}/{obj['mesh']}",
                  scale=[obj["scale"]] * 3)
    spec.add_texture(name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      file=f"{assets_dir}/{obj['texture']}")
    spec.add_material(name=f"{name}_mat", textures=["", f"{name}_tex"],
                       reflectance=obj["reflectance"], texrepeat=[3, 3], texuniform=True)

    body = world.add_body(name=name, pos=pos)
    body.add_freejoint(name=f"{name}_joint")
    body.add_geom(
        name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{name}_mesh", material=f"{name}_mat",
        mass=0.03, friction=[0.8, 0.005, 0.0001],
    )
    return body


def _add_target_marker(world, table_top_z):
    """黑色膠帶貼成的空心正方形,用mocap body承載,每條trajectory直接改
    data.mocap_pos就能移動位置,不用加joint也不用重新compile。
    4段細長box組成正方形輪廓(picture-frame方式,四邊各自沿邊長中心擺放)。"""
    half = TARGET_SQUARE_SIZE / 2
    marker = world.add_body(name="target_marker", mocap=True,
                             pos=[0.5, 0.0, table_top_z + TAPE_THICKNESS / 2])
    segments = [
        ("top",    [half, TAPE_WIDTH / 2, TAPE_THICKNESS / 2], [0,  half - TAPE_WIDTH/2, 0]),
        ("bottom", [half, TAPE_WIDTH / 2, TAPE_THICKNESS / 2], [0, -half + TAPE_WIDTH/2, 0]),
        ("left",   [TAPE_WIDTH / 2, half, TAPE_THICKNESS / 2], [-half + TAPE_WIDTH/2, 0, 0]),
        ("right",  [TAPE_WIDTH / 2, half, TAPE_THICKNESS / 2], [ half - TAPE_WIDTH/2, 0, 0]),
    ]
    for seg_name, size, pos in segments:
        marker.add_geom(
            name=f"target_tape_{seg_name}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size, pos=pos, rgba=[0.05, 0.05, 0.05, 1],
            contype=0, conaffinity=0,  # 只做視覺標記,不參與物理碰撞
        )
    return marker


def build_spec(menagerie_path: str, robosuite_path: str) -> mujoco.MjSpec:
    # 用絕對路徑,避免MuJoCo內部meshdir/相對路徑合併邏輯把".."處理錯誤
    # (相對路徑在這裡曾經實測會被錯誤解析,詳見generate_dataset.py使用時的debug紀錄)
    menagerie_path = os.path.abspath(menagerie_path)
    robosuite_path = os.path.abspath(robosuite_path)

    arm = mujoco.MjSpec.from_file(f"{menagerie_path}/franka_fr3/fr3.xml")
    hand = mujoco.MjSpec.from_file(f"{menagerie_path}/franka_emika_panda/hand.xml")

    # 手臂基座掛載到桌面邊緣(貼著桌面高度),不是floor level的世界原點——
    # 原本base在(0,0,0),桌面卻在z=0.4高度,手臂看起來像懸空在桌子旁邊而不是鎖在桌上
    arm.body("base").pos = ROBOT_MOUNT_POS

    attach_site = arm.site("attachment_site")
    hand_body = arm.attach(hand, site=attach_site, prefix="hand_")

    # D415原生解析度(1920x1080)超過MuJoCo預設的離屏渲染緩衝區大小(640x480),
    # 要在這裡宣告夠大的offwidth/offheight,不然renderer建立時會直接報錯
    arm.visual.global_.offwidth = 1920
    arm.visual.global_.offheight = 1080

    # skybox: 填滿桌子/地板以外的空間(原本沒定義會被MuJoCo渲染成純黑,這是很明顯的
    # 「這是模擬環境」線索,對sim-to-real的vision encoder遷移不利)。用程序生成的
    # 漸層,不需要額外檔案。
    arm.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.55, 0.6, 0.65], rgb2=[0.15, 0.15, 0.18], width=256, height=256,
    )

    # 地板: 比桌子大很多的平面,鋪在地面(z=0),材質候選在compile時全部加進去,
    # runtime用跟桌面一樣的geom_matid切換手法隨機換材質
    for mat_name in FLOOR_MATERIALS:
        tex_file = f"{robosuite_path}/robosuite/models/assets/textures/{mat_name}.png"
        arm.add_texture(name=f"floor_{mat_name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=tex_file)
        arm.add_material(name=f"floor_{mat_name}_mat", textures=["", f"floor_{mat_name}_tex"],
                          texrepeat=[8, 8], texuniform=True, reflectance=0.05)

    # 加TCP site,IK跟資料紀錄都用這個當end-effector參考點
    tcp_site = hand_body.add_site(name="tcp", pos=[0, 0, TCP_OFFSET_Z])

    # 加桌子(材質候選在compile時全部加進去,runtime再用geom_matid切換,
    # 不用重新compile就能每條trajectory換一種材質)
    world = arm.worldbody
    for mat_name in TABLE_MATERIALS:
        tex_file = f"{robosuite_path}/robosuite/models/assets/textures/{mat_name}.png"
        arm.add_texture(name=f"{mat_name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         file=tex_file)
        arm.add_material(name=f"{mat_name}_mat", textures=["", f"{mat_name}_tex"],
                          texrepeat=[4, 4], texuniform=True, reflectance=0.1)

    world.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_TOP_THICKNESS / 2],
        pos=[0.5, 0, TABLE_HEIGHT - TABLE_TOP_THICKNESS / 2],
        material=f"{TABLE_MATERIALS[0]}_mat",
    )
    leg_z = (TABLE_HEIGHT - TABLE_TOP_THICKNESS) / 2
    leg_x = 0.5 - (TABLE_HALF_X - TABLE_LEG_INSET), 0.5 + (TABLE_HALF_X - TABLE_LEG_INSET)
    leg_y = -(TABLE_HALF_Y - TABLE_LEG_INSET), (TABLE_HALF_Y - TABLE_LEG_INSET)
    for i, (lx, ly) in enumerate([(leg_x[0], leg_y[0]), (leg_x[0], leg_y[1]),
                                    (leg_x[1], leg_y[0]), (leg_x[1], leg_y[1])]):
        world.add_geom(
            name=f"table_leg_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[TABLE_LEG_HALF, TABLE_LEG_HALF, leg_z],
            pos=[lx, ly, leg_z],
            material=f"{TABLE_MATERIALS[0]}_mat",
        )

    # 地板平面(比桌子大很多,鋪滿相機視野範圍以外的地面,避免黑色背景)。
    # 只做視覺用,不參與物理碰撞(contype/conaffinity=0),避免影響已經驗證過的抓取動力學
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[4, 4, 0.01], pos=[0.5, 0, 0],
        material=f"floor_{FLOOR_MATERIALS[0]}_mat",
        contype=0, conaffinity=0,
    )

    # 加目標方形標記(黑色膠帶空心正方形,mocap body,位置由generate_dataset.py每條
    # trajectory設定)
    _add_target_marker(world, table_top_z=TABLE_HEIGHT)

    # 加綠色方塊(free joint,才能被抓起來移動)
    cube_body = world.add_body(name="cube", pos=[0.5, 0.0, TABLE_HEIGHT + CUBE_SIZE])
    cube_body.add_freejoint(name="cube_joint")
    cube_body.add_geom(
        name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE], rgba=[0.1, 0.8, 0.1, 1],
        mass=0.05, friction=[1.0, 0.005, 0.0001],
    )

    # 加干擾物(真實瓶罐mesh,位置/朝向之後runtime隨機化)
    for i, obj in enumerate(DISTRACTOR_OBJECTS):
        _add_mesh_distractor(arm, world, obj, robosuite_path,
                              pos=[0.5, 0.3 + i * 0.1, TABLE_HEIGHT + 0.1])

    # 加相機: third-person view,回到桌邊、離地110cm的貼近架設(不強求整隻手臂入鏡,
    # 你只需要相對距離資訊,近距離反而讓目標物件在畫面裡的比例更大更清楚)。
    # fovy=42.5度是D415官方RGB鏡頭垂直視角(水平69.4°/垂直42.5°/對角77°,16:9)
    # + 夾爪上的wrist camera
    world.add_camera(
        name="scene_cam", pos=[0.95, 0.0, 1.10], xyaxes=[0, 1, 0, -0.768, 0, 0.640],
        fovy=42.5,
    )
    hand_body.add_camera(
        name="wrist_cam", pos=[0, 0, 0.03], xyaxes=[1, 0, 0, 0, -1, 0], fovy=60,
    )

    # 光源
    world.add_light(name="top_light", pos=[0.5, 0, 2.0],
                     type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)

    return arm


def build_model(menagerie_path: str, robosuite_path: str):
    spec = build_spec(menagerie_path, robosuite_path)
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data