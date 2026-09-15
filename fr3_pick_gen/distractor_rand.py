"""
每條trajectory開始前呼叫一次,把干擾物(真實瓶罐mesh: bottle/can/milk,來自robosuite)
隨機擺到桌面上。

跟之前primitive組合版不同,mesh物體的形狀/大小是compile時就固定的(mesh vertex資料不在
這個腳本的randomize範圍內),這裡隨機化:
    - 位置: workspace範圍內,跟目標方塊、跟彼此都保持最小間距(rejection sampling)
    - 朝向: 只繞z軸隨機yaw(物體保持直立站在桌上,不會東倒西歪,比較接近真實場景)
    - 顏色調性: 實測發現bottle原本的glass材質偏綠色,容易跟目標方塊搞混,所以對geom_rgba
      做非綠色的隨機色調(乘法疊加在材質上,實測確認MuJoCo會正確套用,不是被material蓋掉)

每個物體的mesh原點到底部的偏移量(BOTTOM_OFFSETS)取自robosuite原始XML裡的`bottom_site`
定義,用來讓物體正確「站」在桌面上,不會懸空或穿模。
"""
import numpy as np
import colorsys

DISTRACTOR_NAMES = ["distractor_bottle", "distractor_can", "distractor_milk"]
BOTTOM_OFFSETS = {"distractor_bottle": 0.082, "distractor_can": 0.06, "distractor_milk": 0.085}

MIN_CLEARANCE_FROM_CUBE = 0.07   # 瓶罐比方塊大,任兩個「重要位置」(cube/target/其他干擾物)
                                   # 之間都至少要距離這麼遠


def _random_non_green_rgb(rng):
    """在HSV色輪上排除綠色系(90~150度hue),避免干擾物顏色跟目標方塊搞混。
    值偏亮(0.7~1.0)是因為這是乘法tint疊在material上,太暗會讓整個材質變得死黑看不出紋理。"""
    hue = rng.uniform(0, 1)
    while 0.25 < hue < 0.42:
        hue = rng.uniform(0, 1)
    sat = rng.uniform(0.3, 0.7)
    val = rng.uniform(0.7, 1.0)
    return np.array(colorsys.hsv_to_rgb(hue, sat, val))


def _sample_xy(rng, workspace_x, workspace_y, existing_xy, min_clearance, max_tries=50):
    for _ in range(max_tries):
        xy = np.array([rng.uniform(*workspace_x), rng.uniform(*workspace_y)])
        if all(np.linalg.norm(xy - other) > min_clearance for other in existing_xy):
            return xy
    return xy  # 找不到夠遠的位置就將就用最後一次抽到的(機率很低,workspace通常夠大)


def randomize_distractors(model, data, rng, cube_pos, workspace_x, workspace_y,
                           table_top_z=0.4, target_pos=None):
    placed_xy = [cube_pos[:2]]  # 先把cube位置放進去,確保干擾物離它夠遠
    if target_pos is not None:
        placed_xy.append(target_pos[:2])  # 目標方形位置也要避開,不然干擾物可能卡在放置區裡

    for name in DISTRACTOR_NAMES:
        qpos_addr = model.jnt_qposadr[model.joint(f"{name}_joint").id]
        geom_id = model.geom(f"{name}_geom").id

        xy = _sample_xy(rng, workspace_x, workspace_y, placed_xy, MIN_CLEARANCE_FROM_CUBE)
        placed_xy.append(xy)

        z = table_top_z + BOTTOM_OFFSETS[name]
        yaw = rng.uniform(0, 2 * np.pi)
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

        data.qpos[qpos_addr:qpos_addr + 3] = [xy[0], xy[1], z]
        data.qpos[qpos_addr + 3:qpos_addr + 7] = quat

        model.geom_rgba[geom_id, :3] = _random_non_green_rgb(rng)