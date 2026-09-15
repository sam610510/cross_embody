"""
Domain Randomization,每條trajectory開始前呼叫一次,直接改model的runtime array,
不用重新compile MjSpec(500+條資料才不會慢)。

重要: model物件是整個生成過程共用的同一個,所以每次randomize前要先把相關欄位
還原到capture_nominal()存的基準值,再疊加新的隨機值——不然像light_pos這種用
"+="疊加的欄位,會每條trajectory累積漂移,越跑到後面燈光位置越誇張。

分兩類:
    - 視覺DR: 燈光、桌面顏色、方塊顏色深淺、相機pose jitter、相機視角(fovy)jitter
    - 動力學DR: 方塊質量、摩擦力、手臂position servo gain(模擬真實機器人低階控制器的不完美)

回傳一個dict紀錄這次抽到的參數,存進npz方便之後debug或做domain分層分析。
"""
import numpy as np
import mujoco
import colorsys

COLOR_DISTANCE_THRESHOLD = 0.15  # 低於這個歐氏距離,人眼很難分辨兩種材質的顏色差異
# (實測過:例如dark-wood vs dirt只有0.053、gray-woodgrain vs white-bricks只有0.059,
#  雖然是不同材質但顏色幾乎一樣,畫面上會看不出桌子邊界在哪裡)


def _material_avg_color(model, mat_id):
    """算出一個material實際貼的texture的平均RGB顏色(0~1),直接讀compile好的model資料,
    不用重新開PNG檔。role index 1對應add_material(textures=["", tex_name])裡放texture的那個槽位。"""
    tex_id = model.mat_texid[mat_id][1]
    if tex_id < 0:
        return model.mat_rgba[mat_id, :3].copy()  # 沒貼texture的話退回用材質本身的rgba
    adr = model.tex_adr[tex_id]
    h, w, nch = model.tex_height[tex_id], model.tex_width[tex_id], model.tex_nchannel[tex_id]
    arr = np.asarray(model.tex_data[adr: adr + h * w * nch]).reshape(h, w, nch)
    return arr[..., :3].mean(axis=(0, 1)) / 255.0


def capture_nominal(model, table_geom_names=("table_top", "table_leg_0", "table_leg_1",
                                              "table_leg_2", "table_leg_3"),
                     cube_geom_name="cube_geom",
                     cube_body_name="cube", cam_names=("scene_cam", "wrist_cam"),
                     table_materials=None, floor_geom_name="floor", floor_materials=None):
    """在還沒套用任何randomization之前呼叫一次,存下所有DR會動到的欄位的原始值當基準。
    table_geom_names是複數(桌面板+4支桌腳),材質切換時要一起套用到全部,不然會出現
    桌面跟桌腳材質不一致的怪畫面。"""
    table_ids = [model.geom(name).id for name in table_geom_names]
    cube_id = model.geom(cube_geom_name).id
    cube_body_id = model.body(cube_body_name).id
    cam_ids = [model.camera(c).id for c in cam_names]
    table_mat_ids = ([model.material(f"{name}_mat").id for name in table_materials]
                      if table_materials else None)
    floor_id = model.geom(floor_geom_name).id if floor_materials else None
    floor_mat_ids = ([model.material(f"floor_{name}_mat").id for name in floor_materials]
                      if floor_materials else None)
    table_mat_colors = ({mid: _material_avg_color(model, mid) for mid in table_mat_ids}
                         if table_mat_ids else None)
    floor_mat_colors = ({mid: _material_avg_color(model, mid) for mid in floor_mat_ids}
                         if floor_mat_ids else None)

    return {
        "table_ids": table_ids, "cube_id": cube_id, "cube_body_id": cube_body_id,
        "cam_ids": cam_ids, "cam_names": cam_names, "table_mat_ids": table_mat_ids,
        "floor_id": floor_id, "floor_mat_ids": floor_mat_ids,
        "table_mat_colors": table_mat_colors, "floor_mat_colors": floor_mat_colors,
        "light_pos": model.light_pos.copy(),
        "light_dir": model.light_dir.copy(),
        "cam_pos": model.cam_pos.copy(),
        "cam_quat": model.cam_quat.copy(),
        "cam_fovy": model.cam_fovy.copy(),
        "body_mass": model.body_mass[cube_body_id],
        "geom_friction": model.geom_friction[cube_id].copy(),
        "arm_gainprm0": model.actuator_gainprm[:7, 0].copy(),
        "arm_biasprm1": model.actuator_biasprm[:7, 1].copy(),
    }


def apply_domain_randomization(model, rng, nominal, enable_dynamics_dr=True):
    p = nominal
    params = {}

    # ---------- 視覺: 燈光(從nominal位置往外jitter,不會累積漂移) ----------
    n_light = model.light_pos.shape[0]
    model.light_pos[:] = p["light_pos"] + rng.uniform(-0.5, 0.5, size=(n_light, 3))
    model.light_diffuse[:] = np.clip(rng.uniform(0.4, 0.9, size=(n_light, 3)), 0, 1)
    model.light_ambient[:] = np.clip(rng.uniform(0.1, 0.4, size=(n_light, 3)), 0, 1)
    params["light_pos"] = model.light_pos.copy()
    params["light_diffuse"] = model.light_diffuse.copy()

    # ---------- 視覺: 桌面材質(離散切換,不是連續調色——從compile時就準備好的
    # 木紋/磁磚/灰泥牆材質庫裡隨機選一種,對應到真實世界「桌子是某一種材質」而不是
    # 顏色連續變化) ----------
    if p["table_mat_ids"] is not None:
        chosen_mat_id = rng.choice(p["table_mat_ids"])
        for tid in p["table_ids"]:
            model.geom_matid[tid] = chosen_mat_id
        params["table_mat_id"] = int(chosen_mat_id)

    # ---------- 視覺: 地板材質(離散切換,而且要跟桌面顏色夠不一樣才選,
    # 不然就算材質不同,顏色太接近的話畫面上還是分不出桌子邊界在哪) ----------
    if p["floor_mat_ids"] is not None:
        if p["table_mat_ids"] is not None:
            table_color = p["table_mat_colors"][chosen_mat_id]
            valid_floor_ids = [
                fid for fid in p["floor_mat_ids"]
                if np.linalg.norm(p["floor_mat_colors"][fid] - table_color) >= COLOR_DISTANCE_THRESHOLD
            ]
            if not valid_floor_ids:
                # 極端狀況(理論上不會發生,除非候選材質庫顏色都很接近):
                # 退而求其次,選跟桌面顏色距離最遠的那個
                valid_floor_ids = [max(
                    p["floor_mat_ids"],
                    key=lambda fid: np.linalg.norm(p["floor_mat_colors"][fid] - table_color),
                )]
        else:
            valid_floor_ids = p["floor_mat_ids"]

        chosen_floor_id = rng.choice(valid_floor_ids)
        model.geom_matid[p["floor_id"]] = chosen_floor_id
        params["floor_mat_id"] = int(chosen_floor_id)

    # ---------- 視覺: 方塊顏色(維持綠色語意,只jitter色相深淺) ----------
    cube_rgb = np.array([
        rng.uniform(0.0, 0.25),
        rng.uniform(0.55, 0.95),
        rng.uniform(0.0, 0.25),
    ])
    model.geom_rgba[p["cube_id"], :3] = cube_rgb
    params["cube_rgba"] = cube_rgb.copy()

    # ---------- 視覺: 相機pose jitter(模擬真實安裝時的校準誤差,從nominal pose疊加) ----------
    for cam_id in p["cam_ids"]:
        pos_jitter = rng.uniform(-0.02, 0.02, size=3)
        model.cam_pos[cam_id] = p["cam_pos"][cam_id] + pos_jitter

        angle_jitter = np.deg2rad(rng.uniform(-3, 3, size=3))
        jitter_quat = np.zeros(4)
        mujoco.mju_euler2Quat(jitter_quat, angle_jitter, "xyz")
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, jitter_quat, p["cam_quat"][cam_id])
        model.cam_quat[cam_id] = new_quat

        model.cam_fovy[cam_id] = p["cam_fovy"][cam_id] * rng.uniform(0.95, 1.05)
        params[f"cam{cam_id}_pos_jitter"] = pos_jitter.copy()

    # ---------- 動力學: 方塊質量 + 摩擦力(從nominal值縮放) ----------
    if enable_dynamics_dr:
        mass_scale = rng.uniform(0.7, 1.4)
        model.body_mass[p["cube_body_id"]] = p["body_mass"] * mass_scale
        params["cube_mass"] = float(model.body_mass[p["cube_body_id"]])

        friction_scale = rng.uniform(0.6, 1.3)
        model.geom_friction[p["cube_id"]] = p["geom_friction"] * friction_scale
        params["cube_friction"] = float(model.geom_friction[p["cube_id"], 0])

        # ---------- 動力學: 手臂position servo gain(模擬真實控制器不是完美剛性) ----------
        gain_scale = rng.uniform(0.85, 1.15)
        model.actuator_gainprm[:7, 0] = p["arm_gainprm0"] * gain_scale
        model.actuator_biasprm[:7, 1] = p["arm_biasprm1"] * gain_scale
        params["arm_gain_scale"] = float(gain_scale)

    return params