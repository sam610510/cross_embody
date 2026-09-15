"""
產生一條 FR3 pick-and-place trajectory 的 waypoint 序列:
    home -> pregrasp -> grasp -> close -> lift -> transport -> place -> open -> retreat
每個phase之間用cartesian插值 + per-step IK(warm-start前一步的q)。

重點:每條trajectory的「動作形狀」本身也隨機化(不只是cube/target位置不同而已),包括:
    - pregrasp/lift高度隨機
    - 各phase步數隨機(=速度快慢不同)
    - 閉合/開啟夾爪停留時間隨機
    - 進場(home->pregrasp)、搬運(transport)路徑加隨機側向弧度,不是死板的直線
    - 下降/上升/放下階段加一點點jitter(不能太大,不然會抓不準/放不準)
這樣同一組cube/target位置也可能對應到不太一樣的軌跡,避免BC模型學到「固定速度剖面」這種捷徑。
"""
import numpy as np
import mujoco
from ik import solve_ik, N_ARM_JOINTS

DOWN_QUAT = np.array([0, 1, 0, 0])  # gripper朝下
GRIPPER_OPEN = 255.0
GRIPPER_CLOSE = 0.0


def _slerp(q0, q1, t):
    """quaternion球面線性插值(wxyz格式)。標準slerp公式,處理dot<0的最短路徑修正
    跟dot接近1時退化成線性插值避免數值問題。"""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(np.clip(dot, -1, 1))
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def _interp_ik(model, data, q_start, pos_start, pos_end, quat, n_steps,
                lateral_bump=0.0, jitter=0.0, rng=None, start_quat=None):
    """
    在task space插值,每個中繼點各自解IK,q用前一步warm-start。
    lateral_bump: 路徑中點側向偏移量(m),用sin曲線讓路徑變成弧線而不是直線,只影響phase中段
    jitter: 每個中繼點疊加的小隨機擾動(m),獨立同分布,製造細微的路徑不規則性
    start_quat: 若提供,朝向會從start_quat漸進slerp到quat(整個phase內完成),而不是一開始
                就直接跳到quat。用在「手臂朝向可能跟grasp_quat差很多」的情況(例如隨機化過的
                起始姿態),避免IK在第一步就被迫做出巨大的關節角跳動(實測過:起始姿態隨機化後,
                沒有slerp的話單步dq可以到1.3+ rad,後面追不回來,誤差一路帶到trajectory結束)。
                不提供的話維持舊行為(整個phase固定用quat,適合手臂已經對齊過的中間phase)。
    """
    qs = []
    q_cur = q_start.copy()
    delta = pos_end - pos_start
    # 找一個跟delta垂直的水平方向,當作側向弧線的偏移軸
    if lateral_bump > 0 and np.linalg.norm(delta[:2]) > 1e-6:
        perp = np.array([-delta[1], delta[0], 0])
        perp = perp / (np.linalg.norm(perp) + 1e-8)
    else:
        perp = np.zeros(3)

    for alpha in np.linspace(0, 1, n_steps):
        target = pos_start + alpha * delta
        target = target + perp * lateral_bump * np.sin(np.pi * alpha)  # 中點偏移最大,兩端為0
        if jitter > 0 and rng is not None:
            target = target + rng.uniform(-jitter, jitter, size=3)
        target_quat = _slerp(start_quat, quat, alpha) if start_quat is not None else quat
        q_cur = solve_ik(model, data, target, target_quat, q_cur)
        qs.append(q_cur.copy())
    return qs


def generate_pick_trajectory(model, data, cube_pos, cube_quat, target_pos, q_start, q_home_end, rng):
    """
    rng: np.random.Generator,用來隨機化這條trajectory的動作形狀
    cube_quat: 方塊當前朝向(wxyz),用來讓夾爪閉合方向對齊方塊(mod 90度,因為方塊四面對稱)
    target_pos: 目標方形標記的世界座標中心點(x,y,桌面高度z)
    q_start: 這條trajectory開始時手臂的關節角(每條trajectory隨機化,不是每次都從同一個
             固定姿態開始,避免模型學到「起始畫面長這樣才開始動作」這種捷徑)
    q_home_end: 放置完成後要回到的固定關節角(每條trajectory都一樣,給一個明確、可重複辨識的
                「完成」訊號,不是隨便停在某個retreat高度就算了事)
    回傳一個list,每個元素是 dict(q=7維手臂角, gripper_ctrl=純量, phase=str)
    """
    tcp_id = model.site("tcp").id
    grip_id = model.actuator("hand_actuator8").id
    traj = []

    def record(q, grip_ctrl, phase):
        traj.append({"q": q.copy(), "gripper_ctrl": float(grip_ctrl), "phase": phase})

    # -- 讓夾爪開合方向對齊方塊朝向(mod 90度) --
    cube_yaw = 2 * np.arctan2(cube_quat[3], cube_quat[0])
    grasp_yaw = cube_yaw % (np.pi / 2)
    z_quat = np.array([np.cos(grasp_yaw / 2), 0, 0, np.sin(grasp_yaw / 2)])
    grasp_quat = np.zeros(4)
    mujoco.mju_mulQuat(grasp_quat, z_quat, DOWN_QUAT)

    # -- 隨機化這條trajectory的動作參數 --
    pregrasp_height = rng.uniform(0.10, 0.20)
    lift_height = rng.uniform(0.15, 0.25)
    steps_approach = rng.integers(18, 32)     # start -> pregrasp,速度快慢
    steps_descend = rng.integers(18, 30)      # pregrasp -> grasp
    steps_lift = rng.integers(18, 30)         # grasp -> lift
    steps_transport = rng.integers(20, 36)    # lift -> target上方,通常比其他phase長(水平距離較遠)
    steps_place = rng.integers(15, 26)        # target上方 -> 放下
    steps_retreat = rng.integers(15, 26)      # 放開後撤離
    settle_steps = rng.integers(10, 20)       # 閉合夾爪停留久一點或短一點
    open_steps = rng.integers(10, 18)         # 開啟夾爪(放開方塊)停留時間
    approach_bump = rng.uniform(0.0, 0.04)    # 進場路徑的側向弧度
    descend_jitter = rng.uniform(0.0, 0.003)  # 下降階段微小jitter,不能太大否則影響對準
    lift_bump = rng.uniform(0.0, 0.015)       # 舉起階段的弧度不能太誇張,不然握力不夠會滑掉
    transport_bump = rng.uniform(0.0, 0.05)   # 搬運階段路徑弧度,水平移動距離長,弧度可以大一點
    place_jitter = rng.uniform(0.0, 0.004)    # 放下階段微小jitter

    # -- Phase 0: start(隨機化過的起始姿態,不是固定值) --
    data.qpos[:N_ARM_JOINTS] = q_start
    data.ctrl[grip_id] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    record(q_start, GRIPPER_OPEN, "start")
    start_tcp = data.site_xpos[tcp_id].copy()
    start_mat = data.site_xmat[tcp_id].reshape(3, 3).copy()
    start_quat = np.zeros(4)
    mujoco.mju_mat2Quat(start_quat, start_mat.flatten())

    pregrasp_pos = cube_pos + np.array([0, 0, pregrasp_height])
    grasp_pos = cube_pos.copy()
    lift_pos = cube_pos + np.array([0, 0, lift_height])
    # 搬運終點: target正上方,維持跟lift一樣的高度,再垂直下降到place_pos
    transport_pos = np.array([target_pos[0], target_pos[1], lift_pos[2]])
    # 放下高度: 跟grasp_pos一樣是方塊中心貼著桌面的高度,因為方塊是被剛性夾住搬運,
    # 維持跟原本抓取時一樣的相對高度,才能讓方塊平順地放回桌面
    place_pos = np.array([target_pos[0], target_pos[1], grasp_pos[2]])

    # -- Phase 1: start -> pregrasp (帶隨機側向弧度,近似人類teleop不會走死板直線。
    # 朝向從start_quat漸進slerp到grasp_quat,避免起始姿態隨機化後手臂朝向差很多時,
    # IK第一步就被迫做出巨大關節角跳動) --
    for q in _interp_ik(model, data, q_start, start_tcp, pregrasp_pos, grasp_quat,
                         steps_approach, lateral_bump=approach_bump, rng=rng,
                         start_quat=start_quat):
        record(q, GRIPPER_OPEN, "to_pregrasp")

    # -- Phase 2: pregrasp -> grasp (下降,只加極小jitter,精準度優先) --
    q_pregrasp = traj[-1]["q"]
    for q in _interp_ik(model, data, q_pregrasp, pregrasp_pos, grasp_pos, grasp_quat,
                         steps_descend, jitter=descend_jitter, rng=rng):
        record(q, GRIPPER_OPEN, "descend")

    # -- Phase 3: 閉合夾爪(手臂不動,只改變gripper target,讓接觸力自然限位) --
    q_grasp = traj[-1]["q"]
    for _ in range(settle_steps):
        record(q_grasp, GRIPPER_CLOSE, "close_gripper")

    # -- Phase 4: grasp -> lift (夾住上升,帶隨機側向弧度) --
    for q in _interp_ik(model, data, q_grasp, grasp_pos, lift_pos, grasp_quat,
                         steps_lift, lateral_bump=lift_bump, rng=rng):
        record(q, GRIPPER_CLOSE, "lift")

    # -- Phase 5: lift -> target上方 (水平搬運,帶隨機弧度) --
    q_lift = traj[-1]["q"]
    for q in _interp_ik(model, data, q_lift, lift_pos, transport_pos, grasp_quat,
                         steps_transport, lateral_bump=transport_bump, rng=rng):
        record(q, GRIPPER_CLOSE, "transport")

    # -- Phase 6: target上方 -> 放下高度 (下降,小jitter) --
    q_transport = traj[-1]["q"]
    for q in _interp_ik(model, data, q_transport, transport_pos, place_pos, grasp_quat,
                         steps_place, jitter=place_jitter, rng=rng):
        record(q, GRIPPER_CLOSE, "place_descend")

    # -- Phase 7: 開啟夾爪(手臂不動,放開方塊) --
    q_place = traj[-1]["q"]
    for _ in range(open_steps):
        record(q_place, GRIPPER_OPEN, "release")

    # -- Phase 8: 放開後撤離,垂直上升回到跟lift一樣的高度 --
    for q in _interp_ik(model, data, q_place, place_pos, transport_pos, grasp_quat,
                         steps_retreat, rng=rng):
        record(q, GRIPPER_OPEN, "retreat")

    # -- Phase 9: 撤離後回到固定的完成點(joint space直接線性插值,不用IK——
    # 目標本來就是已知的關節角q_home_end,直接插值保證精確回到同一個點,這個固定終點
    # 才能當作明確的「任務完成」訊號)
    #
    # FR3手腕關節(joint5/6/7)的真實馬達力矩上限只有12Nm(相對joint1-4的87Nm小很多,
    # 這是Franka官方規格,不是bug)。q_retreat跟q_home_end在手腕關節上可能差距很大
    # (實測過可以到2.2+ rad,因為grasp_quat會依cube朝向做mod 90度旋轉),如果只是單純拉長
    # 整段線性插值的步數,插值全程都用同樣慢的速度會讓trajectory變得很長很沒效率——更有效的
    # 做法是插值到位後,額外加一段跟close_gripper/release一樣的靜止停留,讓伺服機構在終點
    # 附近有時間追上真正收斂,不用讓整個過程都被迫變慢。 --
    q_retreat = traj[-1]["q"]
    steps_return = int(np.clip(np.max(np.abs(q_home_end - q_retreat)) / 0.03, 25, 60))
    settle_return_steps = rng.integers(50, 70)  # 拉長settle時間,實測過20~35步殘留誤差還有
                                                  # 0.097rad,純粹hold在q_home_end不動100步的
                                                  # 穩態誤差基準只有0.005rad,代表是settle時間
                                                  # 不夠長,不是伺服機構的物理極限
    for alpha in np.linspace(0, 1, steps_return):
        q_interp = q_retreat + alpha * (q_home_end - q_retreat)
        record(q_interp, GRIPPER_OPEN, "return_home")
    for _ in range(settle_return_steps):
        record(q_home_end, GRIPPER_OPEN, "return_home_settle")

    return traj