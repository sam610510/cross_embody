"""
Milestone 1用: 純reaching trajectory,不抓不放,只有手臂移動到目標位置。
動作序列: start(隨機起始姿態) -> approach(移到目標上方一點,朝向漸進slerp過去)
         -> reach(精確對準目標) -> hold(停留,給一個明確的「到達」訊號)
         -> return_home(回到固定完成點) -> settle

跟pick_task.py共用同一套_interp_ik/_slerp邏輯(這裡直接re-export,不重複實作),
差別只在動作序列簡化很多、沒有夾爪動作。
"""
import numpy as np
import mujoco
from ik import N_ARM_JOINTS
from pick_task import _interp_ik, DOWN_QUAT, GRIPPER_OPEN


def generate_reach_trajectory(model, data, target_pos, q_start, q_home_end, rng,
                               approach_quat=None, hold_steps=None, observe_steps=None):
    """
    target_pos: 要伸手過去的世界座標點(x,y,z)
    q_start: 這條trajectory的起始關節角(隨機化過的,見generate_reach_dataset.py)
    q_home_end: 完成後要回到的固定關節角
    approach_quat: 手臂靠近目標時的朝向,預設DOWN_QUAT(垂直向下,呼應pick task的習慣,
                   之後如果要無縫接軌到抓取任務,這個朝向假設可以維持一致)
    hold_steps: 到位後停留幾個waypoint。不指定的話用預設隨機範圍(15~25,沒有對應到
                精確的物理時間);想要停留精確秒數的話,外部算好對應步數傳進來——見
                generate_reach_dataset.py的HOLD_DURATION_SEC,換算方式是
                hold_steps = 目標秒數 / (SETTLE_SUBSTEPS * model.opt.timestep)
    observe_steps: 一開始在高處停留幾個waypoint才開始移動,模擬「先看到目標才動作」
                   這個時序結構(scripted trajectory沒辦法真的做到感知觸發,但至少讓
                   demo資料呈現「一開始不動是正常的」這個pattern,而不是每條trajectory
                   一開始就立刻動作,對training data的時序分佈比較寫實)。不指定的話用
                   隨機範圍(見下方預設值)
    回傳一個list,每個元素是 dict(q=7維手臂角, gripper_ctrl=GRIPPER_OPEN固定值, phase=str)
    """
    if approach_quat is None:
        approach_quat = DOWN_QUAT

    tcp_id = model.site("tcp").id
    traj = []

    def record(q, phase):
        traj.append({"q": q.copy(), "gripper_ctrl": GRIPPER_OPEN, "phase": phase})

    # -- 隨機化動作參數(跟pick_task.py同樣的邏輯,避免固定速度剖面這種捷徑) --
    hover_height = rng.uniform(0.08, 0.18)     # 先到目標正上方多高的地方
    steps_approach = rng.integers(20, 36)      # start -> 目標上方
    steps_reach = rng.integers(15, 26)         # 目標上方 -> 精確到位
    if hold_steps is None:
        hold_steps = rng.integers(15, 25)      # 沒指定的話用舊的隨機範圍(沒有物理時間意義)
    if observe_steps is None:
        observe_steps = rng.integers(20, 60)   # 沒指定的話,約0.16~0.48秒(4×0.002s/step)
    approach_bump = rng.uniform(0.0, 0.05)
    reach_jitter = rng.uniform(0.0, 0.003)

    # -- Phase 0: start --
    data.qpos[:N_ARM_JOINTS] = q_start
    mujoco.mj_forward(model, data)
    record(q_start, "start")
    start_tcp = data.site_xpos[tcp_id].copy()
    start_mat = data.site_xmat[tcp_id].reshape(3, 3).copy()
    start_quat = np.zeros(4)
    mujoco.mju_mat2Quat(start_quat, start_mat.flatten())

    # -- Phase 0.5: 在高處停留不動,模擬「先看到目標才開始動作」的時序結構 --
    for _ in range(observe_steps):
        record(q_start, "observe")

    hover_pos = target_pos + np.array([0, 0, hover_height])

    # -- Phase 1: start -> 目標上方(朝向漸進slerp,原因見pick_task.py同樣的debug紀錄:
    # 起始姿態隨機化後,朝向差太多會讓IK單步跳動過大追不回來) --
    for q in _interp_ik(model, data, q_start, start_tcp, hover_pos, approach_quat,
                         steps_approach, lateral_bump=approach_bump, rng=rng,
                         start_quat=start_quat):
        record(q, "approach")

    # -- Phase 2: 目標上方 -> 精確到位(小jitter,精準度優先) --
    q_hover = traj[-1]["q"]
    for q in _interp_ik(model, data, q_hover, hover_pos, target_pos, approach_quat,
                         steps_reach, jitter=reach_jitter, rng=rng):
        record(q, "reach")

    # -- Phase 3: 到位後停留,手臂不動 --
    q_reach = traj[-1]["q"]
    for _ in range(hold_steps):
        record(q_reach, "hold")

    # -- Phase 4: 先垂直上升(task space,沿用hover_height的高度),避免直接在joint space
    # 插值回home時手臂路徑不可預期地橫掃過桌面/物體。這段結束後才切換成joint space插值。 --
    lift_off_pos = target_pos + np.array([0, 0, hover_height])
    steps_lift_off = rng.integers(15, 26)
    for q in _interp_ik(model, data, q_reach, target_pos, lift_off_pos, approach_quat,
                         steps_lift_off, rng=rng):
        record(q, "lift_off")

    # -- Phase 5: 回到固定完成點(joint space線性插值+settle,做法照抄pick_task.py
    # 驗證過的版本:純插值到最後一步結尾誤差還有0.85rad,要額外加settle時間才會真的收斂) --
    q_lift_off = traj[-1]["q"]
    steps_return = int(np.clip(np.max(np.abs(q_home_end - q_lift_off)) / 0.03, 25, 60))
    settle_return_steps = rng.integers(50, 70)
    for alpha in np.linspace(0, 1, steps_return):
        q_interp = q_lift_off + alpha * (q_home_end - q_lift_off)
        record(q_interp, "return_home")
    for _ in range(settle_return_steps):
        record(q_home_end, "return_home_settle")

    return traj