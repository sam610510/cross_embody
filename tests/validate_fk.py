"""
validate_fk.py — FK Encoder (Layer 1) 验证脚本
=================================================
对你的 fk_encoder.FKEncoder 做七项验证, ground-truth 来源 = MuJoCo。

涵盖:
  (1) p_ee 对拍 MuJoCo FK            -> 数值
  (2) 随机取样误差 histogram          -> 数值 + 图
  (3) v_kp 尺度不变性 / q=0 sanity    -> 数值
  (4) m 合理性检查                    -> 数值
  (5) 3D keyvector 叠图 (单姿态)      -> 图
  (6) 轨迹动画 (扫过一条 trajectory)  -> gif
  (7) 跨机器人 p_ee 散点 (workspace)  -> 图 (FR3 + 任一可选第二台)

用法:
    conda activate cross_embody
    cd ~/cross_embody
    python validate_fk.py

输出:
    validation_out/  下的 png / gif / 一份 report.txt

=== 你需要确认/填写的地方, 都集中在下面 CONFIG ===
重点: MuJoCo model 路径, 以及 4 个 key frame 在 MuJoCo 里的 body/site 名称。
"""

import os
import sys
import contextlib
import numpy as np
import torch

# ----------------------------------------------------------------------
# CONFIG —— 只有这一段需要你按环境改
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.expanduser("~/cross_embody")
ROBOTS_YAML  = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
ROBOT_NAME   = "fr3"
ARM_DOF      = 7

OUT_DIR = os.path.join(PROJECT_ROOT, "validation_out")

# --- MuJoCo Franka model ---
# 用 build_fr3_with_hand.py 生成的带夹爪 model, TCP (fr3_hand_tcp) 已对齐 encoder。
MJCF_PATH = os.path.expanduser("~/cross_embody/urdfs/franka_description/fr3_with_hand.xml")

# 带夹爪 model: EE 直接对拍 fr3_hand_tcp (已验证 q=0 误差 0.0mm)。
MJ_FRAMES = {
    "shoulder": ("body", "fr3_link2"),
    "elbow":    ("body", "fr3_link4"),
    "wrist":    ("body", "fr3_link6"),
    "ee":       ("site", "fr3_hand_tcp"),   # 带夹爪 model 的 TCP, 对齐 encoder p_ee
}
MJ_BASE_BODY = "base"

# 带夹爪 model 已可直接对拍 TCP, 不需 flange 绕路。保留 FLANGE_LINK 仅供参考。
FLANGE_LINK = "fr3_link7"

# 随机取样数量 (点 2)
N_RANDOM = 5000
# 轨迹动画帧数 (点 6)
N_TRAJ = 60

# ----------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)
os.makedirs(OUT_DIR, exist_ok=True)


# ====== util: 把你的 FKEncoder import 进来 ======
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder  # noqa: E402


def print_mujoco_names():
    """先跑这个, 把 body/site 清单贴给我 (或自己对照填 MJ_FRAMES)。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    print("=== BODIES ===")
    print([m.body(i).name for i in range(m.nbody)])
    print("=== SITES ===")
    print([m.site(i).name for i in range(m.nsite)])
    print("=== JOINTS (actuated order) ===")
    print([m.joint(i).name for i in range(m.njnt)])


# ====== MuJoCo ground-truth FK ======
class MujocoFK:
    """给一组 arm 关节角, 回传各 key frame 在 *base frame* 下的 pose。"""
    def __init__(self, mjcf_path, frames, base_body, arm_dof):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.frames = frames
        self.base_body = base_body
        self.arm_dof = arm_dof
        # 找手臂关节的 qpos 地址。FR3 手臂为 7 个 hinge; 夹爪 finger 为 slide。
        # 优先用名称匹配 fr3_joint1..N (最稳); 退而求其次取前 arm_dof 个 hinge。
        self.arm_qadr = []
        named = []
        for i in range(self.model.njnt):
            jn = self.model.joint(i).name
            if jn and jn.startswith("fr3_joint"):
                named.append((jn, self.model.jnt_qposadr[i]))
        if len(named) >= arm_dof:
            named.sort(key=lambda x: x[0])   # fr3_joint1..7 顺序
            self.arm_qadr = [adr for _, adr in named[:arm_dof]]
        else:
            # fallback: 取前 arm_dof 个 hinge joint (排除 slide/finger)
            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == self.mj.mjtJoint.mjJNT_HINGE:
                    self.arm_qadr.append(self.model.jnt_qposadr[i])
                if len(self.arm_qadr) >= arm_dof:
                    break

    def _get_pose(self, ftype, fname):
        """回传 (pos(3), R(3,3)) in world frame。"""
        if ftype == "body":
            bid = self.model.body(fname).id
            pos = self.data.xpos[bid].copy()
            R = self.data.xmat[bid].reshape(3, 3).copy()
        elif ftype == "site":
            sid = self.model.site(fname).id
            pos = self.data.site_xpos[sid].copy()
            R = self.data.site_xmat[sid].reshape(3, 3).copy()
        else:
            raise ValueError(ftype)
        return pos, R

    def fk(self, q_arm):
        """q_arm: (arm_dof,) numpy。回传 dict role -> (pos_base(3), R_base(3,3))。"""
        self.data.qpos[:] = 0.0
        for k, adr in enumerate(self.arm_qadr):
            self.data.qpos[adr] = q_arm[k]
        self.mj.mj_forward(self.model, self.data)

        # base world pose
        bp, bR = self._get_pose("body", self.base_body)
        out = {}
        for role, (ftype, fname) in self.frames.items():
            p, R = self._get_pose(ftype, fname)
            # world -> base:  p_base = bR^T (p - bp)
            p_base = bR.T @ (p - bp)
            R_base = bR.T @ R
            out[role] = (p_base, R_base)
        return out


# ====== helpers ======
def quat_geodesic_deg(q1_wxyz, q2_wxyz):
    """两 quaternion (wxyz) 间的最短测地角 (deg), 处理 q ~ -q 双覆盖。"""
    q1 = q1_wxyz / np.linalg.norm(q1_wxyz)
    q2 = q2_wxyz / np.linalg.norm(q2_wxyz)
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    return np.degrees(2.0 * np.arccos(d))


def random_q(n, arm_dof, lo=-2.8, hi=2.8):
    return np.random.uniform(lo, hi, size=(n, arm_dof)).astype(np.float32)


def enc_link_pose(enc, link_name, q_arm):
    """从 encoder 的 chain 取任意 link 在 base frame 的 (pos(3), quat_wxyz(4))。
    用来抓 flange(link7) 与 MuJoCo 的 attachment_site 对拍 (两边都是 flange)。"""
    qt = torch.from_numpy(q_arm.astype(np.float32)).unsqueeze(0)
    full = enc._full_q(qt.to(enc.device, enc.dtype))
    ret = enc.chain.forward_kinematics(full)
    M = ret[link_name].get_matrix()          # (1,4,4)
    p = (M[:, :3, 3] @ enc.base_R.T)[0].numpy()
    Rm = (enc.base_R.unsqueeze(0) @ M[:, :3, :3])
    import pytorch_kinematics as pk
    q = pk.matrix_to_quaternion(Rm)[0].numpy()
    return p, q


# ====================================================================
# 主流程
# ====================================================================
def main():
    report = []

    def log(s=""):
        print(s)
        report.append(str(s))

    enc = FKEncoder(ROBOT_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT)
    log(f"[init] FKEncoder({ROBOT_NAME}) loaded. L={float(enc.L):.6f}  m={enc.m.tolist()}")

    try:
        gt = MujocoFK(MJCF_PATH, MJ_FRAMES, MJ_BASE_BODY, ARM_DOF)
        log(f"[init] MuJoCo model loaded from {MJCF_PATH}")
        have_mj = True
    except Exception as e:
        log(f"[WARN] MuJoCo init failed: {e}")
        log("       -> 数值对拍(1,2)与跨机器人(7)需要 MuJoCo; 先跑 print_mujoco_names() 确认 frame 名称。")
        have_mj = False

    # ---------------- (1) p_ee (TCP) 直接对拍 ----------------
    if have_mj:
        log("\n=== (1) p_ee (fr3_hand_tcp) vs MuJoCo (带夹爪 model, 6D 旋转) ===")
        for q_test, tag in [(np.zeros(ARM_DOF, dtype=np.float32), "q=0"),
                            (np.array([0.2,-0.3,0.1,-1.5,0.2,1.0,-0.1], dtype=np.float32)[:ARM_DOF], "random pose")]:
            enc_out = enc.forward(torch.from_numpy(q_test)).detach().numpy()[0]
            p_enc, rot6d_enc = enc_out[:3], enc_out[3:9]      # 6D 旋转
            g = gt.fk(q_test)
            p_gt = enc.base_R.numpy() @ g["ee"][0]
            R_gt = enc.base_R.numpy() @ g["ee"][1]
            rot6d_gt = _R_to_6d(R_gt)
            pos_err = np.linalg.norm(p_enc - p_gt) * 1000
            rot_err = _rot6d_geodesic_deg(rot6d_enc, rot6d_gt)
            log(f"  [{tag}] TCP pos err = {pos_err:.4f} mm | rot err = {rot_err:.4f} deg")
            log(f"         encoder={np.round(p_enc,5)}  mujoco={np.round(p_gt,5)}")

    # ---------------- (2) 随机取样误差 histogram ----------------
    if have_mj:
        log(f"\n=== (2) random sampling error histogram (N={N_RANDOM}) ===")
        log("    shoulder/elbow/wrist + TCP 全部对拍 (带夹爪 model, TCP 已对齐)。")
        Q = random_q(N_RANDOM, ARM_DOF)
        roles = ["shoulder", "elbow", "wrist"]
        pos_errs = {r: [] for r in roles}
        tcp_pos_err, tcp_rot_err = [], []
        enc_out_all = enc.forward(torch.from_numpy(Q)).detach().numpy()
        for i in range(N_RANDOM):
            g = gt.fk(Q[i])
            kf, _ = enc._key_frames(torch.from_numpy(Q[i]).unsqueeze(0))
            for r in roles:
                p_enc = kf[r][0].numpy()
                p_gt = enc.base_R.numpy() @ g[r][0]
                pos_errs[r].append(np.linalg.norm(p_enc - p_gt) * 1000)
            # TCP 直接对拍
            p_gt_tcp = enc.base_R.numpy() @ g["ee"][0]
            R_gt_tcp = enc.base_R.numpy() @ g["ee"][1]
            tcp_pos_err.append(np.linalg.norm(enc_out_all[i, :3] - p_gt_tcp) * 1000)
            tcp_rot_err.append(_rot6d_geodesic_deg(enc_out_all[i, 3:9], _R_to_6d(R_gt_tcp)))
        for r in roles:
            arr = np.array(pos_errs[r])
            log(f"  {r:9s} pos err (mm): mean={arr.mean():.5f}  p99={np.percentile(arr,99):.5f}  max={arr.max():.5f}")
        ta = np.array(tcp_pos_err); tr = np.array(tcp_rot_err)
        log(f"  TCP       pos err (mm): mean={ta.mean():.5f}  p99={np.percentile(ta,99):.5f}  max={ta.max():.5f}")
        log(f"  TCP       rot err (deg): mean={tr.mean():.5f}  p99={np.percentile(tr,99):.5f}  max={tr.max():.5f}")
        _plot_hist_multi(pos_errs, ta, tr, os.path.join(OUT_DIR, "02_error_hist.png"))
        log(f"  -> saved 02_error_hist.png")

    # ---------------- (3) v_kp 尺度不变性 + q=0 sanity ----------------
    log("\n=== (3) v_kp invariance & sanity ===")
    q0t = torch.zeros(ARM_DOF)
    out0 = enc.forward(q0t).detach().numpy()[0]
    v_kp0 = out0[9:18]   # 6D 布局: p_ee(0:9) | v_kp(9:18) | m(18:21)
    log(f"  v_kp @ q=0 = {np.round(v_kp0,4)}")
    # v_SE 在 q=0 应为单位长 (因除以 L=|E-S|)
    v_SE = v_kp0[0:3]
    log(f"  |v_SE| @ q=0 = {np.linalg.norm(v_SE):.6f}  (期望 ~1.0)")
    log("  (尺度不变性: 需另建一份 link 等比例放大的 URDF 跑同样 q, 比对 v_kp 应不变 — 见 report 说明)")

    # ---------------- (4) m 合理性 ----------------
    log("\n=== (4) m sanity ===")
    arm_ratio, workspace_r, dof_norm = enc.m.tolist()
    log(f"  arm_ratio   = {arm_ratio:.4f}")
    log(f"  workspace_r = {workspace_r:.4f} m   (FR3 触及半径量级应 ~0.8-1.0)")
    log(f"  dof_norm    = {dof_norm:.4f}   (期望 7/6 = {7/6:.4f})")
    assert abs(dof_norm - 7/6) < 1e-6, "dof_norm 应为 7/6"

    # ---------------- (5) 3D keyvector 叠图 ----------------
    log("\n=== (5) 3D keyvector overlay ===")
    _plot_keyvectors(enc, q_arm=np.zeros(ARM_DOF, dtype=np.float32),
                     path=os.path.join(OUT_DIR, "05_keyvectors_q0.png"))
    # 再画一个非零姿态
    q_demo = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float32)[:ARM_DOF]
    _plot_keyvectors(enc, q_arm=q_demo,
                     path=os.path.join(OUT_DIR, "05_keyvectors_pose.png"))
    log("  -> saved 05_keyvectors_q0.png / 05_keyvectors_pose.png")

    # ---------------- (6) 轨迹动画 ----------------
    log("\n=== (6) trajectory animation ===")
    try:
        _animate_trajectory(enc, ARM_DOF, N_TRAJ,
                            path=os.path.join(OUT_DIR, "06_trajectory.gif"))
        log("  -> saved 06_trajectory.gif")
    except Exception as e:
        log(f"  [skip] animation failed ({e}); 需 pillow: pip install pillow")

    # ---------------- (7) 跨机器人 workspace 散点 ----------------
    log("\n=== (7) cross-robot p_ee scatter ===")
    Q = random_q(2000, ARM_DOF)
    P = enc.forward(torch.from_numpy(Q)).detach().numpy()[:, :3]
    _plot_workspace({ROBOT_NAME: P}, os.path.join(OUT_DIR, "07_workspace.png"))
    log("  -> saved 07_workspace.png  (加入第二台机器人后会自动并排比较)")

    # ---------------- write report ----------------
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(report))
    log(f"\n[done] outputs in {OUT_DIR}")


# ====== plotting helpers ======
def _R_to_quat_wxyz(R):
    """3x3 -> quaternion wxyz (numpy)."""
    import pytorch_kinematics as pk
    q = pk.matrix_to_quaternion(torch.from_numpy(R).unsqueeze(0).float())
    return q[0].numpy()


def _R_to_6d(R):
    """3x3 旋转矩阵 -> 6D (前两列, 与 fk_encoder 一致)。"""
    return np.concatenate([R[:, 0], R[:, 1]])


def _6d_to_R(r6):
    """6D -> 3x3, Gram-Schmidt 还原 (Zhou 2019)。"""
    a1 = r6[:3]; a2 = r6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _rot6d_geodesic_deg(r6_a, r6_b):
    """两个 6D 旋转间的测地角 (deg), 经还原成 R 后算 relative rotation。"""
    Ra = _6d_to_R(np.asarray(r6_a)); Rb = _6d_to_R(np.asarray(r6_b))
    Rrel = Ra.T @ Rb
    cos = (np.trace(Rrel) - 1) / 2
    cos = min(1.0, max(-1.0, cos))
    return np.degrees(np.arccos(cos))


def _plot_hist_multi(pos_errs_dict, tcp_pos, tcp_rot, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    roles = list(pos_errs_dict.keys())
    n = len(roles) + 2
    # 加宽每个子图, 给旋转标签留空间
    fig, ax = plt.subplots(1, n, figsize=(3.8 * n, 4.0))
    colors = ["#2F5C9E", "#6B4C7A", "#B5651D"]

    mm_fmt = FormatStrFormatter("%.5f")   # 统一 5 位小数

    def _style(a, xlabel, fmt):
        a.set_xlabel(xlabel, fontsize=10)
        a.set_ylabel("Number of poses (count)", fontsize=10)   # Y 轴名称
        a.xaxis.set_major_formatter(fmt)                        # 统一小数位数
        a.xaxis.set_major_locator(MaxNLocator(nbins=3))         # 只留 3 个刻度, 避免挤
        a.tick_params(axis="x", labelrotation=45, labelsize=8)
        a.tick_params(axis="y", labelsize=8)
        # x 轴标签右对齐, 旋转后不会压到隔壁
        for lbl in a.get_xticklabels():
            lbl.set_horizontalalignment("right")
        a.grid(True, axis="y", linestyle="--", alpha=0.4)

    for k, r in enumerate(roles):
        ax[k].hist(np.array(pos_errs_dict[r]), bins=50, color=colors[k % 3])
        ax[k].set_title(f"{r} pos err", fontsize=11, fontweight="bold")
        _style(ax[k], "Position error (mm)", mm_fmt)

    ax[len(roles)].hist(tcp_pos, bins=50, color="#2C6B3F")
    ax[len(roles)].set_title("TCP pos err", fontsize=11, fontweight="bold")
    _style(ax[len(roles)], "Position error (mm)", mm_fmt)

    ax[len(roles) + 1].hist(tcp_rot, bins=50, color="#B03030")
    ax[len(roles) + 1].set_title("TCP rot err", fontsize=11, fontweight="bold")
    _style(ax[len(roles) + 1], "Rotation error (deg)", mm_fmt)

    # 子图间留更多水平间距
    fig.subplots_adjust(wspace=0.45, bottom=0.22)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_keyvectors(enc, q_arm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    qt = torch.from_numpy(q_arm)

    # 取所有 link 的 frame 位置 (完整骨架, 不只 4 个 key point)
    full = enc._full_q(qt.unsqueeze(0).to(enc.device, enc.dtype))
    ret = enc.chain.forward_kinematics(full)
    all_links = []
    for lname, T in ret.items():
        p = (T.get_matrix()[:, :3, 3] @ enc.base_R.T)[0].numpy()
        all_links.append((lname, p))

    pos, _ = enc._key_frames(qt.unsqueeze(0))
    S = pos["shoulder"][0].numpy(); E = pos["elbow"][0].numpy()
    W = pos["wrist"][0].numpy();    EE = pos["ee"][0].numpy()
    BASE = np.zeros(3)

    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection="3d")

    # 完整骨架: 依 chain 顺序连线 (用所有 link 原点)。先抓主臂链 link0..link7/tcp 的顺序
    main_order = [n for n in ["fr3_link0","fr3_link1","fr3_link2","fr3_link3",
                              "fr3_link4","fr3_link5","fr3_link6","fr3_link7",
                              "fr3_link8","fr3_hand","fr3_hand_tcp"] ]
    link_map = {n: p for n, p in all_links}
    chain_pts = [BASE] + [link_map[n] for n in main_order if n in link_map]
    chain_pts = np.array(chain_pts)
    ax.plot(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2],
            "-", color="black", linewidth=4.0, zorder=2, label="arm links")

    # 所有关节点 (每个 link 原点) 都标出来
    for lname, p in all_links:
        if lname in main_order:
            ax.scatter(p[0], p[1], p[2], color="#222", s=35, zorder=3)

    # base
    ax.scatter([0],[0],[0], color="black", marker="s", s=90, zorder=4, label="base (0,0,0)")

    # 四个 key frame 用大点 + 旁边标注
    keypts = [(S,"S (shoulder)","#2F5C9E"), (E,"E (elbow)","#2F5C9E"),
              (W,"W (wrist)","#6B4C7A"), (EE,"EE (ee-pose)","#B5651D")]
    for p, name, c in keypts:
        ax.scatter(p[0], p[1], p[2], color=c, s=110, zorder=6, edgecolors="white", linewidths=1.5)
        ax.text(p[0]+0.02, p[1]+0.02, p[2], name, fontsize=11, fontweight="bold", color=c, zorder=7)

    # keyvectors from shoulder
    for vec, c, name in [(E - S, "#2F5C9E", "v_SE"), (W - S, "#6B4C7A", "v_SW"), (EE - S, "#B5651D", "v_SEE")]:
        ax.quiver(S[0], S[1], S[2], vec[0], vec[1], vec[2], color=c,
                  arrow_length_ratio=0.1, linewidth=2.5, alpha=0.9)

    _set_equal_3d(ax, np.vstack([BASE, chain_pts]))
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("FK full skeleton + keyvectors (base frame)", fontsize=13, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _set_equal_3d(ax, pts):
    xmin, ymin, zmin = pts.min(0); xmax, ymax, zmax = pts.max(0)
    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    r = max(xmax-xmin, ymax-ymin, zmax-zmin) / 2
    r = max(r, 0.1)
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)


def _animate_trajectory(enc, arm_dof, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    t = np.linspace(0, 2 * np.pi, n)
    # 以展示姿态为中心小幅摆动 (与 render_mujoco 的 gif 一致, 便于并排对比)
    demo = np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.2, 0.0], dtype=np.float32)[:arm_dof]
    traj = np.stack([demo[j] + 0.4 * np.sin(t + j) for j in range(arm_dof)], axis=1).astype(np.float32)

    main_order = ["fr3_link0","fr3_link1","fr3_link2","fr3_link3","fr3_link4",
                  "fr3_link5","fr3_link6","fr3_link7","fr3_link8","fr3_hand","fr3_hand_tcp"]

    def frames_of(q):
        full = enc._full_q(torch.from_numpy(q).unsqueeze(0).to(enc.device, enc.dtype))
        ret = enc.chain.forward_kinematics(full)
        lm = {}
        for lname, T in ret.items():
            lm[lname] = (T.get_matrix()[:, :3, 3] @ enc.base_R.T)[0].numpy()
        pos, _ = enc._key_frames(torch.from_numpy(q).unsqueeze(0))
        S = pos["shoulder"][0].numpy(); E = pos["elbow"][0].numpy()
        W = pos["wrist"][0].numpy();    EE = pos["ee"][0].numpy()
        chain = np.array([np.zeros(3)] + [lm[k] for k in main_order if k in lm])
        return chain, S, E, W, EE

    # 预扫求固定等比例轴范围
    allpts = []
    cache = []
    for i in range(n):
        c = frames_of(traj[i]); cache.append(c)
        allpts.append(c[0])
    allpts = np.vstack(allpts)
    xmin, ymin, zmin = allpts.min(0); xmax, ymax, zmax = allpts.max(0)
    cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
    r = max(xmax-xmin, ymax-ymin, zmax-zmin)/2; r = max(r, 0.1)

    fig = plt.figure(figsize=(8, 8)); ax = fig.add_subplot(111, projection="3d")

    def frame(i):
        ax.cla()
        chain, S, E, W, EE = cache[i]
        # 完整骨架: 黑色粗线 + 所有关节点
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], "-", color="black", linewidth=4.0, zorder=2)
        ax.scatter(chain[1:, 0], chain[1:, 1], chain[1:, 2], color="#222", s=30, zorder=3)
        ax.scatter([0],[0],[0], color="black", marker="s", s=80, zorder=4)
        # 四个 key frame 大点 + 旁注
        for p, name, c in [(S,"S",  "#2F5C9E"), (E,"E","#2F5C9E"),
                           (W,"W","#6B4C7A"), (EE,"EE-pose","#B5651D")]:
            ax.scatter(p[0], p[1], p[2], color=c, s=90, zorder=6, edgecolors="white", linewidths=1.2)
            ax.text(p[0]+0.02, p[1]+0.02, p[2], name, fontsize=10, fontweight="bold", color=c, zorder=7)
        # keyvectors
        for vec, c in [(E - S, "#2F5C9E"), (W - S, "#6B4C7A"), (EE - S, "#B5651D")]:
            ax.quiver(S[0], S[1], S[2], vec[0], vec[1], vec[2], color=c,
                      arrow_length_ratio=0.1, linewidth=2.2, alpha=0.9)
        ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=18, azim=-60)
        ax.set_title(f"FK skeleton + keyvectors  (frame {i+1}/{n})", fontsize=12, fontweight="bold")

    ani = FuncAnimation(fig, frame, frames=n, interval=60)
    ani.save(path, writer=PillowWriter(fps=15)); plt.close(fig)


def _plot_workspace(robot_pts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    n = len(robot_pts)
    fig = plt.figure(figsize=(6 * n, 6))
    colors = ["#6B4C7A", "#2C6B3F", "#B5651D"]
    for k, (name, P) in enumerate(robot_pts.items()):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=3, alpha=0.4, color=colors[k % 3])
        ax.set_title(f"{name} EE workspace"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "names":
        print_mujoco_names()
    else:
        main()