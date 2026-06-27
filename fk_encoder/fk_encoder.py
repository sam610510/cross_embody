"""
fk_encoder.py — Layer 1: Forward Kinematics Encoder (non-learnable)

设计原则:程式不认机器人,只认设定。
  - 加新手臂 = robots.yaml 加 entry + 放 URDF,本档零修改。
  - 纯几何,无可学习参数。
  - 所有座标以各机器人 base (root_link) 为参考原点。

输出 19 维: [ p_ee(7) | v_kp(9) | m(3) ]
  p_ee  : EE position(3) + quaternion(4)          [姿态相关,每次 forward 计算]
  v_kp  : 三个 keyvector (v_SE, v_SW, v_SEE),各3维 [姿态相关],以 upper-arm L 正规化
  m     : [arm_ratio, workspace_r, dof_norm]        [结构常数,init 算一次]

keyvector 定义 (以 shoulder S 为锚):
  v_SE  = elbow    - shoulder
  v_SW  = wrist    - shoulder
  v_SEE = ee       - shoulder
均除以 L = |elbow - shoulder| (q=0 时的 upper-arm 长度,结构常数)。

m 定义:
  arm_ratio   = |elbow-shoulder| / |wrist-elbow|       (上臂/前臂比, q=0)
  workspace_r = Σ 相邻 key-frame 距离 (base→EE 最大触及半径, q=0)
  dof_norm    = arm_dof / 6
"""

import os
import sys
import contextlib

import yaml
import torch
import pytorch_kinematics as pk


@contextlib.contextmanager
def _suppress_stderr():
    """暂时静音 stderr。urdf_parser_py 会对 Franka 特有标签
    (dynamics D/K、visual name 等) 直接写 stderr 警告,无害但吵。"""
    devnull = open(os.devnull, "w")
    old = sys.stderr
    try:
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old
        devnull.close()


class FKEncoder:
    OUTPUT_DIM = 19
    DOF_NORM_BASE = 6.0

    def __init__(self, robot_name, robots_yaml, project_root="", device="cpu", dtype=torch.float32):
        self.robot_name = robot_name
        self.device = device
        self.dtype = dtype

        with open(robots_yaml, "r") as f:
            registry = yaml.safe_load(f)
        if robot_name not in registry:
            raise KeyError(f"'{robot_name}' not in {robots_yaml}. Available: {list(registry)}")
        cfg = registry[robot_name]

        self.arm_dof = int(cfg["arm_dof"])
        self.key_links = cfg["key_links"]            # shoulder/elbow/wrist/ee -> link name
        self.root_link = cfg.get("root_link", None)

        # base_rotation: 固定旋转,把 base 轴向转到统一约定 (默认单位矩阵)
        R = cfg.get("base_rotation", [[1,0,0],[0,1,0],[0,0,1]])
        self.base_R = torch.tensor(R, device=device, dtype=dtype)

        # 建链
        urdf_path = cfg["urdf"]
        if project_root:
            urdf_path = os.path.join(project_root, urdf_path)
        with open(urdf_path, "rb") as f:
            urdf_bytes = f.read()
        with _suppress_stderr():
            self.chain = pk.build_chain_from_urdf(urdf_bytes).to(device=device, dtype=dtype)

        # actuated joints 总数 (可能含夹爪/finger,> arm_dof);forward 时补 0
        self.n_actuated = len(self.chain.get_joint_parameter_names())

        # 预算结构常数 m,与正规化用的 L
        self._precompute_structural()

    # ---------- internal ----------
    def _full_q(self, q_arm):
        """把 arm 关节 (arm_dof) 补成 forward 需要的完整 actuated 向量 (其余补 0)。"""
        B = q_arm.shape[0]
        if self.n_actuated == self.arm_dof:
            return q_arm
        pad = torch.zeros(B, self.n_actuated - self.arm_dof, device=self.device, dtype=self.dtype)
        return torch.cat([q_arm, pad], dim=1)

    def _key_frames(self, q_arm):
        """回传 shoulder/elbow/wrist/ee 的 (positions, ee_quaternion),已套 base_rotation。
        positions: dict role -> (B,3);ee_quat: (B,4)。"""
        full = self._full_q(q_arm)
        ret = self.chain.forward_kinematics(full)   # dict: link -> Transform3d
        pos, ee_quat = {}, None
        for role, link in self.key_links.items():
            M = ret[link].get_matrix()              # (B,4,4)
            p = M[:, :3, 3]                          # (B,3)
            p = p @ self.base_R.T                    # 套统一 base 轴向
            pos[role] = p
            if role == "ee":
                Rm = M[:, :3, :3]                    # (B,3,3)
                Rm = self.base_R.unsqueeze(0) @ Rm   # 旋转也对齐
                ee_quat = pk.matrix_to_quaternion(Rm)  # (B,4) wxyz
        return pos, ee_quat

    def _precompute_structural(self):
        """在 q=0 算 L、arm_ratio、workspace_r、dof_norm。皆结构常数。"""
        q0 = torch.zeros(1, self.arm_dof, device=self.device, dtype=self.dtype)
        pos, _ = self._key_frames(q0)
        S, E, W, EE = pos["shoulder"][0], pos["elbow"][0], pos["wrist"][0], pos["ee"][0]

        self.L = torch.norm(E - S).clamp_min(1e-6)           # upper-arm 长度 (正规化用)
        forearm = torch.norm(W - E).clamp_min(1e-6)
        arm_ratio = (self.L / forearm).item()

        # workspace_r: base→shoulder→elbow→wrist→ee 相邻距离总和 (最大触及半径)
        base0 = torch.zeros(3, device=self.device, dtype=self.dtype)
        seg = (torch.norm(S - base0) + torch.norm(E - S)
               + torch.norm(W - E) + torch.norm(EE - W))
        workspace_r = seg.item()

        dof_norm = self.arm_dof / self.DOF_NORM_BASE

        self.m = torch.tensor([arm_ratio, workspace_r, dof_norm],
                              device=self.device, dtype=self.dtype)  # (3,)

    # ---------- public ----------
    def forward(self, q_arm):
        """q_arm: (B, arm_dof) 或 (arm_dof,)。回传 (B,19)。"""
        if q_arm.dim() == 1:
            q_arm = q_arm.unsqueeze(0)
        q_arm = q_arm.to(device=self.device, dtype=self.dtype)
        B = q_arm.shape[0]

        pos, ee_quat = self._key_frames(q_arm)
        S, E, W, EE = pos["shoulder"], pos["elbow"], pos["wrist"], pos["ee"]

        # p_ee: position(3) + quaternion(4)
        p_ee = torch.cat([EE, ee_quat], dim=1)                # (B,7)

        # v_kp: 三个 keyvector,以 L 正规化
        v_SE  = (E  - S) / self.L
        v_SW  = (W  - S) / self.L
        v_SEE = (EE - S) / self.L
        v_kp = torch.cat([v_SE, v_SW, v_SEE], dim=1)          # (B,9)

        # m: 结构常数,broadcast 到 batch
        m = self.m.unsqueeze(0).expand(B, -1)                 # (B,3)

        out = torch.cat([p_ee, v_kp, m], dim=1)               # (B,19)
        assert out.shape[1] == self.OUTPUT_DIM
        return out

    __call__ = forward
