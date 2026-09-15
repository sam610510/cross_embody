import os
import sys
import contextlib

import yaml
import torch
import pytorch_kinematics as pk


@contextlib.contextmanager
def _suppress_stderr():
    devnull = open(os.devnull, "w")
    old = sys.stderr
    try:
        sys.stderr = devnull
        yield
    finally:
        sys.stderr = old
        devnull.close()


class FKEncoder:
    OUTPUT_DIM = 21
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

        R = cfg.get("base_rotation", [[1,0,0],[0,1,0],[0,0,1]])
        self.base_R = torch.tensor(R, device=device, dtype=dtype)

        urdf_path = cfg["urdf"]
        if project_root:
            urdf_path = os.path.join(project_root, urdf_path)
        with open(urdf_path, "rb") as f:
            urdf_bytes = f.read()
        with _suppress_stderr():
            self.chain = pk.build_chain_from_urdf(urdf_bytes).to(device=device, dtype=dtype)

        self.n_actuated = len(self.chain.get_joint_parameter_names())

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
        full = self._full_q(q_arm)
        ret = self.chain.forward_kinematics(full)   # dict: link -> Transform3d
        pos, ee_rot6d = {}, None
        for role, link in self.key_links.items():
            M = ret[link].get_matrix()              # (B,4,4)
            p = M[:, :3, 3]                          # (B,3)
            p = p @ self.base_R.T                    # 套统一 base 轴向
            pos[role] = p
            if role == "ee":
                Rm = M[:, :3, :3]                   
                Rm = self.base_R.unsqueeze(0) @ Rm   
                col0 = Rm[:, :, 0]                   
                col1 = Rm[:, :, 1]                   
                ee_rot6d = torch.cat([col0, col1], dim=1)  
        return pos, ee_rot6d

    def _precompute_structural(self):
        q0 = torch.zeros(1, self.arm_dof, device=self.device, dtype=self.dtype)
        pos, _ = self._key_frames(q0)
        S, E, W, EE = pos["shoulder"][0], pos["elbow"][0], pos["wrist"][0], pos["ee"][0]

        self.L = torch.norm(E - S).clamp_min(1e-6)           
        forearm = torch.norm(W - E).clamp_min(1e-6)
        arm_ratio = (self.L / forearm).item()

        base0 = torch.zeros(3, device=self.device, dtype=self.dtype)
        seg = (torch.norm(S - base0) + torch.norm(E - S)
               + torch.norm(W - E) + torch.norm(EE - W))
        workspace_r = seg.item()

        dof_norm = self.arm_dof / self.DOF_NORM_BASE

        self.m = torch.tensor([arm_ratio, workspace_r, dof_norm],
                              device=self.device, dtype=self.dtype)  # (3,)


    def forward(self, q_arm):
        if q_arm.dim() == 1:
            q_arm = q_arm.unsqueeze(0)
        q_arm = q_arm.to(device=self.device, dtype=self.dtype)
        B = q_arm.shape[0]

        pos, ee_rot6d = self._key_frames(q_arm)
        S, E, W, EE = pos["shoulder"], pos["elbow"], pos["wrist"], pos["ee"]

        # p_ee: position(3) + 6D rotation(6)
        p_ee = torch.cat([EE, ee_rot6d], dim=1)               # (B,9)

        # v_kp
        v_SE  = (E  - S) / self.L
        v_SW  = (W  - S) / self.L
        v_SEE = (EE - S) / self.L
        v_kp = torch.cat([v_SE, v_SW, v_SEE], dim=1)          # (B,9)

        # morphological features
        m = self.m.unsqueeze(0).expand(B, -1)                 # (B,3)

        out = torch.cat([p_ee, v_kp, m], dim=1)               # (B,21)
        assert out.shape[1] == self.OUTPUT_DIM
        return out

    __call__ = forward