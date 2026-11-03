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


# ---------------------------------------------------------------- MJCF robots (e.g. MuJoCo Menagerie)
def resolve_path(p, project_root=""):
    """'{menagerie}/...' -> the menagerie directory used by build_scene.py (or $MENAGERIE_DIR)"""
    if "{menagerie}" in p:
        mdir = os.environ.get("MENAGERIE_DIR")
        if not mdir:
            sys.path.insert(0, os.path.join(project_root, "fr3_pick_gen", "scripts"))
            from build_scene import MENAGERIE_DIR as mdir
        p = p.replace("{menagerie}", str(mdir))
    elif project_root and not os.path.isabs(p):
        p = os.path.join(project_root, p)
    return p


def build_mjcf_robot_spec(cfg, project_root=""):
    """The robot as an MjSpec: arm MJCF (+ gripper MJCF attached at a site) + a fixed body 'tcp' at the
    gripper's TCP site. The SAME function builds the kinematic chain of the FK encoder and the physics
    model of the evaluations, so the two cannot disagree."""
    import mujoco
    arm = mujoco.MjSpec.from_file(resolve_path(cfg["mjcf"], project_root))
    for k in list(arm.keys):
        arm.delete(k)
    g = cfg.get("gripper")
    if g:
        grip = mujoco.MjSpec.from_file(resolve_path(g["mjcf"], project_root))
        for k in list(grip.keys):
            grip.delete(k)
        # the TCP: a fixed body where the gripper's TCP site is (found on the compiled gripper)
        gm = grip.compile()
        sid = mujoco.mj_name2id(gm, mujoco.mjtObj.mjOBJ_SITE, g["tcp_site"])
        if sid < 0:
            raise KeyError(f"site '{g['tcp_site']}' not in {g['mjcf']}")
        parent = mujoco.mj_id2name(gm, mujoco.mjtObj.mjOBJ_BODY, gm.site_bodyid[sid])
        grip.body(parent).add_body(name="tcp", pos=gm.site_pos[sid].tolist(), quat=gm.site_quat[sid].tolist())
        # the gripper file sets its own solver options (impratio, cone); the arm's are kept on attach --
        # copy them over so attach() does not warn about the conflict every time the robot is built
        try:
            grip.option.impratio = arm.option.impratio
            grip.option.cone = arm.option.cone
        except Exception:
            pass
        arm.attach(grip, prefix=g.get("prefix", "grip_"), site=arm.site(g["attach_site"]))
        # attach() prefixes the gripper's names: the TCP body is now prefix + "tcp"
    return arm


def _kinematics_xml(spec):
    """A minimal MJCF with the same body tree and joints, written from the COMPILED model (so default
    classes, meshes, materials and textures are already resolved and simply left out). pytorch_kinematics
    only needs bodies (pos/quat) and joints (axis/pos/range)."""
    import mujoco
    import numpy as np
    m = spec.compile()
    name = lambda t, i, pre: mujoco.mj_id2name(m, t, i) or f"{pre}{i}"
    kids = {}
    for b in range(1, m.nbody):
        kids.setdefault(int(m.body_parentid[b]), []).append(b)
    f = lambda v: " ".join(f"{x:.9g}" for x in v)

    def body_xml(b, ind):
        bname = name(mujoco.mjtObj.mjOBJ_BODY, b, "b")
        js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b
              and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
        jpos = m.jnt_pos[js[0]] if js else np.zeros(3)
        out = []
        if js and np.linalg.norm(jpos) > 1e-12:
            # pytorch_kinematics rotates about the joint position without shifting back, so a joint
            # that is not at the body origin is put in an outer body AT the joint, and the original
            # body (same name, same frame as in MuJoCo) hangs under it at -jpos
            shift = np.zeros(3); mujoco.mju_rotVecQuat(shift, jpos, m.body_quat[b])
            out.append(f'{ind}<body name="{bname}__joint" pos="{f(m.body_pos[b] + shift)}" quat="{f(m.body_quat[b])}">')
            out.append(joint_xml(js[0], ind + "  ", at_origin=True))
            out.append(f'{ind}  <body name="{bname}" pos="{f(-jpos)}" quat="1 0 0 0">')
            inner = ind + "    "; close = [f"{ind}  </body>", f"{ind}</body>"]
        else:
            out.append(f'{ind}<body name="{bname}" pos="{f(m.body_pos[b])}" quat="{f(m.body_quat[b])}">')
            if js:
                out.append(joint_xml(js[0], ind + "  ", at_origin=True))
            inner = ind + "  "; close = [f"{ind}</body>"]
        out.append(f'{inner}<geom type="sphere" size="0.005" contype="0" conaffinity="0" mass="0.01"/>')
        for c in kids.get(b, []):
            out += body_xml(c, inner)
        return out + close

    def joint_xml(j, ind, at_origin):
        jt = int(m.jnt_type[j])
        typ = "hinge" if jt == int(mujoco.mjtJoint.mjJNT_HINGE) else "slide"
        rng = f' range="{f(m.jnt_range[j])}" limited="true"' if m.jnt_limited[j] else ""
        return (f'{ind}<joint name="{name(mujoco.mjtObj.mjOBJ_JOINT, j, "j")}" type="{typ}" '
                f'axis="{f(m.jnt_axis[j])}" pos="0 0 0"{rng}/>')

    lines = ['<mujoco model="kinematics">', '  <compiler angle="radian"/>', "  <worldbody>"]
    for b in kids.get(0, []):
        lines += body_xml(b, "    ")
    lines += ["  </worldbody>", "</mujoco>"]
    return "\n".join(lines)


def _patch_pk_mjcf():
    """newer MuJoCo returns joint types as numpy ints; pytorch_kinematics' map is keyed by the enum"""
    try:
        import mujoco
        from pytorch_kinematics import mjcf as _m

        class _AnyIntKey(dict):
            def __missing__(self, k):
                return dict.__getitem__(self, mujoco.mjtJoint(int(k)))

        if not isinstance(_m.JOINT_TYPE_MAP, _AnyIntKey.__mro__[0]) or type(_m.JOINT_TYPE_MAP) is dict:
            _m.JOINT_TYPE_MAP = _AnyIntKey(_m.JOINT_TYPE_MAP)
    except Exception:
        pass


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
        self.cfg = cfg

        self.arm_dof = int(cfg["arm_dof"])
        self.key_links = dict(cfg["key_links"])      # shoulder/elbow/wrist/ee -> link (body) name
        self.root_link = cfg.get("root_link", None)

        R = cfg.get("base_rotation", [[1,0,0],[0,1,0],[0,0,1]])
        self.base_R = torch.tensor(R, device=device, dtype=dtype)
        # optional fixed rotation of the TCP frame (right-multiplied), so that every robot's TCP frame
        # follows the SAME physical convention as the source robot (approach axis, finger-closing axis)
        self.ee_R = torch.tensor(cfg.get("ee_rotation", [[1,0,0],[0,1,0],[0,0,1]]), device=device, dtype=dtype)

        if "mjcf" in cfg:                            # MuJoCo model (e.g. Menagerie), optional gripper
            spec = build_mjcf_robot_spec(cfg, project_root)
            if cfg.get("gripper") and self.key_links.get("ee") == "tcp":
                self.key_links["ee"] = cfg["gripper"].get("prefix", "grip_") + "tcp"
            _patch_pk_mjcf()
            with _suppress_stderr():
                self.chain = pk.build_chain_from_mjcf(_kinematics_xml(spec)).to(device=device, dtype=dtype)
        else:
            urdf_path = cfg["urdf"]
            if project_root:
                urdf_path = os.path.join(project_root, urdf_path)
            with open(urdf_path, "rb") as f:
                urdf_bytes = f.read()
            with _suppress_stderr():
                self.chain = pk.build_chain_from_urdf(urdf_bytes).to(device=device, dtype=dtype)

        names = self.chain.get_joint_parameter_names()
        self.n_actuated = len(names)
        # the arm joints must come first (the gripper's joints, if any, are padded with 0)
        arm_names = cfg.get("arm_joints")
        if arm_names and list(names[:self.arm_dof]) != list(arm_names):
            raise ValueError(f"arm joints are not the first {self.arm_dof} chain joints: {names}")

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
                Rm = self.base_R.unsqueeze(0) @ Rm @ self.ee_R.unsqueeze(0)
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