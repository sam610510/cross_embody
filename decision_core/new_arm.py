"""
The arm side of the cross-arm (zero-shot) evaluation: ANY arm of configs/robots.yaml under the FR3-trained vision core.
Everything arm-specific comes from the robots.yaml entry (nothing is written for one arm here):

  arm_dof, key_links {shoulder, elbow, wrist}       joints = the first arm_dof joints of the arm (model order),
                                                    elbow-up test / table clearance of the start pose
  gripper.prefix        (default grip_)             gripper bodies / actuator
  gripper.finger_bodies (default: *left_pad / *right_pad)   finger distance -> Franka-equivalent finger value
  gripper.open_ctrl     (low | high, default low)   which end of the gripper actuator's ctrlrange is open
  ik_seed               (optional, arm_dof values)  first IK seed when mapping the FR3 start pose (default: 0)
  branch_joints         (optional, joint indices)   joints whose sign must not change while running (branch lock);
                                                    absent = no lock

Pipeline per control step (as scripts/eval_pp_ur5e.py --encoder posture_dict, the protocol's UR5e set-up):
  arm joints --posture dictionary (Stage 2, contrastive)--> FR3 pose --FR3 FK + W_FR3--> latent      (arm history)
  finger distance -> Franka-equivalent finger value (half the gap, 0..0.04 m)                       (finger history)
  core latent --virtual FR3 reads it (inverse decode)--> TCP intent --(anchor blend)--> the arm's IK   (action)
  gripper logit -> gripper open / closed
Nothing here is trained; the decision core and the image encoder are the FR3 ones, unchanged.
"""
import os
import re
import sys

import mujoco
import numpy as np
import torch
import torch.nn as nn

R = os.path.expanduser("~/cross_embody")
for _p in (R, os.path.join(R, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREFIX, MOUNT, TCP_SITE = "arm_", "arm_mount", "arm_tcp_site"      # as build_scene_arm.py
ALIASES = {"ur5e": "ur5e_menagerie"}


def robot_entry(name):
    return ALIASES.get(name, name)


def scene_flags(cfg):
    """gen_vision_data.Scene flags for the scene of build_scene_arm.py"""
    gp = (cfg.get("gripper") or {}).get("prefix", "grip_")
    return ["--scene_module", "build_scene_arm", "--robot_root", MOUNT,
            "--arm_joint_regex", rf"{PREFIX}(?!{re.escape(gp)}).*", "--finger_joint_regex", rf"{PREFIX}{re.escape(gp)}.*",
            "--ee_site", TCP_SITE]


def add_args(ap):
    ap.add_argument("--robot", default="fr3",
                    help="fr3, or a robots.yaml entry (e.g. ur5e_menagerie; 'ur5e' = ur5e_menagerie): zero-shot "
                         "cross-arm run, the arm in the same scene (cameras, table, cube, tape)")
    ap.add_argument("--posture_ckpt", default="",
                    help="Stage 2 posture-aware contrastive alignment of that arm (train_stage2_posture.py); "
                         "default checkpoints/stage2_posture_<entry>_clean.pt")
    ap.add_argument("--refine", type=int, default=5, help="TCP-IK steps after the learned FR3 pose")
    ap.add_argument("--arm_kp", type=float, default=900.0, help="joint tracker gains of the new arm")
    ap.add_argument("--arm_kd", type=float, default=60.0)
    ap.add_argument("--branch", choices=["config", "free"], default="config",
                    help="config = lock the signs of robots.yaml branch_joints (if given); free = no lock")


class PostureDict(nn.Module):
    """same network as train_stage2_posture.py: new-arm features -> FR3 joint angles in the FR3 limits"""
    def __init__(self, d_in):
        super().__init__()
        self.register_buffer("lo", torch.zeros(7)); self.register_buffer("hi", torch.ones(7))
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 7))

    def forward(self, x):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.net(x))


def _name(m, t, i):
    return mujoco.mj_id2name(m, t, i) or ""


def _subtree(m, root):
    out = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            p = m.body_parentid[p]
            if p == root:
                out.add(b); break
    return out


class NewArm:
    def __init__(self, sc, fk, emb, args, dev, control_hz, sim_hz, half):
        import yaml
        import train_stage2_align as TA
        from fk_encoder.fk_encoder import FKEncoder
        from decision_core.model import tcp_ik
        from decision_core.swivel import swivel
        from joint_tracker import JointTracker, UR5E_TAU
        self.name = robot_entry(args.robot)
        self.cfg = cfg = yaml.safe_load(open(TA.ROBOTS_YAML))[self.name]
        self.sc, self.m, self.d = sc, sc.m, sc.d
        self.fk, self.emb, self.a, self.dev = fk, emb, args, dev
        self.tcp_ik, self.swivel = tcp_ik, swivel
        self.hz, self.sim_hz, self.half = control_hz, sim_hz, half
        m, d = self.m, self.d
        dof = int(cfg["arm_dof"])
        gp = PREFIX + (cfg.get("gripper") or {}).get("prefix", "grip_")
        gcfg = cfg.get("gripper") or {}

        # ---- joints / actuators / bodies, all from the arm subtree and robots.yaml
        sub = _subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, MOUNT))
        arm_j = [j for j in range(m.njnt) if m.jnt_bodyid[j] in sub and not _name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(gp)
                 and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
        if len(arm_j) < dof:
            raise SystemExit(f"[{self.name}] found {len(arm_j)} arm joints, robots.yaml arm_dof = {dof}")
        arm_j = arm_j[:dof]
        self.joint_names = [_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_j]
        self.qadr = np.array([m.jnt_qposadr[j] for j in arm_j]); self.vadr = np.array([m.jnt_dofadr[j] for j in arm_j])
        self.arm_acts = [i for i in range(m.nu) if int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                         and m.actuator_trnid[i, 0] in arm_j]
        g_acts = [i for i in range(m.nu) if _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).startswith(gp)]
        if not g_acts:
            raise SystemExit(f"[{self.name}] no gripper actuator with prefix {gp}")
        self.g_act = g_acts[0]
        lo_c, hi_c = m.actuator_ctrlrange[self.g_act]
        self.G_OPEN, self.G_CLOSED = (lo_c, hi_c) if gcfg.get("open_ctrl", "low") == "low" else (hi_c, lo_c)
        fb = gcfg.get("finger_bodies")
        if fb:
            self.pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, gp + n) for n in fb]
        else:
            self.pads = [i for i in range(m.nbody) if re.search(r"(left|right)_pad$", _name(m, mujoco.mjtObj.mjOBJ_BODY, i))
                         and _name(m, mujoco.mjtObj.mjOBJ_BODY, i).startswith(gp)]
        if len(self.pads) != 2 or min(self.pads) < 0:
            raise SystemExit(f"[{self.name}] two finger bodies needed (robots.yaml gripper.finger_bodies)")
        kl = cfg["key_links"]
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + n)
        self.b_sh, self.b_el, self.b_wr = b(kl["shoulder"]), b(kl["elbow"]), b(kl["wrist"])
        if min(self.b_sh, self.b_el, self.b_wr) < 0:
            raise SystemExit(f"[{self.name}] key_links bodies not found: {kl}")
        self.arm_links = [i for i in _subtree(m, self.b_sh) if not _name(m, mujoco.mjtObj.mjOBJ_BODY, i).startswith(gp)]
        self.branch_j = list(cfg.get("branch_joints") or []) if args.branch == "config" else []
        print(f"[{self.name}] joints {self.joint_names} | gripper actuator {_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, self.g_act)}"
              f" open={self.G_OPEN:g} closed={self.G_CLOSED:g} | branch lock on joints {self.branch_j or 'none'}")

        tau = np.asarray(cfg.get("tau_max", UR5E_TAU), np.float64)       # torque limits of the joint tracker
        self.trk = JointTracker(m, self.qadr, self.vadr, tau, actuator_ids=self.arm_acts, kp=args.arm_kp, kd=args.arm_kd)
        self.enc_u = FKEncoder(self.name, TA.ROBOTS_YAML, project_root=R, device=dev)
        TA.setup_target(self.name, self.enc_u)
        self.vstep = np.asarray(TA.TGT_VEL_LIMIT, np.float64).reshape(-1) / control_hz
        self.lim_f = torch.tensor(TA.FR3_JOINT_LIMITS, dtype=torch.float32, device=dev)
        lim_u = np.asarray(TA.TGT_JOINT_LIMITS, np.float64).reshape(dof, 2)
        seed = cfg.get("ik_seed")
        self.seed0 = torch.tensor([seed if seed is not None else np.clip(lim_u.mean(1), -np.pi, np.pi)],
                                  dtype=torch.float32, device=dev)

        p = args.posture_ckpt or f"checkpoints/stage2_posture_{self.name}_clean.pt"
        p = p if os.path.isabs(p) else os.path.join(R, p)
        ck = torch.load(p, map_location=dev, weights_only=False)
        self.pdict = PostureDict(ck["config"]["d_in"]).to(dev)
        self.pdict.load_state_dict(ck["state_dict"]); self.pdict.eval()
        print(f"[{self.name}] posture dictionary {p} (+{args.refine} IK steps), joint tracker kp {args.arm_kp} kd {args.arm_kd}")

        # TCP of the physics model == TCP of the FK encoder?
        q = self.seed0[0].cpu().numpy().astype(np.float64); d.qpos[self.qadr] = q; mujoco.mj_kinematics(m, d)
        p_fk = self.enc_u.forward(self.seed0)[0, :3].cpu().numpy()
        err = (p_fk - d.site_xpos[sc.ee]) * 1000
        print(f"[{self.name}] TCP check (FK encoder - physics): {np.round(err, 2)} mm"
              + ("   <-- WARNING: > 2 mm, the scene's arm is not the encoder's" if np.linalg.norm(err) > 2 else ""))

        # gripper: finger distance closed (nothing in it) and open -> Franka-equivalent finger value
        o_adr = sc.cube.qadr
        mujoco.mj_resetData(m, d); d.qpos[:] = sc.qpos_init; d.qpos[self.qadr] = q
        d.qpos[o_adr:o_adr + 3] = [2, 2, 0.025]
        self._settle(self.G_CLOSED); self.d_closed = self.pad_dist()
        self._settle(self.G_OPEN); self.d_open = self.pad_dist()
        print(f"[{self.name}] finger distance closed {self.d_closed * 1000:.1f} mm, open {self.d_open * 1000:.1f} mm")
        mujoco.mj_resetData(m, d); d.qpos[:] = sc.qpos_init

        # FR3 start poses (same sampler as the FR3 evaluation), mapped to the new arm by IK on the TCP
        from build_scene import build_model as build_fr3
        from generate_fr3_expert_data import resolve_joint_and_actuator_ids
        self.mf = build_fr3(); self.dfr = mujoco.MjData(self.mf)
        self.fj = resolve_joint_and_actuator_ids(self.mf)[0]
        oj = mujoco.mj_name2id(self.mf, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        self.o_adr_f = self.mf.jnt_qposadr[oj] if oj >= 0 else None
        self.branch = None
        self.qf = None
        self.n_flip = 0

    # ------------------------------------------------------------------ gripper
    def _settle(self, ctrl):
        m, d = self.m, self.d
        d.ctrl[self.g_act] = ctrl
        for _ in range(int(1.0 * self.sim_hz)):
            d.qfrc_applied[:] = 0
            d.qfrc_applied[self.vadr] = d.qfrc_bias[self.vadr]
            mujoco.mj_step(m, d)

    def pad_dist(self):
        return float(np.linalg.norm(self.d.xpos[self.pads[0]] - self.d.xpos[self.pads[1]]))

    def finger(self):
        return float(np.clip((self.pad_dist() - self.d_closed) / 2, 0.0, 0.04))

    def set_grip(self, closed):
        self.d.ctrl[self.g_act] = self.G_CLOSED if closed else self.G_OPEN

    # ------------------------------------------------------------------ encoder / decoder
    def z(self, qt):
        """arm joints (1, dof) -> shared latent (D,), via the FR3 pose of the posture dictionary"""
        fu = self.enc_u.forward(qt); psi = self.swivel(self.enc_u, qt)
        x = torch.cat([fu[:, :18], torch.sin(psi)[:, None], torch.cos(psi)[:, None]], 1)
        lo, hi = self.lim_f[:, 0], self.lim_f[:, 1]
        qf = torch.max(torch.min(self.pdict(x), hi), lo)
        if self.a.refine > 0:
            qf = torch.max(torch.min(self.tcp_ik(self.fk, qf, fu[:, :3], fu[:, 3:9], iters=self.a.refine), hi), lo)
        self.qf = qf
        return self.emb(self.fk.forward(qf)[:, :18])[0]

    def tcp(self, qt):
        return self.enc_u.forward(qt)

    def read(self, out, qt):
        """the virtual FR3 reads the core's latent -> TCP intent (1, >=9)"""
        from decision_core.model import inverse_decode
        lo, hi = self.lim_f[:, 0], self.lim_f[:, 1]
        fu = self.enc_u.forward(qt)
        q0 = self.qf if self.qf is not None else self.lim_f.mean(1)[None]
        q0 = torch.max(torch.min(self.tcp_ik(self.fk, q0, fu[:, :3], fu[:, 3:9]), hi), lo)
        q_read = inverse_decode(out[None], q0, self.fk, self.emb, lo=lo, hi=hi)
        return self.fk.forward(q_read)

    def ik(self, qt, pos_t, rot_t):
        q_tgt = self.tcp_ik(self.enc_u, qt, pos_t, rot_t)
        if self.branch_j and self.branch is not None:
            same = lambda q: all(torch.sign(q[0, j]) == self.branch[k] for k, j in enumerate(self.branch_j))
            if not same(q_tgt):                       # no branch flip: move only part of the way
                self.n_flip += 1
                for frac in (0.5, 0.25, 0.1, 0.0):
                    q_try = qt + frac * (q_tgt - qt)
                    if same(q_try):
                        q_tgt = q_try; break
        return q_tgt

    # ------------------------------------------------------------------ start pose
    def start(self, cube_np, sample_random_init, home_q, frac, tries):
        """FR3 start pose (same sampler as the FR3 run) -> new-arm pose with the same TCP, elbow above the
        shoulder-wrist line, arm links > 5 cm above the table. None if there is none (counted as a failure)."""
        m, d, dev = self.m, self.d, self.dev
        if self.o_adr_f is not None:
            mujoco.mj_resetData(self.mf, self.dfr)
            self.dfr.qpos[self.o_adr_f:self.o_adr_f + 3] = cube_np
            mujoco.mj_forward(self.mf, self.dfr)
        q0f = torch.tensor(np.asarray(sample_random_init(self.mf, self.dfr, home_q, self.fj, frac, tries), np.float32)
                           [None], device=dev)
        f0 = self.fk.forward(q0f)
        best = None
        qsave = d.qpos.copy()
        for s_ in range(24):
            qs = self.seed0 + (0 if s_ == 0 else 0.5 * torch.randn_like(self.seed0))
            qu = self.tcp_ik(self.enc_u, qs, f0[:, :3], f0[:, 3:9], iters=60)
            qu = torch.atan2(torch.sin(qu), torch.cos(qu))
            if (self.enc_u.forward(qu)[0, :3] - f0[0, :3]).norm().item() < 0.003:
                d.qpos[self.qadr] = qu[0].cpu().numpy(); mujoco.mj_kinematics(m, d)
                up = d.xpos[self.b_el][2] - 0.5 * (d.xpos[self.b_sh][2] + d.xpos[self.b_wr][2])
                clear = min(d.xpos[i][2] for i in self.arm_links)
                if up > 0.02 and clear > 0.05 and (best is None or up > best[1]):
                    best = (qu[0].cpu().numpy(), up)
        d.qpos[:] = qsave
        if best is None:
            return None
        self.branch = [torch.sign(torch.tensor(best[0][j])).to(dev) for j in self.branch_j]
        self.qf = None
        self.n_flip = 0
        return best[0]