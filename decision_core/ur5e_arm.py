"""
The arm side of the cross-arm (zero-shot) evaluation: UR5e + Robotiq 2F-85 under the FR3-trained vision core.

Same pieces as scripts/eval_pp_ur5e.py --encoder posture_dict (the protocol's UR5e set-up), only the
scene comes from gen_vision_data.Scene(build_scene_ur5e) so the cameras / table / cube / tape are the FR3 ones:

  UR5e joints --posture dictionary (Stage 2, contrastive)--> FR3 pose --FR3 FK + W_FR3--> latent      (arm history)
  2F-85 pad distance -> Franka-equivalent finger value (half the gap, 0..0.04 m)                    (finger history)
  core latent --virtual FR3 reads it (inverse decode)--> TCP intent --(anchor blend)--> UR5e IK      (action)
  gripper logit -> 2F-85 open / closed

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

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
UR_SCENE_FLAGS = ["--scene_module", "build_scene_ur5e", "--robot_root", "ur_mount",
                  "--arm_joint_regex", r"ur_(shoulder_pan|shoulder_lift|elbow|wrist_[123])_joint",
                  "--finger_joint_regex", r"ur_grip_.*", "--ee_site", "ur_tcp_site"]


def add_args(ap):
    ap.add_argument("--robot", choices=["fr3", "ur5e"], default="fr3",
                    help="ur5e = zero-shot cross-arm: UR5e + 2F-85 in the same scene (cameras, table, cube, tape)")
    ap.add_argument("--ur_robot", default="ur5e_menagerie", help="robots.yaml entry of the UR5e")
    ap.add_argument("--posture_ckpt", default="checkpoints/stage2_posture_ur5e_menagerie_posture_up.pt",
                    help="Stage 2 posture-aware contrastive alignment (train_stage2_posture.py)")
    ap.add_argument("--refine", type=int, default=5, help="TCP-IK steps after the learned FR3 pose")
    ap.add_argument("--ur_kp", type=float, default=900.0)
    ap.add_argument("--ur_kd", type=float, default=60.0)
    ap.add_argument("--ur_branch", choices=["lock", "free"], default="lock",
                    help="lock = elbow / wrist-2 keep the sign of the start pose (as the protocol's UR5e runs)")


class PostureDict(nn.Module):
    """same network as train_stage2_posture.py: new-arm features -> FR3 joint angles in the FR3 limits"""
    def __init__(self, d_in):
        super().__init__()
        self.register_buffer("lo", torch.zeros(7)); self.register_buffer("hi", torch.ones(7))
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 7))

    def forward(self, x):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.net(x))


class UR5eArm:
    def __init__(self, sc, fk, emb, args, dev, control_hz, sim_hz, half):
        import train_stage2_align as TA
        from fk_encoder.fk_encoder import FKEncoder
        from decision_core.model import tcp_ik
        from decision_core.swivel import swivel
        from joint_tracker import JointTracker, UR5E_TAU
        self.sc, self.m, self.d = sc, sc.m, sc.d
        self.fk, self.emb, self.a, self.dev = fk, emb, args, dev
        self.tcp_ik, self.swivel = tcp_ik, swivel
        self.hz, self.sim_hz, self.half = control_hz, sim_hz, half
        m, d = self.m, self.d
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ur_" + n) for n in ARM]
        assert min(jid) >= 0, "UR5e joints not found (scene module build_scene_ur5e?)"
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid]); self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        aname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        self.g_act = next(i for i in range(m.nu) if aname(i).startswith("ur_grip_"))
        self.arm_acts = [i for i in range(m.nu) if aname(i).startswith("ur_") and not aname(i).startswith("ur_grip_")]
        self.G_OPEN, self.G_CLOSED = m.actuator_ctrlrange[self.g_act]        # menagerie 2F-85: 0 open, 255 closed
        self.pads = [i for i in range(m.nbody) if re.search(r"ur_grip_.*(left|right)_pad$",
                                                             mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
        assert len(self.pads) == 2, "2F-85 pad bodies not found"
        self.trk = JointTracker(m, self.qadr, self.vadr, UR5E_TAU, actuator_ids=self.arm_acts, kp=args.ur_kp, kd=args.ur_kd)

        self.enc_u = FKEncoder(args.ur_robot, TA.ROBOTS_YAML, project_root=R, device=dev)
        TA.setup_target(args.ur_robot, self.enc_u)
        self.vstep = np.asarray(TA.TGT_VEL_LIMIT, np.float64).reshape(-1) / control_hz
        self.lim_f = torch.tensor(TA.FR3_JOINT_LIMITS, dtype=torch.float32, device=dev)

        p = args.posture_ckpt if os.path.isabs(args.posture_ckpt) else os.path.join(R, args.posture_ckpt)
        ck = torch.load(p, map_location=dev, weights_only=False)
        self.pdict = PostureDict(ck["config"]["d_in"]).to(dev)
        self.pdict.load_state_dict(ck["state_dict"]); self.pdict.eval()
        print(f"[ur5e] posture dictionary {args.posture_ckpt} (+{args.refine} IK steps), joint tracker kp {args.ur_kp} "
              f"kd {args.ur_kd}, branch {args.ur_branch}")

        # TCP of the physics model == TCP of the FK encoder?
        q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0]); d.qpos[self.qadr] = q; mujoco.mj_kinematics(m, d)
        p_fk = self.enc_u.forward(torch.tensor(q[None], dtype=torch.float32, device=dev))[0, :3].cpu().numpy()
        err = (p_fk - d.site_xpos[sc.ee]) * 1000
        print(f"[ur5e] TCP check (FK encoder - physics): {np.round(err, 2)} mm"
              + ("   <-- WARNING: > 2 mm, the scene's UR5e is not the encoder's" if np.linalg.norm(err) > 2 else ""))

        # gripper: pad distance closed (nothing in it) and open -> Franka-equivalent finger value
        o_adr = sc.cube.qadr
        mujoco.mj_resetData(m, d); d.qpos[:] = sc.qpos_init; d.qpos[self.qadr] = q
        d.qpos[o_adr:o_adr + 3] = [2, 2, 0.025]
        self._settle(self.G_CLOSED); self.d_closed = self.pad_dist()
        self._settle(self.G_OPEN); self.d_open = self.pad_dist()
        print(f"[ur5e] 2F-85 pad distance closed {self.d_closed * 1000:.1f} mm, open {self.d_open * 1000:.1f} mm")
        mujoco.mj_resetData(m, d); d.qpos[:] = sc.qpos_init

        # FR3 start poses (same sampler as the FR3 evaluation), mapped to the UR5e by IK on the TCP
        from build_scene import build_model as build_fr3
        from generate_fr3_expert_data import resolve_joint_and_actuator_ids
        self.mf = build_fr3(); self.dfr = mujoco.MjData(self.mf)
        self.fj = resolve_joint_and_actuator_ids(self.mf)[0]
        oj = mujoco.mj_name2id(self.mf, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        self.o_adr_f = self.mf.jnt_qposadr[oj] if oj >= 0 else None
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.b_sh, self.b_el, self.b_wr = b("ur_upper_arm_link"), b("ur_forearm_link"), b("ur_wrist_1_link")
        self.arm_links = [b("ur_" + n) for n in ("upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
                                                 "wrist_3_link")]
        assert min([self.b_sh, self.b_el, self.b_wr] + self.arm_links) >= 0, "UR5e link bodies not found"
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
        """UR5e joints (1, 6) -> shared latent (D,), via the FR3 pose of the posture dictionary"""
        fu = self.enc_u.forward(qt); psi = self.swivel(self.enc_u, qt)
        x = torch.cat([fu[:, :18], torch.sin(psi)[:, None], torch.cos(psi)[:, None]], 1)
        lo, hi = self.lim_f[:, 0], self.lim_f[:, 1]
        qf = torch.max(torch.min(self.pdict(x), hi), lo)
        if self.a.refine > 0:
            qf = torch.max(torch.min(self.tcp_ik(self.fk, qf, fu[:, :3], fu[:, 3:9], iters=self.a.refine), hi), lo)
        self.qf = qf
        return self.emb(self.fk.forward(qf)[:, :18])[0]

    def tcp(self, qt):
        """(1, >=9): TCP position + rot6d of the UR5e (same convention as the FR3 FK)"""
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
        if self.a.ur_branch == "lock" and self.branch is not None:
            flip = ((torch.sign(q_tgt[0, 2]) != self.branch[0]) | (torch.sign(q_tgt[0, 4]) != self.branch[1])).item()
            if flip:                                  # do not flip elbow / wrist: move only part of the way
                self.n_flip += 1
                for frac in (0.5, 0.25, 0.1, 0.0):
                    q_try = qt + frac * (q_tgt - qt)
                    if torch.sign(q_try[0, 2]) == self.branch[0] and torch.sign(q_try[0, 4]) == self.branch[1]:
                        q_tgt = q_try; break
        return q_tgt

    # ------------------------------------------------------------------ start pose
    def start(self, cube_np, sample_random_init, home_q, frac, tries):
        """FR3 start pose (same sampler as the FR3 run) -> UR5e pose with the same TCP, elbow-up, arm clear of
        the table. None if there is none (counted as a failure, as in the protocol)."""
        m, d, dev = self.m, self.d, self.dev
        if self.o_adr_f is not None:
            mujoco.mj_resetData(self.mf, self.dfr)
            self.dfr.qpos[self.o_adr_f:self.o_adr_f + 3] = cube_np
            mujoco.mj_forward(self.mf, self.dfr)
        q0f = torch.tensor(np.asarray(sample_random_init(self.mf, self.dfr, home_q, self.fj, frac, tries), np.float32)
                           [None], device=dev)
        f0 = self.fk.forward(q0f)
        best = None
        base = torch.tensor([[0.0, -1.57, 1.57, -1.57, -1.57, 0.0]], device=dev)       # elbow-up seed
        qsave = d.qpos.copy()
        for s_ in range(24):
            qs = base + (0 if s_ == 0 else 0.5 * torch.randn(1, 6, device=dev))
            qu = self.tcp_ik(self.enc_u, qs, f0[:, :3], f0[:, 3:9], iters=60)
            qu = torch.atan2(torch.sin(qu), torch.cos(qu))
            if (self.enc_u.forward(qu)[0, :3] - f0[0, :3]).norm().item() < 0.003:
                d.qpos[self.qadr] = qu[0].cpu().numpy(); mujoco.mj_kinematics(m, d)
                up = d.xpos[self.b_el][2] - 0.5 * (d.xpos[self.b_sh][2] + d.xpos[self.b_wr][2])
                clear = min(d.xpos[b][2] for b in self.arm_links)
                if up > 0.02 and clear > 0.05 and (best is None or up > best[1]):
                    best = (qu[0].cpu().numpy(), up)
        d.qpos[:] = qsave
        if best is None:
            return None
        self.branch = (torch.sign(torch.tensor(best[0][2])).to(dev), torch.sign(torch.tensor(best[0][4])).to(dev))
        self.qf = None
        self.n_flip = 0
        return best[0]
