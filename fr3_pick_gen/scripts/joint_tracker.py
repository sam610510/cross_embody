"""
One low-level joint controller for every arm (FR3, UR5e, ...), in MuJoCo.

The decision layer says WHERE the arm should be next (a joint state); this controller is the
arm's own servo that gets it there -- like the internal controllers of the real robots
(Franka torque control with gravity compensation, UR servoJ). Same law for every arm, each
with its OWN dynamics (mass matrix and gravity from its own model), torque limits and gains:

  within one control period the reference moves on a straight joint-space ramp from where the
  arm was to the commanded state, with velocity feed-forward, and computed-torque PD:
      tau = M(q) [ kp (q_ref - q) + kd (qd_ref - qd) ] + bias(q, qd)        (clipped to the limits)

Position actuators the model may have on these joints (menagerie FR3) are switched off, so they
do not add their own gravity sag on top.

Usage:
  trk = JointTracker(model, qadr, vadr, tau_limit, actuator_ids=arm_act)   # once
  trk.step(data, q_cmd, substeps, before_substep=None)                     # every control step
"""
import numpy as np
import mujoco

FR3_TAU = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)
UR5E_TAU = np.array([150, 150, 150, 28, 28, 28], dtype=np.float64)


class JointTracker:
    def __init__(self, m, qadr, vadr, tau_limit, actuator_ids=(), kp=900.0, kd=60.0):
        self.m, self.qadr, self.vadr = m, np.asarray(qadr), np.asarray(vadr)
        self.tau = np.asarray(tau_limit, dtype=np.float64)
        self.kp, self.kd = kp, kd
        self.M = np.zeros((m.nv, m.nv))
        self.max_err = 0.0
        for a in actuator_ids:                           # disable built-in position servos on these joints
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0

    def _mass(self, d):
        try:
            mujoco.mj_fullM(self.m, self.M, d.qM)
        except TypeError:
            mujoco.mj_fullM(self.m, d, self.M)
        return self.M

    def step(self, d, q_cmd, substeps, before_substep=None):
        """track q_cmd over one control period (substeps physics steps)"""
        q0 = d.qpos[self.qadr].copy()
        q_cmd = np.asarray(q_cmd, dtype=np.float64)
        qd_ref = (q_cmd - q0) / (substeps * self.m.opt.timestep)
        for k in range(substeps):
            if before_substep is not None:
                before_substep(k)
            q_ref = q0 + (q_cmd - q0) * (k + 1) / substeps
            acc = np.zeros(self.m.nv)
            acc[self.vadr] = self.kp * (q_ref - d.qpos[self.qadr]) + self.kd * (qd_ref - d.qvel[self.vadr])
            tau = (self._mass(d) @ acc + d.qfrc_bias)[self.vadr]
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[self.vadr] = np.clip(tau, -self.tau, self.tau)
            mujoco.mj_step(self.m, d)
        self.max_err = max(self.max_err, float(np.abs(d.qpos[self.qadr] - q_cmd).max()))

    def hold(self, d, steps):
        """hold the current joints (gravity compensation) for a number of physics steps"""
        q = d.qpos[self.qadr].copy()
        for _ in range(steps):
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[self.vadr] = np.clip(
                (self._mass(d) @ self._acc_hold(d, q) + d.qfrc_bias)[self.vadr], -self.tau, self.tau)
            mujoco.mj_step(self.m, d)

    def _acc_hold(self, d, q):
        acc = np.zeros(self.m.nv)
        acc[self.vadr] = self.kp * (q - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
        return acc


class CommandShaper:
    """Smooths the stream of joint targets coming from the decision layer before the servo sees it
    (part of each arm's control layer; same for every arm):
      - first-order low-pass on the target (alpha = 1: off),
      - velocity limit and ACCELERATION limit on the command, so a target that jumps between two
        control steps becomes a ramp instead of a jerk.
    The command is integrated from the previous command (not the measured joints), which keeps it
    smooth; the joint tracker keeps the arm on it."""

    def __init__(self, v_max_step, accel_steps=8, alpha=0.5):
        self.vmax = np.asarray(v_max_step, dtype=np.float64)
        self.amax = self.vmax / max(1, accel_steps)          # reach full speed in accel_steps control steps
        self.alpha = alpha
        self.tgt = self.cmd = self.v = None

    def reset(self):
        self.tgt = self.cmd = self.v = None

    def __call__(self, q_now, q_target):
        q_now = np.asarray(q_now, dtype=np.float64); q_target = np.asarray(q_target, dtype=np.float64)
        if self.cmd is None:
            self.cmd, self.v, self.tgt = q_now.copy(), np.zeros_like(q_now), q_target.copy()
        self.tgt = self.alpha * q_target + (1 - self.alpha) * self.tgt
        v_des = np.clip(self.tgt - self.cmd, -self.vmax, self.vmax)
        self.v = np.clip(v_des, self.v - self.amax, self.v + self.amax)
        self.cmd = self.cmd + self.v
        return self.cmd.copy()


def tcp_jerk(tcp_seq_m):
    """smoothness of a TCP path sampled at the control rate: |second difference| in mm/step^2
    (median and 99th percentile)"""
    p = np.asarray(tcp_seq_m) * 1000.0
    if len(p) < 3:
        return float("nan"), float("nan")
    a = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
    return float(np.median(a)), float(np.percentile(a, 99))