"""A STATE-BASED (Markovian) pick-and-place expert: the next K TCP targets are a function of the current state only.

Why: with an expert that replays a pre-planned trajectory, the label of a state depends on WHEN and FROM WHERE
the plan was made (the base demos come from one planner, the DAgger corrections from re-plans that start
elsewhere), so the same state gets different labels and the regression core averages them (hovering, pumping).
Here the same state always gets the same label -- in the base demos, in DAgger relabelling and in rescues.

Rules (true sim state; the policy never sees them):
  gripper open, cube resting away from the place point      -> GRASP: to the cube, down a FUNNEL
        target = (cube xy, GRASP_Z + min(funnel * xy_error + rot_funnel * yaw_error, HOVER))
        close when xy error < tol_xy, height error < tol_z and yaw error < tol_yaw
  gripper closed on the cube, cube not lifted               -> wait until the fingers stop, then lift
  gripper closed, nothing in it                             -> open (missed grasp)
  cube held                                                  -> PLACE: the CUBE (not the TCP) to the place
        point down the same funnel, then open (hold offset = cube xy - TCP xy is compensated)
  gripper open, cube resting within succ_mm of the place     -> wait until the fingers are open, retreat up
  cube rolled / tipped, or IK fails                          -> no label (the episode ends)
Motion: straight line in Cartesian space towards the rule's target, speed min(v_max, sqrt(2 a d)) (slows down
at the target), orientation turned towards the target at <= rot_speed. The chunk is the rule applied K times to
its own predicted state (the cube moves with the TCP while held; a gripper event = wait for the rest of the
chunk). Joint labels come from IK of each TCP target, warm-started along the chunk.
"""
import mujoco
import numpy as np

# phase indices as in generate_fr3_pickplace_data.PHASES
APPROACH, DESCEND, CLOSE, LIFT, TRANSFER, LOWER, OPEN, RETREAT = range(8)


def add_args(ap):
    ap.add_argument("--se_speed", type=float, default=0.15, help="state expert: max TCP speed [m/s]")
    ap.add_argument("--se_acc", type=float, default=0.5, help="state expert: deceleration at the target [m/s^2]")
    ap.add_argument("--se_vmin", type=float, default=0.01, help="state expert: min speed while not at the target")
    ap.add_argument("--se_funnel", type=float, default=3.0,
                    help="state expert: allowed height above the grasp / set-down height per metre of xy error")
    ap.add_argument("--se_rot_funnel", type=float, default=0.002, help="... per degree of yaw error [m/deg]")
    ap.add_argument("--se_rot_speed", type=float, default=90.0, help="state expert: max turn rate [deg/s]")
    ap.add_argument("--se_tol_xy", type=float, default=0.002)
    ap.add_argument("--se_tol_z", type=float, default=0.003)
    ap.add_argument("--se_tol_yaw", type=float, default=3.0, help="[deg]")


def _quat(Rm):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(Rm, float).reshape(9)); return q


def _mat(q):
    Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q); return Rm.reshape(3, 3)


def _turn(R0, R1, max_ang):
    """rotate R0 towards R1 by at most max_ang [rad] (slerp)"""
    q0, q1 = _quat(R0), _quat(R1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    c = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    ang = 2 * np.arccos(c)
    if ang <= max_ang or ang < 1e-9:
        return _mat(q1)
    s = max_ang / ang
    th = np.arccos(c)
    q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return _mat(q / np.linalg.norm(q))


def _yaw(Rm):
    return float(np.arctan2(Rm[1, 0], Rm[0, 0]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class StateExpert:
    def __init__(self, GEN, P, half, hz, args, succ_mm=10.0):
        self.G, self.P, self.half, self.hz, self.a = GEN, P, half, hz, args
        self.succ = succ_mm / 1000.0
        self.max_rot = np.radians(args.se_rot_speed) / hz

    # ------------------------------------------------------------------ one rule step
    def _rule(self, p, Rm, closed, cube, cube_up, place, settled, opened):
        """-> (target xyz, target R, gripper closed?, wait, phase, done)"""
        G, a, half = self.G, self.a, self.half
        # held: closed on the cube and it is lifted -- OR it is over the place point (being set down: when it
        # touches the table it must not count as 'a cube on the table to pick up', which would lift it again)
        near_place = np.linalg.norm(cube[:2] - place[:2]) < self.succ
        held = closed and np.linalg.norm(cube - p) < 0.05 and (cube[2] > half + 0.004 or near_place)
        if closed and not held:
            at = np.linalg.norm(cube[:2] - p[:2]) < 0.02 and p[2] < G.GRASP_Z + 0.015
            if not at:
                return p, Rm, False, True, OPEN, False                       # missed: open where it is
            if not settled:
                return p, Rm, True, True, CLOSE, False                       # fingers still closing
            return np.r_[p[:2], G.GRASP_Z + G.LIFT], Rm, True, False, LIFT, False
        if held:
            off = cube[:2] - p[:2]
            tgt = place[:2] - off                                            # the CUBE onto the place point
            e = float(np.linalg.norm(tgt - p[:2]))
            if e < a.se_tol_xy and (p[2] - G.PLACE_Z < a.se_tol_z or cube[2] < half + 0.004):
                return p, Rm, False, True, OPEN, False
            z = G.PLACE_Z + min(a.se_funnel * e, G.HOVER)
            Rd = G.R_down(_yaw(Rm))
            return np.r_[tgt, z], Rd, True, False, (TRANSFER if e > 0.02 else LOWER), False
        # gripper open
        if cube_up < 0.95:
            return None
        resting = cube[2] < half + 0.004
        if resting and np.linalg.norm(cube[:2] - place[:2]) < self.succ:
            if not opened:
                return p, Rm, False, True, OPEN, False                       # fingers still opening
            zr = G.PLACE_Z + G.RETREAT
            return np.r_[p[:2], zr], Rm, False, False, RETREAT, p[2] > zr - 0.005
        cy = self.cube_yaw
        k = min(range(4), key=lambda k: abs(_wrap(cy + k * np.pi / 2 - _yaw(Rm))))
        Rd = G.R_down(cy + k * np.pi / 2)
        yaw_err = abs(np.degrees(_wrap(cy + k * np.pi / 2 - _yaw(Rm))))
        e = float(np.linalg.norm(cube[:2] - p[:2]))
        if e < a.se_tol_xy and abs(p[2] - G.GRASP_Z) < a.se_tol_z and yaw_err < a.se_tol_yaw:
            return p, Rm, True, True, CLOSE, False
        z = G.GRASP_Z + min(a.se_funnel * e + a.se_rot_funnel * yaw_err, G.HOVER)
        return np.r_[cube[:2], z], Rd, False, False, (APPROACH if e > 0.02 else DESCEND), False

    def _move(self, p, tgt):
        d = tgt - p
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return p.copy()
        v = float(np.clip(np.sqrt(2 * self.a.se_acc * n), self.a.se_vmin, self.a.se_speed))
        return p + d / n * min(n, v / self.hz)

    # ------------------------------------------------------------------ the label of a state
    def label(self, q, tcp, R_tcp, closed, finger, finger_prev, cube, cube_quat, place, K, finger_open=None):
        """-> dict(q (K,7), grip (K,), phase (int), done (bool)) or None (no valid label: tipped / IK)"""
        self.fail = None
        Rc = _mat(np.asarray(cube_quat, float))
        if Rc[2, 2] < 0.95:
            self.fail = "tipped"
            return None                                   # rolled / tipped: on the real robot the episode ends
        self.cube_yaw = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        moving = abs(finger - finger_prev) > 1e-4
        fo = finger if finger_open is None else finger_open      # finger position of the open gripper
        p, Rm, g = np.asarray(tcp, float).copy(), np.asarray(R_tcp, float).reshape(3, 3).copy(), bool(closed)
        cube = np.asarray(cube, float).copy()
        hold = cube - p
        settled = (not moving) and finger < fo - 0.003        # closed on the cube and no longer moving
        opened = (not moving) and finger > fo - 0.002         # fully open again
        qs, gs = [], []
        q_prev = np.asarray(q, float).copy()
        phase0, done0, waiting = None, False, False
        for k in range(K):
            if not waiting:
                r = self._rule(p, Rm, g, cube, Rc[2, 2], place, settled, opened)
                if r is None:
                    self.fail = "tipped"
                    return None
                tgt, Rt, g_new, wait, ph, done = r
                if phase0 is None:
                    phase0, done0 = ph, done
                if g_new != g:                            # a gripper event: wait for the rest of the chunk
                    g, waiting = g_new, True
                elif wait:
                    waiting = True
                else:
                    held = g and (cube[2] > self.half + 0.004 or ph in (LIFT, TRANSFER, LOWER))
                    p = self._move(p, tgt)
                    Rm = _turn(Rm, Rt, self.max_rot)
                    if held:
                        cube = p + hold
            qk, err = self.G.ik(self.P["m"], self.P["d"], self.P["site"], self.P["qadr"], self.P["vadr"], p, Rm,
                                q_prev, self.P["lo"], self.P["hi"])
            if err > 0.003:
                self.fail = "ik"
                return None
            qs.append(np.asarray(qk, float)[:7]); gs.append(float(g)); q_prev = qk
        return dict(q=np.stack(qs).astype(np.float32), grip=np.asarray(gs, np.float32), phase=phase0, done=done0)