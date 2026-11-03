"""Final approach with the ANCHOR head (vision_core.DecisionCoreVision(anchor=True)).

Far from the goal the arm follows the residual latent chunk as before. Near it, the TCP POSITION target is
taken from the anchor head instead:
    gripper open   (grasp)      target = observed cube  + offset_to_cube[k]
    gripper closed (placement)  target = observed tape  + offset_to_tape[k]
blended in between r_out (residual only) and r_in (anchor only). The orientation always comes from the residual
output. The anchored target is absolute -- re-computed every step from the triangulated anchor -- so the last
centimetres do not depend on a chain of small residual steps.

    blend = AnchorBlend(core, args)          # args: anchor_mode, anc_k, anc_r_in, anc_r_out
    anc = core.last_anc[0]                   # right after a core forward pass (inside predict())
    pos = blend(pos_residual, anc, tcp_now, cube_obs, tape_obs, gripper_closed)
"""
import argparse

import numpy as np
import torch

from decision_core.vision_core import REL_M


def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--anchor_mode", choices=["auto", "off"], default="auto",
                    help="auto: use the anchor head for the final approach if the checkpoint has one")
    ap.add_argument("--anc_k", type=int, default=3, help="chunk step of the anchored target (0 = next step)")
    ap.add_argument("--anc_r_in", type=float, default=0.03, help="anchor only within this xy distance [m]")
    ap.add_argument("--anc_r_out", type=float, default=0.06, help="residual only beyond this xy distance [m]")
    ap.add_argument("--lock_rot", choices=["on", "off"], default="off",
                    help="with the anchor head: while the gripper is closed, keep the TCP orientation it had when it "
                         "closed (as the experts do while carrying) instead of the residual's drifting orientation")
    ap.add_argument("--anc_vmax", type=float, default=0.0,
                    help="while the anchor is active (alpha > 0): the TCP target may be at most anc_vmax / anc_hz "
                         "from the current TCP [m/s]. The anchor target is ABSOLUTE (where the expert would be k steps "
                         "ahead), so from an off-distribution state (e.g. carried too high) it can jump several cm and "
                         "the arm then runs there at full joint speed and overshoots. 0 = off")
    ap.add_argument("--anc_hz", type=float, default=30.0, help="control rate for --anc_vmax [Hz]")


class AnchorBlend:
    def __init__(self, core, args):
        self.on = bool(getattr(core, "anchor", False)) and args.anchor_mode != "off"
        self.k = int(args.anc_k)
        self.r_in, self.r_out = float(args.anc_r_in), float(args.anc_r_out)
        self.last_alpha = 0.0
        self.lock = args.lock_rot == "on" and self.on
        self.r6_lock = None
        self.step_max = float(getattr(args, "anc_vmax", 0.0)) / float(getattr(args, "anc_hz", 30.0))
        self.n_clamp = 0

    def rot(self, r6_res, r6_now, closed):
        """orientation target (rot6d, same convention as fk): residual while open; while closed the orientation
        at the moment of closing (the cube turns with the wrist, so a drifting wrist = a drifting cube yaw)"""
        if not self.lock:
            return r6_res
        if not closed:
            self.r6_lock = None
            return r6_res
        if self.r6_lock is None:
            self.r6_lock = r6_now.detach().clone()
        return self.r6_lock

    def __call__(self, pos_res, anc, tcp_now, cube_obs, tape_obs, closed):
        """pos_res (3,) torch residual TCP target, anc (K, 6) torch, tcp_now / cube_obs / tape_obs (3,) numpy"""
        self.last_alpha = 0.0
        if not self.on or anc is None or cube_obs is None or tape_obs is None:
            return pos_res
        k = min(self.k, anc.shape[0] - 1)
        if closed:
            d = float(np.linalg.norm(np.asarray(cube_obs)[:2] - np.asarray(tape_obs)[:2]))
            p_anc = torch.as_tensor(tape_obs, dtype=pos_res.dtype, device=pos_res.device) + anc[k, 3:] * REL_M
        else:
            d = float(np.linalg.norm(np.asarray(tcp_now)[:2] - np.asarray(cube_obs)[:2]))
            p_anc = torch.as_tensor(cube_obs, dtype=pos_res.dtype, device=pos_res.device) + anc[k, :3] * REL_M
        a = float(np.clip((self.r_out - d) / max(1e-6, self.r_out - self.r_in), 0.0, 1.0))
        self.last_alpha = a
        p = (1 - a) * pos_res + a * p_anc
        if self.step_max > 0 and a > 0:                  # no jumps towards an absolute anchor target
            tcp = torch.as_tensor(tcp_now, dtype=p.dtype, device=p.device)
            dv = p - tcp
            n = float(dv.norm())
            if n > self.step_max:
                p = tcp + dv * (self.step_max / n)
                self.n_clamp += 1
        return p