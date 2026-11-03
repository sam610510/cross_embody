"""
Arm posture across embodiments: the SWIVEL (arm) angle.

For any arm with a shoulder - elbow - wrist structure, with S, E, W the key points the FK encoder
already uses (robots.yaml key_links):
    a  = (W - S) / |W - S|                      shoulder-wrist axis
    e  = (E - S) - ((E - S) . a) a              elbow offset, perpendicular to the axis
    r  = up - (up . a) a  (normalized)          reference: world 'up' in that plane
    psi = atan2( (r x e) . a , r . e )          0 = elbow straight up over the S-W line,
                                                +-pi/2 = elbow to the side, pi = elbow down
It depends only on the directions S->E and S->W (the key vectors), not on link lengths, so it
means the same thing on every arm.

project_swivel(): for a TCP pose and a target swivel, find the FR3 configuration ON the FR3
null-space (same TCP exactly) whose swivel is closest to the target -- 'the FR3 posed like the
new arm'. It is a valid FR3 pose by construction (TCP IK), the posture is matched along the
1-D null-space (7 joints - 6 TCP DOF).
"""
import torch

UP = torch.tensor([0.0, 0.0, 1.0])
REF2 = torch.tensor([1.0, 0.0, 0.0])          # fallback reference when the S-W axis is vertical


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def swivel_from_points(S, E, W):
    """S, E, W (B,3) -> psi (B,)"""
    a = W - S
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    se = E - S
    e = se - (se * a).sum(-1, keepdim=True) * a
    up = UP.to(S)
    r = up - (up * a).sum(-1, keepdim=True) * a
    bad = r.norm(dim=-1) < 1e-3                                   # axis (almost) vertical
    if bad.any():
        r2 = REF2.to(S) - (REF2.to(S) * a).sum(-1, keepdim=True) * a
        r = torch.where(bad[:, None], r2, r)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.atan2((torch.cross(r, e, dim=-1) * a).sum(-1), (r * e).sum(-1))


@torch.no_grad()
def swivel(enc, q):
    """swivel angle of joint configurations q (B,dof) of the robot behind FK encoder enc.

    Arms like the UR5e have lateral link offsets along their parallel joint axes (shoulder-lift,
    elbow, wrist-1), so the key points do not lie in one plane and a straight-up elbow would read
    as ~20 deg. robots.yaml can name that axis:
        swivel_plane: {link: upper_arm_link, axis: [0, 1, 0]}
    and S and W are then moved along it into the plane of E, which removes the offsets.
    Alternatively name points that already lie in that plane:
        swivel_links: {shoulder: shoulder_link, elbow: forearm_link, wrist: wrist_1_link}"""
    cfg = getattr(enc, "cfg", None) or {}
    names = cfg.get("swivel_links")                       # points chosen to lie in the arm's plane
    if names:
        full = enc._full_q(q.to(device=enc.device, dtype=enc.dtype))
        ret = enc.chain.forward_kinematics(full)
        pos = {r: ret[l].get_matrix()[:, :3, 3] @ enc.base_R.T for r, l in names.items()}
    else:
        pos, _ = enc._key_frames(q)
    S, E, W = pos["shoulder"], pos["elbow"], pos["wrist"]
    sp = cfg.get("swivel_plane")
    if sp:
        full = enc._full_q(q.to(device=enc.device, dtype=enc.dtype))
        M = enc.chain.forward_kinematics(full)[sp["link"]].get_matrix()            # (B,4,4)
        n = (enc.base_R.unsqueeze(0) @ M[:, :3, :3]) @ torch.tensor(sp["axis"], dtype=M.dtype, device=M.device)
        n = n / n.norm(dim=-1, keepdim=True)
        S = S - ((S - E) * n).sum(-1, keepdim=True) * n
        W = W - ((W - E) * n).sum(-1, keepdim=True) * n
    return swivel_from_points(S, E, W)


@torch.no_grad()
def _tcp_jac(enc, q, eps=1e-4):
    """numerical Jacobian of the TCP features [p, r6] (B,9,dof)"""
    f0 = enc.forward(q)[:, :9]
    B, n = q.shape
    dq = torch.eye(n, device=q.device, dtype=q.dtype) * eps
    qq = (q[:, None, :] + dq[None]).reshape(B * n, n)
    f = enc.forward(qq)[:, :9].reshape(B, n, 9)
    return ((f - f0[:, None]) / eps).transpose(1, 2)


@torch.no_grad()
def project_swivel(enc, q, p_t, r6_t, psi_t, lo, hi, tcp_ik, outer=6, max_step=0.25):
    """FR3 pose with TCP (p_t, r6_t) and swivel as close as possible to psi_t, warm-started at q"""
    clamp = lambda x: torch.max(torch.min(x, hi), lo)
    q = clamp(tcp_ik(enc, q, p_t, r6_t))
    q_tcp = q.clone()
    for _ in range(outer):
        J = _tcp_jac(enc, q)                                       # (B,9,7)
        _, _, Vh = torch.linalg.svd(J)
        nvec = Vh[:, -1, :]                                        # null-space direction (B,7)
        psi = swivel(enc, q)
        eps = 1e-3
        dpsi = wrap(swivel(enc, clamp(q + eps * nvec)) - psi) / eps
        dpsi = torch.where(dpsi.abs() < 1e-4, torch.full_like(dpsi, 1e-4), dpsi)
        step = (wrap(psi_t - psi) / dpsi).clamp(-max_step, max_step)
        q = clamp(q + step[:, None] * nvec)
        q = clamp(tcp_ik(enc, q, p_t, r6_t))
    # safety: the TCP must be exact. Where the posture search drifted off the TCP (joint limits,
    # singular poses), polish with more IK; if that still misses, fall back to the plain TCP solution
    q = clamp(tcp_ik(enc, q, p_t, r6_t, iters=30))
    miss = (enc.forward(q)[:, :3] - p_t).norm(dim=1) > 1e-3
    if miss.any():
        q[miss] = q_tcp[miss]
    return q