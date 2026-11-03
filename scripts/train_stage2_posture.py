"""
Stage 2, posture-aware contrastive alignment of a NEW arm into the FR3 latent space.

    q_N --phi_N--> g --> q_F (FR3 joint angles, inside the FR3 limits) --phi_F--> W_FR3 --> z

The dictionary g outputs FR3 JOINT ANGLES, so whatever it outputs is a pose the FR3 can take (the
learned latent dictionary of v15/m1/m2 could output latents no FR3 pose has, and the decision core
then failed). What counts as "the same" is defined physically -- TCP position, TCP orientation
and the swivel (arm) angle psi -- and the alignment is contrastive:

  physical distance   d2(a, b) = |dp|^2 / s_p^2 + |dr6|^2 / s_r^2 + wrap(dpsi)^2 / s_psi^2
  target              P_ik ∝ exp(-d2(new_i, FR3_k) / 2)          over a set of FR3 candidates k
  model               Q_ik ∝ exp(-|z_g(i) - z_k|^2 / (2 s_z^2))  in the latent
  L_align = mean_i KL(P_i || Q_i)                                 (soft-label InfoNCE)
  L_phys  = mean_i d2(new_i, g(new_i))                            (match the physical quantities)
  L_smooth= mean_i |g(q_N + dq) - g(q_N)|^2 / |dq|^2             (no jumps, e.g. near singularities)
  L = L_align + lam_phys * L_phys + lam_smooth * L_smooth

The candidates are, for every new-arm sample, the exact projection (same TCP and same swivel,
found by IK on the FR3 null-space) when the FR3 can take it, otherwise the closest FR3 pose found,
plus a pool of random FR3 poses. Where the FR3 can take the posture, P peaks on the exact
projection; where it cannot (outside its workspace or swivel range), P peaks on the most similar
FR3 poses -- the alignment then degrades smoothly instead of being undefined.

Constants (origin):
  s_p   = 10 mm      the end-effector error target of the proposal
  s_r   = 0.12       |dr6| of a 5 deg rotation (each of the two columns moves ~0.087, sqrt(2) of that)
  s_psi = 5 deg      a third of the FR3's swivel range at grasp height (about +-15 deg)
  s_z   = s_p / 347 mm = 0.029   the same bandwidth in latent units (1 latent unit ~ 347 mm of TCP)

  python scripts/train_stage2_posture.py --robot ur5e_menagerie --tag _posture
"""
import argparse, os, sys, math, time
import numpy as np
import torch
import torch.nn as nn

R = os.path.expanduser("~/cross_embody")
for p in (R, os.path.join(R, "scripts")):
    sys.path.insert(0, p)
import train_stage2_align as T
from fk_encoder.fk_encoder import FKEncoder
from decision_core.model import load_frozen_stage1, tcp_ik
from decision_core.swivel import swivel, swivel_from_points, project_swivel, wrap

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="ur5e_menagerie")
ap.add_argument("--tag", default="_posture")
ap.add_argument("--n_uniform", type=int, default=60000, help="new-arm poses sampled uniformly above the table")
ap.add_argument("--n_tabletop", type=int, default=40000, help="new-arm poses with the TCP near the table, pointing down")
ap.add_argument("--n_pool", type=int, default=60000, help="random FR3 poses used as extra candidates")
ap.add_argument("--n_cand", type=int, default=1536, help="random pool candidates per batch")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--s_p", type=float, default=0.010)
ap.add_argument("--s_r", type=float, default=0.12)
ap.add_argument("--s_psi_deg", type=float, default=5.0)
ap.add_argument("--posture_term", choices=["swivel", "keypoints", "both"], default="keypoints",
                help="what 'same posture' means: swivel = same arm angle (natural for 7-DOF arms, only 0/180 deg "
                     "for a UR5e); keypoints = elbow and wrist at the same place relative to the shoulder (works "
                     "across arms with different numbers of joints)")
ap.add_argument("--s_k", type=float, default=0.10,
                help="key-point scale, in upper-arm lengths (0.10 ~ 30-40 mm of elbow / wrist position)")
ap.add_argument("--lam_phys", type=float, default=1.0)
ap.add_argument("--lam_smooth", type=float, default=0.1)
ap.add_argument("--elbow_up", action="store_true",
                help="train only on new-arm poses with |swivel| < 90 deg (the branch the arm is deployed in)")
ap.add_argument("--refine", type=int, default=5,
                help="evaluation also reports g + this many TCP-IK steps (the learned pose as the IK seed)")
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MM_PER_LATENT = 347.0
S_PSI = math.radians(args.s_psi_deg)
S_Z = args.s_p * 1000.0 / MM_PER_LATENT
CACHE = os.path.join(R, "cache", f"posture_pairs_{args.robot}_u{args.n_uniform}_t{args.n_tabletop}_p{args.n_pool}.pt")
CKPT = os.path.join(R, "checkpoints", f"stage2_posture_{args.robot}{args.tag}.pt")


def phys(enc, q, grad=False):
    """TCP position, rot6d and swivel of joint configurations (differentiable if grad)"""
    f = enc.forward(q)
    pos, _ = enc._key_frames(q)
    if enc.cfg.get("swivel_links"):
        psi = swivel(enc, q)                    # (new arm, no gradient needed)
    else:
        psi = swivel_from_points(pos["shoulder"], pos["elbow"], pos["wrist"])
    return f[:, :3], f[:, 3:9], psi, f


def keyvec(enc, q):
    """elbow and wrist relative to the shoulder, in upper-arm lengths, from points IN the arm's plane
    (robots.yaml swivel_links if given -- e.g. the UR5e's laterally offset upper-arm origin is not
    used); differentiable. (B, 6)"""
    names = enc.cfg.get("swivel_links")
    if names:
        ret = enc.chain.forward_kinematics(enc._full_q(q))
        pos = {r: ret[l].get_matrix()[:, :3, 3] @ enc.base_R.T for r, l in names.items()}
    else:
        pos, _ = enc._key_frames(q)
    S, E, W = pos["shoulder"], pos["elbow"], pos["wrist"]
    L = (E - S).norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([(E - S) / L, (W - S) / L], -1)


def d2(pa, ra, sa, pb, rb, sb, ka=None, kb=None):
    out = ((pa - pb) ** 2).sum(-1) / args.s_p ** 2 + ((ra - rb) ** 2).sum(-1) / args.s_r ** 2
    if args.posture_term in ("swivel", "both"):
        out = out + wrap(sa - sb) ** 2 / S_PSI ** 2
    if args.posture_term in ("keypoints", "both") and ka is not None:
        out = out + ((ka - kb) ** 2).sum(-1) / args.s_k ** 2
    return out


# ----------------------------------------------------------------------------- data
@torch.no_grad()
def build(fk, emb, enc_n, lim_f):
    if os.path.exists(CACHE):
        print(f"[data] cache {CACHE}")
        return torch.load(CACHE, map_location=DEV)
    lo, hi = lim_f[:, 0], lim_f[:, 1]
    # new-arm poses: uniform above the table + near-table, gripper-down poses (IK to FR3 tabletop TCPs)
    qn = T.sample_q(T.TGT_JOINT_LIMITS, args.n_uniform * 3)
    pz = enc_n.forward(qn)[:, :3]
    keep = (pz[:, 2] > 0.02) & (pz.norm(dim=1) < 0.95)
    qn = qn[keep][:args.n_uniform]
    with torch.enable_grad():                  # this sampler runs a gradient-based IK
        q_tab_f = T.sample_tabletop_fr3(fk, args.n_tabletop).detach()
    ft = fk.forward(q_tab_f)
    seeds = torch.tensor([[np.pi, -1.57, 1.57, -1.57, -1.57, 0.0], [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
                          [np.pi / 2, -1.57, 1.57, -1.57, -1.57, 0.0], [-np.pi / 2, -1.57, 1.57, -1.57, -1.57, 0.0]],
                         device=DEV)
    best_q, best_e = None, None
    for s in seeds:
        qq = tcp_ik(enc_n, s[None].repeat(len(ft), 1), ft[:, :3], ft[:, 3:9], iters=60)
        e = (enc_n.forward(qq)[:, :3] - ft[:, :3]).norm(dim=1)
        if best_q is None:
            best_q, best_e = qq, e
        else:
            m = e < best_e
            best_q[m], best_e[m] = qq[m], e[m]
    q_tab_n = T.project_tgt(best_q[best_e < 1e-3])
    qn = torch.cat([qn, q_tab_n], 0)
    pn, rn, sn, fn = phys(enc_n, qn)
    print(f"[data] new-arm poses: {len(qn)} ({len(q_tab_n)} near the table)")

    # exact projection onto the FR3: same TCP and same swivel, several seeds, keep the closest
    home = torch.tensor([[0, -0.3, 0, -2.2, 0, 2.0, 0.785]], device=DEV)
    cand_seeds = [home, home + torch.tensor([[0.6, 0, -0.6, 0, 0.6, 0, 0]], device=DEV),
                  home + torch.tensor([[-0.6, 0, 0.6, 0, -0.6, 0, 0]], device=DEV)]
    qf, dbest = torch.zeros(len(qn), 7, device=DEV), torch.full((len(qn),), float("inf"), device=DEV)
    t0 = time.time()
    for c0 in range(0, len(qn), 4096):
        sl = slice(c0, c0 + 4096)
        for s in cand_seeds:
            q0 = s.repeat(len(pn[sl]), 1).clamp(lo, hi)
            q0 = tcp_ik(fk, q0, pn[sl], rn[sl], iters=40).clamp(lo, hi)
            q1 = project_swivel(fk, q0, pn[sl], rn[sl], sn[sl], lo, hi, tcp_ik, outer=6)
            pf, rf, sf, _ = phys(fk, q1)
            dd = d2(pn[sl], rn[sl], sn[sl], pf, rf, sf)
            m = dd < dbest[sl]
            qf[sl][m], dbest[sl][m] = q1[m], dd[m]
        print(f"[data] projection {min(c0 + 4096, len(qn))}/{len(qn)}  ({time.time() - t0:.0f}s)", flush=True)
    pf, rf, sf, ff = phys(fk, qf)
    exact = ((pf - pn).norm(dim=1) < 1e-3) & ((rf - rn).norm(dim=1) < 0.02) & (wrap(sf - sn).abs() < math.radians(2))
    print(f"[data] the FR3 can take the same TCP and swivel for {exact.float().mean():.1%} of the new-arm poses")

    # random FR3 pool (uniform above the table + tabletop)
    qp = T.sample_q(T.FR3_JOINT_LIMITS, args.n_pool * 2)
    qp = qp[fk.forward(qp)[:, 2] > 0.0][:args.n_pool // 2]
    with torch.enable_grad():
        q_tab_pool = T.sample_tabletop_fr3(fk, args.n_pool // 2).detach()
    qp = torch.cat([qp, q_tab_pool], 0)
    pp, rp, sp, fp = phys(fk, qp)
    data = dict(qn=qn, xn=torch.cat([fn[:, :18], torch.sin(sn)[:, None], torch.cos(sn)[:, None]], 1),
                pn=pn, rn=rn, sn=sn, qf=qf, pf=pf, rf=rf, sf=sf, zf=emb(ff[:, :18]), exact=exact,
                qp=qp, pp=pp, rp=rp, sp=sp, zp=emb(fp[:, :18]))
    torch.save(data, CACHE)
    return data


# ----------------------------------------------------------------------------- model
class PostureDict(nn.Module):
    """new-arm features -> FR3 joint angles inside the FR3 limits"""
    def __init__(self, d_in, lo, hi, home):
        super().__init__()
        self.register_buffer("lo", lo); self.register_buffer("hi", hi)
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 7))
        u = ((home - lo) / (hi - lo)).clamp(0.02, 0.98)
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.log(u / (1 - u)))

    def forward(self, x):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.net(x))


def main():
    fk, emb, _, _ = load_frozen_stage1(device=DEV)
    for p_ in emb.parameters():
        p_.requires_grad_(False)
    enc_n = FKEncoder(args.robot, T.ROBOTS_YAML, project_root=R, device=DEV)
    T.setup_target(args.robot, enc_n)
    lim_f = torch.tensor(T.FR3_JOINT_LIMITS, dtype=torch.float32, device=DEV)
    D = build(fk, emb, enc_n, lim_f)
    with torch.no_grad():
        D["kn"] = torch.cat([keyvec(enc_n, D["qn"][i:i + 8192]) for i in range(0, len(D["qn"]), 8192)])
        D["kf"] = torch.cat([keyvec(fk, D["qf"][i:i + 8192]) for i in range(0, len(D["qf"]), 8192)])
        D["kp"] = torch.cat([keyvec(fk, D["qp"][i:i + 8192]) for i in range(0, len(D["qp"]), 8192)])
    N = len(D["qn"]); perm = torch.randperm(N, device=DEV)
    if args.elbow_up:
        keep = D["sn"][perm].abs() < math.pi / 2
        print(f"[data] elbow-up branch only: {int(keep.sum())}/{N} new-arm poses "
              f"(FR3 can take {D['exact'][perm][keep].float().mean():.1%} of them exactly)")
        perm = perm[keep]; N = len(perm)
    va, tr = perm[:N // 10], perm[N // 10:]
    va = va[:2048]                                  # validation batch (the candidate matrix is |va| x (|va| + n_cand))
    home = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, 0.785], device=DEV)
    g = PostureDict(D["xn"].shape[1], lim_f[:, 0], lim_f[:, 1], home).to(DEV)
    opt = torch.optim.AdamW(g.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    print(f"[train] {len(tr)} train / {len(va)} val  s_p={args.s_p * 1000:.0f} mm  s_r={args.s_r}  "
          f"s_psi={args.s_psi_deg} deg  s_z={S_Z:.3f}  -> {CKPT}")

    def losses(idx):
        x = D["xn"][idx]
        q = g(x)
        p, r, s, f = phys(fk, q, grad=True)
        z = emb(f[:, :18])
        k = keyvec(fk, q)
        l_phys = d2(D["pn"][idx], D["rn"][idx], D["sn"][idx], p, r, s, D["kn"][idx], k).mean()
        # candidates: the batch's projections + random pool poses
        pk = torch.randint(0, len(D["qp"]), (args.n_cand,), device=DEV)
        cp = torch.cat([D["pf"][idx], D["pp"][pk]]); cr = torch.cat([D["rf"][idx], D["rp"][pk]])
        cs = torch.cat([D["sf"][idx], D["sp"][pk]]); cz = torch.cat([D["zf"][idx], D["zp"][pk]])
        ck = torch.cat([D["kf"][idx], D["kp"][pk]])
        tgt = torch.softmax(-0.5 * d2(D["pn"][idx][:, None], D["rn"][idx][:, None], D["sn"][idx][:, None],
                                      cp[None], cr[None], cs[None], D["kn"][idx][:, None], ck[None]), dim=1)
        logq = torch.log_softmax(-((z[:, None] - cz[None]) ** 2).sum(-1) / (2 * S_Z ** 2), dim=1)
        l_align = (tgt * (torch.log(tgt + 1e-12) - logq)).sum(1).mean()
        # smoothness: a small joint motion of the new arm moves the FR3 pose proportionally
        dq = 0.02 * torch.randn_like(D["qn"][idx])
        pn2, rn2, sn2, fn2 = phys(enc_n, D["qn"][idx] + dq)
        q2 = g(torch.cat([fn2[:, :18], torch.sin(sn2)[:, None], torch.cos(sn2)[:, None]], 1))
        l_smooth = (((q2 - q) ** 2).sum(1) / (dq ** 2).sum(1)).mean()
        return l_align, l_phys, l_smooth, q, p, r, s, k

    best = float("inf")
    for ep in range(args.epochs):
        g.train(); t0 = time.time(); tot = 0
        for b in range(0, len(tr), args.batch):
            idx = tr[b:b + args.batch]
            la, lp, ls, *_ = losses(idx)
            loss = la + args.lam_phys * lp + args.lam_smooth * ls
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item()
        tr = tr[torch.randperm(len(tr), device=DEV)]
        g.eval()
        with torch.no_grad():
            la, lp, ls, q, p, r, s, k = losses(va)
            ek = ((k - D["kn"][va]).reshape(-1, 2, 3).norm(dim=-1).max(-1).values)       # worse of elbow / wrist
            ex = D["exact"][va]
            ep_mm = (p - D["pn"][va]).norm(dim=1) * 1000
            epsi = torch.rad2deg(wrap(s - D["sn"][va]).abs())
            proj_mm = (D["pf"][va] - D["pn"][va]).norm(dim=1) * 1000
            proj_psi = torch.rad2deg(wrap(D["sf"][va] - D["sn"][va]).abs())
            proj_k = ((D["kf"][va] - D["kn"][va]).reshape(-1, 2, 3).norm(dim=-1).max(-1).values)
            # learned pose as the seed of a few TCP-IK steps: exact TCP where the FR3 can reach it,
            # the posture (branch / elbow) still chosen by the learned alignment
            qr = tcp_ik(fk, q, D["pn"][va], D["rn"][va], iters=args.refine).clamp(lim_f[:, 0], lim_f[:, 1])
            prr, rrr, srr, _ = phys(fk, qr)
            ekr = ((keyvec(fk, qr) - D["kn"][va]).reshape(-1, 2, 3).norm(dim=-1).max(-1).values)
            er_mm = (prr - D["pn"][va]).norm(dim=1) * 1000
            er_psi = torch.rad2deg(wrap(srr - D["sn"][va]).abs())
        val = la.item() + args.lam_phys * lp.item()
        msg = (f"[ep {ep:3d}] ({time.time() - t0:4.1f}s) align={la.item():.3f} phys={lp.item():.2f} smooth={ls.item():.2f}"
               f" | FR3-can: TCP {ep_mm[ex].median():.1f} mm, swivel {epsi[ex].median():.1f} deg"
               f" | FR3-cannot: TCP {ep_mm[~ex].median():.1f} mm (projection {proj_mm[~ex].median():.1f}),"
               f" swivel {epsi[~ex].median():.1f} deg (projection {proj_psi[~ex].median():.1f})")
        msg += (f"\n          + {args.refine} IK steps | FR3-can: TCP {er_mm[ex].median():.1f} mm, swivel {er_psi[ex].median():.1f} deg"
                f" | FR3-cannot: TCP {er_mm[~ex].median():.1f} mm, swivel {er_psi[~ex].median():.1f} deg"
                f"\n          key points (elbow/wrist, upper-arm lengths): learned+IK {ekr.median():.3f}"
                f"  vs projection {proj_k.median():.3f}")
        if val < best:
            best = val
            torch.save({"state_dict": g.state_dict(), "config": {"robot": args.robot, "d_in": D["xn"].shape[1],
                        "s_p": args.s_p, "s_r": args.s_r, "s_psi_deg": args.s_psi_deg,
                        "posture_term": args.posture_term, "s_k": args.s_k}}, CKPT)
            msg += "  -> best"
        print(msg, flush=True)
    print(f"[done] {CKPT}")


if __name__ == "__main__":
    main()