"""
Stage 2: cross-embodiment latent alignment (FR3 frozen -> UR5e)

L = L_align              multi-positive InfoNCE on LATENT distance (式 3-6)
                         positives defined by PHYSICAL EE distance (式 3-7, 3-8)
  + λ1 * L_decoder       self + cross decoding, EE-space supervision (式 3-9)
  + λ2 * L_dist          multi-bandwidth RBF MMD (式 3-10)

Frozen : FR3 embedding W_src (from stage1_franka.pt)
Trained: UR5e embedding W_tgt (init from W_src) + UR5e decoder

Positives (FR3 -> UR5e direction): sample FR3 poses uniformly over the whole
above-table workspace (main data) plus a small subset of expert-trajectory
states (task region), then solve batched 6-DoF IK on UR5e from IK_SEEDS random starts.
Every distinct converged UR5e solution (elbow/wrist branches) becomes an
anchor whose positive is that FR3 latent -> many-to-one "translation".
Pairs are generated once and cached; negatives / extra positives are
batch-internal.

Two-stage pipeline (default, --pipeline two_stage) -- adding a NEW ARM:
  1. add it to configs/robots.yaml (urdf, arm_dof, key_links, ee/TCP link)
  2. python scripts/train_stage2_align.py --tgt <name> --tag _<run>
  That's all: joint limits, continuous joints and velocity limits are read from
  the URDF (optional overrides in robots.yaml: sample_limits, vel_limits), and the
  script runs
     [pairs]   FR3 poses (uniform + expert) -> IK on the new arm (cached)
     [canon]   canonical FR3 elbow for every new-arm pose (cached)
     [encoder] W_new regression onto the canonical FR3 latents (early stopping)
     [decoder] Dec_new with W_new frozen (self + cross, early stopping)
  and saves checkpoints/stage2_align_<name><tag>.pt = W_new + Dec_new + robot spec.
  FR3, its decoder and the decision core are never modified.
  --pipeline joint  = the earlier single-stage training (InfoNCE + decoder + MMD).

Ablations (joint pipeline):
  --no_align        drop L_align
  --no_cross        drop cross-decoding term
  --no_mmd          drop L_dist
  --freeze_tgt_W    W_tgt = W_src, frozen (baseline: no alignment, decoder only)
"""
import os, sys, argparse, glob, json, math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.expanduser("~/cross_embody")
sys.path.insert(0, PROJECT_ROOT)
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import MLPEmbedding
from layer2.decoder import Decoder

# ============================ EDIT / VERIFY ============================
ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
SRC_CKPT = os.path.join(CKPT_DIR, "stage1_franka.pt")

SRC_NAME, SRC_DOF = "fr3", 7
TGT_NAME, TGT_DOF = "ur5e", 6          # VERIFY: name must exist in robots.yaml

FR3_JOINT_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
], dtype=np.float32)
# UR5e spec is ±2π; sampling ±π already covers every pose and avoids
# wrap-around ambiguity between q_ref and q_target.
UR5E_JOINT_LIMITS = np.array([[-np.pi, np.pi]] * 6, dtype=np.float32)
UR5E_VEL_LIMIT = np.full(6, np.pi, dtype=np.float32)   # rad/s (180°/s, UR5e spec)
# Generic target-robot spec; overwritten by setup_target() from the URDF at start-up.
# (UR5E_* names are kept as aliases because other scripts import them.)
TGT_JOINT_LIMITS = UR5E_JOINT_LIMITS
TGT_VEL_LIMIT = UR5E_VEL_LIMIT
TGT_WRAP = np.ones(6, dtype=bool)
CONTROL_HZ = 30.0                      # matches fr3_pick_gen (generate_fr3_expert_data.py)
# FR3 expert steps are far below the velocity limit: p99|dq| / (vlim/30)
# = 0.07-0.46 across joints, and the median step is ~0 (hold phase).
# UR5e synthetic steps: |dq_j| <= STEP_RATIO * vlim_j / CONTROL_HZ, with a
# per-sample scale u~U(0,1) so small/near-zero steps dominate like the data.
STEP_RATIO = 0.5

# UR5e base origin expressed in FR3 base frame (m). Zero = both bases
# placed at the same world pose, which is what zero-shot deployment assumes.
TGT_BASE_OFFSET = [0.0, 0.0, 0.0]

# VERIFY: where the 6D rotation lives inside fk21. Checked at startup
# (two unit, orthogonal 3-vectors); falls back to position-only if not.
ROT6D_SLICE = slice(3, 9)

EMB_IN, LATENT_DIM, M_DIM = 18, 128, 3
# ======================================================================

# pairing / positives
REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "fr3_pick_gen", "datasets",
                             "expert_v1", "data", "chunk-000")
N_UNIFORM_SRC = 150000 # uniform FR3 poses over the whole above-table workspace (main data)
N_EXPERT_SRC = 10000   # random subset of expert states (~6% of sources): guarantees the
                       # task region is covered without letting one goal dominate
TABLE_Z = 0.0          # m: drop source poses whose TCP is below the table
IK_SEEDS = 4
IK_ITERS = 600
IK_LR = 0.05
IK_CHUNK = 8192
IK_W_ROT = 0.1         # m: rotation weight INSIDE the IK objective (pose must match, not just position)
ACC_POS = 0.003        # m: accept an IK solution if position error < 3 mm ...
ACC_ANG_DEG = 3.0      # ... and orientation error < 3°
DEDUP_RAD = 0.05       # seeds closer than this (max joint diff) are the same branch
W_ROT = 0.02           # m per rad: orientation weight in d_phys / decoder loss (position-dominant)
DELTA = 0.015          # m: positive threshold on d = |Δp| + W_ROT·angle

# training
EPOCHS = 150
BATCH = 512
LR = 1e-3
LAMBDA1 = 0.01        # 1/mm²: decoder EE losses are in mm²; 1 cm (100 mm²) error -> 1.0 in total loss
LAMBDA2 = 1.0
ALPHA_J = 1000.0      # mm²/rad²: SELF joint term converted to mm² (1 rad ≈ 32 mm EE error)
TAU_SCALE = 0.1       # τ = TAU_SCALE * median pairwise ||z_src||² distance
SYNTH_STEP_FRACTION = 0.5   # q_ref: half velocity-limited steps, half q+N(0,Q_NOISE)
Q_NOISE = 0.03
VAL_FRAC = 0.1
EVAL_EVERY = 10
# EE-term curriculum: the EE loss through FK is highly non-convex in joint space
# (1 rad ≈ 500 mm of TCP motion) and drags a fresh decoder into whatever IK
# branch it lands in. Train joints only first (decoder learns to follow q_ref),
# then ramp the EE term in so it only refines locally inside q_ref's branch.
WARMUP_EPOCHS = 10    # decoder terms: joint only
RAMP_EPOCHS = 10      # then EE weight ramps linearly 0 -> 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_ROT = True


# ----------------------------- geometry -----------------------------
def r6_to_R(r6):
    a1, a2 = r6[..., :3], r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def pose(fk21, offset=None):
    p = fk21[:, :3]
    if offset is not None:
        p = p + offset
    R = r6_to_R(fk21[:, ROT6D_SLICE]) if USE_ROT else None
    return p, R


def geo_angle(tr):
    return torch.acos(((tr - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6))


def pose_dist_pairwise(pa, Ra, pb, Rb):
    d = torch.cdist(pa, pb)
    if USE_ROT:
        tr = Ra.flatten(1) @ Rb.flatten(1).T
        d = d + W_ROT * geo_angle(tr)
    return d


def pose_dist_paired(pa, Ra, pb, Rb):
    d = (pa - pb).norm(dim=-1)
    if USE_ROT:
        tr = (Ra * Rb).flatten(-2).sum(-1)
        d = d + W_ROT * geo_angle(tr)
    return d


def check_rot_slice(enc, lim):
    global USE_ROT
    if ROT6D_SLICE is None:
        USE_ROT = False
    else:
        r6 = enc.forward(sample_q(lim, 512))[:, ROT6D_SLICE]
        n1, n2 = r6[:, :3].norm(dim=1), r6[:, 3:6].norm(dim=1)
        dot = (r6[:, :3] * r6[:, 3:6]).sum(1)
        USE_ROT = bool(((n1 - 1).abs().max() < 1e-3) and
                       ((n2 - 1).abs().max() < 1e-3) and
                       (dot.abs().max() < 1e-3))
    if not USE_ROT:
        print("[warn] rot6d slice check failed -> POSITION-ONLY alignment. "
              "Fix ROT6D_SLICE to enable orientation terms.")
    else:
        print(f"[init] rot6d found at fk21[{ROT6D_SLICE.start}:{ROT6D_SLICE.stop}]")


def sample_q(lim, n):
    lim_t = torch.as_tensor(lim, device=DEVICE)
    u = torch.rand(n, lim_t.shape[0], device=DEVICE)
    return lim_t[:, 0] + u * (lim_t[:, 1] - lim_t[:, 0])


# --------------------------- pair generation ---------------------------
def wrap(q):
    return torch.atan2(torch.sin(q), torch.cos(q))


def setup_target(name, enc_tgt):
    """Read the new arm's joint limits / velocity limits / continuous joints from its
    URDF (robots.yaml may override with sample_limits / vel_limits)."""
    import yaml
    import xml.etree.ElementTree as ET
    global TGT_NAME, TGT_DOF, TGT_JOINT_LIMITS, TGT_VEL_LIMIT, TGT_WRAP
    global UR5E_JOINT_LIMITS, UR5E_VEL_LIMIT
    cfg = yaml.safe_load(open(ROBOTS_YAML))[name]
    dof = int(cfg["arm_dof"])
    names = enc_tgt.chain.get_joint_parameter_names()[:dof]
    lim, vel, cont = [], [], []
    if "mjcf" in cfg:                                    # MuJoCo model (e.g. Menagerie): ranges from the
        import mujoco                                     # compiled model; MJCF has no velocity limits ->
        from fk_encoder.fk_encoder import build_mjcf_robot_spec   # vel_limits from robots.yaml or pi rad/s
        mm = build_mjcf_robot_spec(cfg, PROJECT_ROOT).compile()
        for n in names:
            j = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)
            lo, hi = (mm.jnt_range[j] if mm.jnt_limited[j] else (-np.pi, np.pi))
            lo, hi = float(lo), float(hi)
            c = (hi - lo) >= 2 * np.pi - 1e-3
            if c:
                lo, hi = -np.pi, np.pi
            lim.append([lo, hi]); vel.append(np.pi); cont.append(c)
        names_done = True
    else:
        names_done = False
        root = ET.parse(os.path.join(PROJECT_ROOT, cfg["urdf"])).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
    for n in ([] if names_done else names):
        j = joints[n]
        l = j.find("limit")
        lo = float(l.get("lower", -np.pi)) if l is not None else -np.pi
        hi = float(l.get("upper", np.pi)) if l is not None else np.pi
        v = float(l.get("velocity", np.pi)) if l is not None else np.pi
        c = j.get("type") == "continuous" or (hi - lo) >= 2 * np.pi - 1e-3
        if c:                       # >= one full turn: sample one turn, wrap angles
            lo, hi = -np.pi, np.pi
        lim.append([lo, hi]); vel.append(v); cont.append(c)
    if "sample_limits" in cfg:
        lim = cfg["sample_limits"]
    if "vel_limits" in cfg:
        vel = cfg["vel_limits"]
    TGT_NAME, TGT_DOF = name, dof
    TGT_JOINT_LIMITS = np.asarray(lim, dtype=np.float32)
    TGT_VEL_LIMIT = np.asarray(vel, dtype=np.float32)
    TGT_WRAP = np.asarray(cont, dtype=bool)
    UR5E_JOINT_LIMITS, UR5E_VEL_LIMIT = TGT_JOINT_LIMITS, TGT_VEL_LIMIT
    print(f"[robot] {name}: dof={dof}  joints={names}")
    print(f"[robot] limits={np.round(TGT_JOINT_LIMITS, 3).tolist()}")
    print(f"[robot] vel={np.round(TGT_VEL_LIMIT, 3).tolist()} rad/s  continuous={TGT_WRAP.tolist()}")


def project_tgt(q):
    """wrap continuous joints to [-pi, pi], clamp the others to their limits"""
    lim = torch.as_tensor(TGT_JOINT_LIMITS, device=q.device)
    w = torch.as_tensor(TGT_WRAP, device=q.device)
    return torch.where(w, wrap(q), torch.max(torch.min(q, lim[:, 1]), lim[:, 0]))


def load_expert_q():
    files = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "episode_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No episodes in {REAL_DATA_DIR}")
    qs = [np.stack(pd.read_parquet(f)["observation.state"].values)[:, :SRC_DOF]
          for f in files]
    q = np.concatenate(qs).astype(np.float32)
    print(f"[pairs] {len(q)} FR3 expert states from {len(files)} episodes")
    return torch.as_tensor(q, device=DEVICE)


N_TABLETOP_SRC = 0     # extra FR3 poses with the TCP just above the table, gripper pointing down
                       # (set by --n_tabletop). Task-agnostic tabletop-manipulation prior: uniform
                       # joint sampling almost never puts the TCP near the table pointing down, which
                       # is exactly where grasping and placing happen.
FR3_HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]


def sample_tabletop_fr3(enc_src, n, box=((0.25, 0.70), (-0.35, 0.45), (0.015, 0.35)), tilt_deg=15.0):
    """FR3 joint configurations whose TCP is in a box above the table, pointing down (any yaw, tilt up
    to tilt_deg): random targets, Adam IK from HOME + noise, keep the converged ones."""
    lim = torch.as_tensor(FR3_JOINT_LIMITS, device=DEVICE, dtype=torch.float32)
    out = []
    while sum(len(o) for o in out) < n:
        m = min(IK_CHUNK, 2 * n)
        b = torch.as_tensor(box, device=DEVICE, dtype=torch.float32)
        p_t = b[:, 0] + torch.rand(m, 3, device=DEVICE) * (b[:, 1] - b[:, 0])
        p_t[:, 2] += TABLE_Z
        yaw = (torch.rand(m, device=DEVICE) * 2 - 1) * math.pi
        tilt = torch.rand(m, device=DEVICE) * math.radians(tilt_deg)
        tdir = torch.rand(m, device=DEVICE) * 2 * math.pi
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        Rz = torch.zeros(m, 3, 3, device=DEVICE); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s_; Rz[:, 1, 0] = s_
        Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1
        down = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=DEVICE))        # TCP z (approach) down
        ax = torch.stack([torch.cos(tdir), torch.sin(tdir), torch.zeros_like(tdir)], 1)
        K = torch.zeros(m, 3, 3, device=DEVICE)
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = ax[:, 1], -ax[:, 0], -ax[:, 1], ax[:, 0]
        st, ct = torch.sin(tilt)[:, None, None], torch.cos(tilt)[:, None, None]
        Rtilt = torch.eye(3, device=DEVICE) + st * K + (1 - ct) * (K @ K)
        R_t = Rtilt @ Rz @ down
        q = (torch.tensor(FR3_HOME, device=DEVICE) + 0.4 * torch.randn(m, 7, device=DEVICE))
        q = torch.max(torch.min(q, lim[:, 1]), lim[:, 0]).requires_grad_(True)
        opt = torch.optim.Adam([q], lr=IK_LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=IK_ITERS)
        for _ in range(IK_ITERS):
            pf, Rf = pose(enc_src.forward(q))
            loss = ((pf - p_t) ** 2).sum(1) + IK_W_ROT ** 2 * ((Rf - R_t) ** 2).flatten(1).sum(1)
            opt.zero_grad(); loss.sum().backward(); opt.step(); sch.step()
            with torch.no_grad():
                q.copy_(torch.max(torch.min(q, lim[:, 1]), lim[:, 0]))
        with torch.no_grad():
            pf, Rf = pose(enc_src.forward(q))
            ok = ((pf - p_t).norm(dim=1) < ACC_POS) & (geo_angle((Rf * R_t).flatten(-2).sum(-1))
                                                       < np.deg2rad(ACC_ANG_DEG))
            out.append(q.detach()[ok])
        print(f"[tabletop] FR3 IK: {sum(len(o) for o in out)}/{n} poses (this batch {ok.float().mean():.0%})")
    return torch.cat(out, 0)[:n]


def build_pairs(enc_src, enc_tgt, W_src, offset):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = (f"pairs_v3_{SRC_NAME}_{TGT_NAME}_u{N_UNIFORM_SRC}_e{N_EXPERT_SRC}_s{IK_SEEDS}"
           f"_rot{int(USE_ROT)}_p{ACC_POS}_a{ACC_ANG_DEG}"
           + (f"_t{N_TABLETOP_SRC}" if N_TABLETOP_SRC else "") + ".pt")
    path = os.path.join(CACHE_DIR, tag)
    if os.path.exists(path):
        print(f"[pairs] loading cache {path}")
        return torch.load(path, map_location=DEVICE)

    # --- source FR3 poses: all expert states + uniform poses above the table
    q_exp = load_expert_q()
    q_exp = q_exp[torch.randperm(len(q_exp), device=DEVICE)[:N_EXPERT_SRC]]
    q_uni = sample_q(FR3_JOINT_LIMITS, N_UNIFORM_SRC * 3)
    with torch.no_grad():
        z_ok = enc_src.forward(q_uni)[:, 2] > TABLE_Z
    q_uni = q_uni[z_ok][:N_UNIFORM_SRC]
    if N_TABLETOP_SRC:
        q_uni = torch.cat([q_uni, sample_tabletop_fr3(enc_src, N_TABLETOP_SRC)], 0)
    q_src_all = torch.cat([q_exp, q_uni], 0)
    is_exp_all = torch.cat([torch.ones(len(q_exp), dtype=torch.bool, device=DEVICE),
                            torch.zeros(len(q_uni), dtype=torch.bool, device=DEVICE)])
    N = len(q_src_all)
    acc_ang = np.deg2rad(ACC_ANG_DEG)
    out = {k: [] for k in ["q_tgt", "z_src", "p_src", "R_src", "is_exp", "src_id"]}
    n_src_ok = torch.zeros(N, dtype=torch.long, device=DEVICE)

    for c0 in range(0, N, IK_CHUNK):
        q_s = q_src_all[c0:c0 + IK_CHUNK]
        n = len(q_s)
        with torch.no_grad():
            fk_s = enc_src.forward(q_s)
            p_s, R_s = pose(fk_s)
            z_s = W_src(fk_s[:, :EMB_IN])
        p_goal = (p_s - offset).repeat_interleave(IK_SEEDS, 0)   # goal in UR5e base frame
        R_goal = R_s.repeat_interleave(IK_SEEDS, 0) if USE_ROT else None

        q = sample_q(TGT_JOINT_LIMITS, n * IK_SEEDS).requires_grad_(True)
        opt = torch.optim.Adam([q], lr=IK_LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=IK_ITERS)
        for _ in range(IK_ITERS):
            pt, Rt = pose(enc_tgt.forward(q))
            loss = ((pt - p_goal) ** 2).sum(1)
            if USE_ROT:
                loss = loss + IK_W_ROT ** 2 * ((Rt - R_goal) ** 2).flatten(1).sum(1)
            opt.zero_grad(); loss.sum().backward(); opt.step(); sch.step()
            with torch.no_grad():
                q.copy_(project_tgt(q))

        with torch.no_grad():
            pt, Rt = pose(enc_tgt.forward(q))
            ok = (pt - p_goal).norm(dim=1) < ACC_POS
            if USE_ROT:
                ok &= geo_angle((Rt * R_goal).flatten(-2).sum(-1)) < acc_ang
            q3, ok = q.detach().view(n, IK_SEEDS, -1), ok.view(n, IK_SEEDS)
            for a in range(IK_SEEDS):          # drop seeds that found the same branch
                for b in range(a):
                    same = wrap(q3[:, a] - q3[:, b]).abs().max(-1).values < DEDUP_RAD
                    ok[:, a] &= ~(ok[:, b] & same)
            src_i, seed_i = ok.nonzero(as_tuple=True)
            n_src_ok[c0:c0 + n] = ok.sum(1)

        out["q_tgt"].append(q3[src_i, seed_i])
        out["z_src"].append(z_s[src_i].unsqueeze(1))
        out["p_src"].append(p_s[src_i].unsqueeze(1))
        out["R_src"].append(R_s[src_i].reshape(-1, 1, 9) if USE_ROT
                            else torch.zeros(len(src_i), 1, 9, device=DEVICE))
        out["is_exp"].append(is_exp_all[c0:c0 + n][src_i])
        out["src_id"].append(src_i + c0)
        print(f"[pairs] {c0 + n}/{N}  FR3 poses with >=1 UR5e solution: "
              f"{(ok.sum(1) > 0).float().mean():.1%}")

    pairs = {k: torch.cat(v, 0) for k, v in out.items()}
    pairs["valid"] = torch.ones(len(pairs["q_tgt"]), 1, dtype=torch.bool, device=DEVICE)
    cov = n_src_ok > 0
    print(f"[pairs] coverage  expert: {cov[is_exp_all].float().mean():.1%}   "
          f"uniform: {cov[~is_exp_all].float().mean():.1%}")
    print(f"[pairs] UR5e anchors: {len(pairs['q_tgt'])}  "
          f"(distinct branches per covered FR3 pose: {n_src_ok[cov].float().mean():.2f})")
    torch.save(pairs, path)
    return pairs


# ---------------------- canonical FR3 targets (two-stage) ----------------------
MM_PER_LATENT = 347.0     # FR3 latent scale (latent_sensitivity.py): 1 latent unit ≈ 347 mm


def canonical_targets(enc_src, enc_tgt, W_src, pairs, ns_weight, ik_iters=600, refine_iters=300, seeds=8):
    """For every new-arm pose in `pairs`: MANY FR3 configurations reaching the SAME TCP pose
    (seed 0 = HOME_Q, the others uniformly random in FR3's joint range), each accepted at
    < 1 mm / < 1°, duplicates (< 0.1 rad apart) removed.
      z_all, kv_all, ok_all : every distinct FR3 'translation' of the pose (S per pose)
                              -> positives for the contrastive encoder
      z, kv, ok             : the one closest to HOME_Q (canonical) -> regression target
    """
    path = os.path.join(CACHE_DIR, f"canon3_{SRC_NAME}_{TGT_NAME}_n{len(pairs['q_tgt'])}_s{seeds}.pt")
    if os.path.exists(path):
        print(f"[fr3-solutions] loading cache {path}")
        return _smooth_canonical(torch.load(path, map_location=DEVICE))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "fr3_pick_gen", "scripts"))
    from generate_fr3_expert_data import HOME_Q
    lim = torch.tensor(FR3_JOINT_LIMITS, device=DEVICE)
    home = torch.tensor(np.asarray(HOME_Q)[:SRC_DOF], dtype=torch.float32, device=DEVICE)
    out = {k: [] for k in ["z_all", "kv_all", "ok_all", "z", "kv", "ok"]}
    CH = 2048
    for c0 in range(0, len(pairs["q_tgt"]), CH):
        with torch.no_grad():
            p_g, R_g = pose(enc_tgt.forward(pairs["q_tgt"][c0:c0 + CH]))
        n = len(p_g)
        p_g, R_g = p_g.repeat(seeds, 1), R_g.repeat(seeds, 1, 1)          # seed-major
        q = sample_q(FR3_JOINT_LIMITS, n * seeds)
        q[:n] = home
        q = q.requires_grad_(True)
        for iters, ns, lr in ((ik_iters, ns_weight, 0.02), (refine_iters, 0.0, 0.005)):
            opt = torch.optim.Adam([q], lr=lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
            for _ in range(iters):
                p, Rm = pose(enc_src.forward(q))
                loss = ((p - p_g) ** 2).sum(1) + 0.01 * ((Rm - R_g) ** 2).flatten(1).sum(1)
                if ns > 0:
                    loss = loss + ns * ((q - home) ** 2).sum(1)
                opt.zero_grad(); loss.sum().backward(); opt.step(); sch.step()
                with torch.no_grad():
                    q.copy_(torch.max(torch.min(q, lim[:, 1]), lim[:, 0]))
        with torch.no_grad():
            fk = enc_src.forward(q)
            p, Rm = pose(fk)
            ok = ((p - p_g).norm(dim=1) < 0.001) & \
                 (geo_angle((Rm * R_g).flatten(-2).sum(-1)) < np.radians(1.0))
            qs, ok = q.detach().view(seeds, n, -1), ok.view(seeds, n)
            for a_ in range(seeds):                                     # remove duplicate solutions
                for b_ in range(a_):
                    same = (qs[a_] - qs[b_]).abs().max(-1).values < 0.1
                    ok[a_] &= ~(ok[b_] & same)
            z = W_src(fk[:, :EMB_IN]).view(seeds, n, -1).permute(1, 0, 2)
            kv = fk[:, 9:18].view(seeds, n, -1).permute(1, 0, 2)
            okt = ok.t()                                                 # (n, seeds)
            d_home = (qs - home).norm(dim=-1).t().masked_fill(~okt, float("inf"))
            pick = d_home.argmin(1)
            ar = torch.arange(n, device=DEVICE)
            out["z_all"].append(z.half()); out["kv_all"].append(kv); out["ok_all"].append(okt)
            out["z"].append(z[ar, pick]); out["kv"].append(kv[ar, pick]); out["ok"].append(okt.any(1))
        print(f"[fr3-solutions] {c0 + n}/{len(pairs['q_tgt'])}  poses with >=1 solution "
              f"{okt.any(1).float().mean():.1%}  distinct FR3 solutions/pose {okt.sum(1).float().mean():.2f}")
    C = {k: torch.cat(v) for k, v in out.items()}
    torch.save(C, path)
    C = _smooth_canonical(C)
    print(f"[fr3-solutions] {C['ok'].float().mean():.1%} of poses covered, "
          f"{C['ok_all'].sum(1).float().mean():.2f} FR3 translations per pose -> {path}")
    return C


def _smooth_canonical(C):
    """Canonical (expert-style) translation = the solution found from the HOME_Q seed
    (seed 0), when it converged. 'Closest to HOME among all seeds' switches abruptly
    between neighbouring poses whenever a random seed happens to land nearer HOME,
    which makes the target discontinuous; the HOME-seeded solution varies smoothly
    (diag_encoder T2: neighbour ratio 1.3, 0.2% flips). Poses whose HOME seed failed
    keep the closest-to-HOME fallback."""
    home_ok = C["ok_all"][:, 0]
    C["z"] = torch.where(home_ok.unsqueeze(1), C["z_all"][:, 0].float(), C["z"])
    C["kv"] = torch.where(home_ok.unsqueeze(1), C["kv_all"][:, 0], C["kv"])
    print(f"[fr3-solutions] canonical = HOME-seeded solution for {home_ok.float().mean():.1%} of poses")
    return C


def tcp_neighbours(enc_tgt, q_tgt, k):
    """k nearest new-arm poses in TCP position (hard negatives for the contrastive encoder)."""
    with torch.no_grad():
        P = torch.cat([enc_tgt.forward(q_tgt[i:i + 16384])[:, :3] for i in range(0, len(q_tgt), 16384)])
    nb = []
    for i in range(0, len(P), 1024):
        d = torch.cdist(P[i:i + 1024], P)
        d[torch.arange(len(d), device=DEVICE), torch.arange(i, i + len(d), device=DEVICE)] = float("inf")
        nb.append(d.topk(k, largest=False).indices)
    return torch.cat(nb)


# ---------------------- structured encoder (two-stage default) ----------------------
class StructuredEncoder(torch.nn.Module):
    """W_new(x) = W_FR3([p, r6, v_SE', v_SW', v_SEE']) for the new arm's features x = fk21[:18].

    The TCP part (p, rot6d) is passed through UNCHANGED, and v_SEE = (TCP - S)/L is
    computed exactly from the TCP with FR3's fixed shoulder S and upper-arm length L.
    Only FR3's arm-shape key-vectors (v_SE, v_SW: where FR3's elbow and wrist would be)
    are predicted by a small network g. So the TCP information in the latent is exact
    by construction; learning only touches the elbow/wrist directions, to which the
    decision core is insensitive (FR3 null-space test: 100% SR with 0.38 rad elbow change).
    W_FR3 is a frozen copy of the Stage-1 embedding."""

    def __init__(self, W_src_state, s_pos, L, hidden=256, depth=3, learn_see=False):
        super().__init__()
        self.learn_see = learn_see
        self.W = MLPEmbedding(EMB_IN, LATENT_DIM)
        self.W.load_state_dict(W_src_state)
        for prm in self.W.parameters():
            prm.requires_grad_(False)
        self.register_buffer("S", torch.as_tensor(s_pos, dtype=torch.float32))
        self.register_buffer("L", torch.as_tensor(float(L)))
        self.register_buffer("mu", torch.zeros(EMB_IN))
        self.register_buffer("sd", torch.ones(EMB_IN))
        layers, k = [], EMB_IN
        for _ in range(depth):
            layers += [torch.nn.Linear(k, hidden), torch.nn.GELU()]; k = hidden
        self.g = torch.nn.Sequential(*layers, torch.nn.Linear(k, 9 if learn_see else 6))

    def kv(self, x):
        """predicted FR3 key-vectors [v_SE, v_SW] (6) + exact v_SEE (3)"""
        v = self.g((x - self.mu) / self.sd)
        if self.learn_see:                       # all three key-vectors learned
            return v
        v_see = (x[:, :3] - self.S) / self.L     # exact from the TCP
        return torch.cat([v, v_see], 1)

    def forward(self, x):
        return self.W(torch.cat([x[:, :9], self.kv(x)], 1))


def fr3_shoulder(enc_src):
    """FR3 shoulder key point S (must be configuration-independent) and L."""
    qs = sample_q(FR3_JOINT_LIMITS, 256)
    with torch.no_grad():
        pos, _ = enc_src._key_frames(qs)
    S = pos["shoulder"]
    spread = (S - S.mean(0)).norm(dim=1).max().item()
    assert spread < 1e-5, f"FR3 shoulder key point moves with q ({spread:.2e} m) -- cannot compute v_SEE exactly"
    return S.mean(0), enc_src.L.item()


def build_encoder_from_ckpt(ck, W_src_state):
    """Rebuild W_new from a stage-2 checkpoint (plain MLPEmbedding or StructuredEncoder)."""
    cfg = ck.get("config", {})
    if cfg.get("encoder_type") == "structured":
        enc = StructuredEncoder(W_src_state, cfg["fr3_S"], cfg["fr3_L"],
                                cfg.get("g_hidden", 256), cfg.get("g_depth", 3),
                                cfg.get("learn_see", False)).to(DEVICE)
    else:
        enc = MLPEmbedding(EMB_IN, LATENT_DIM).to(DEVICE)
    enc.load_state_dict(ck["W"]); enc.eval()
    return enc


def train_encoder_contrastive(W_tgt, enc_tgt, pairs, canon, tr_idx, val_uni, val_exp, args):
    """Stage A, contrastive ('translation'): multi-positive InfoNCE on the latent.
      anchor     z_new = W_new(new-arm pose)   (structured: TCP exact, g learns arm shape)
      positives  EVERY distinct FR3 configuration reaching the same TCP pose
      negatives  the FR3 configurations of the other anchors in the batch; batches are
                 built from TCP-neighbourhoods (k nearest poses) so negatives are hard
      similarity Gaussian kernel exp(-|z_new - z|² / 2σ²), σ = --enc_sigma
    Early stopping on the validation distance to the NEAREST own translation."""
    ok_all, z_all = canon["ok_all"], canon["z_all"]
    has = ok_all.any(1)
    tr = tr_idx[has[tr_idx]]
    vu, ve = val_uni[has[val_uni]], val_exp[has[val_exp]]
    with torch.no_grad():
        X = torch.cat([enc_tgt.forward(pairs["q_tgt"][i:i + 16384])[:, :EMB_IN]
                       for i in range(0, len(pairs["q_tgt"]), 16384)])
    if isinstance(W_tgt, StructuredEncoder):
        W_tgt.mu.copy_(X[tr].mean(0)); W_tgt.sd.copy_(X[tr].std(0) + 1e-6)
        params = list(W_tgt.g.parameters())
    else:
        params = list(W_tgt.parameters())
    print(f"[encoder] contrastive: σ={args.enc_sigma} (≈{args.enc_sigma * MM_PER_LATENT:.0f} mm)  "
          f"hard-negative neighbourhoods k={args.nbr_k}  "
          f"translations/pose={ok_all[tr].sum(1).float().mean():.2f}")
    nbr = tcp_neighbours(enc_tgt, pairs["q_tgt"], args.nbr_k)
    in_tr = torch.zeros(len(X), dtype=torch.bool, device=DEVICE); in_tr[tr] = True
    two_s2 = 2 * args.enc_sigma ** 2
    S = ok_all.shape[1]
    opt = torch.optim.Adam(params, lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6)

    def info_nce_pose(idx):
        """contrastive within one pose: candidates = this pose's FR3 translations.
        positives = translations in the decision core's 'dialect' (within pos_radius of
        the canonical, expert-like translation); negatives = the other translations of the
        SAME TCP pose (other elbows) -- exactly the part g controls."""
        z = W_tgt(X[idx])                                            # (B,128)
        cand = z_all[idx].float()                                    # (B,S,128)
        valid = ok_all[idx]
        d2 = ((z.unsqueeze(1) - cand) ** 2).sum(-1)                  # (B,S)
        to_canon = (cand - canon["z"][idx].unsqueeze(1)).norm(dim=-1)
        pos = valid & (to_canon < args.pos_radius)
        logits = (-d2 / two_s2).masked_fill(~valid, -1e9)
        lse_pos = torch.logsumexp(logits.masked_fill(~pos, -1e9), 1)
        nce = -lse_pos + torch.logsumexp(logits, 1)
        # InfoNCE alone only fixes RELATIVE distances: with candidates of one pose the logit
        # margin is linear in z, so z can drift arbitrarily far along (positive - negative)
        # and still "win". The pull term (-lse over positives ≈ d²_nearest-positive / 2σ²)
        # anchors z absolutely onto a dialect-correct translation.
        loss = nce + args.pull * (-lse_pos)
        top1 = pos.gather(1, logits.argmax(1, keepdim=True)).float().mean()
        has_neg = (valid & ~pos).any(1).float().mean()
        return loss.mean(), top1, has_neg

    def info_nce(idx):
        B = len(idx)
        z = W_tgt(X[idx])
        cand = z_all[idx].float().reshape(B * S, -1)
        valid = ok_all[idx].reshape(-1)
        owner = torch.arange(B, device=DEVICE).repeat_interleave(S)
        logits = (-(torch.cdist(z, cand) ** 2) / two_s2).masked_fill(~valid[None], -1e9)
        pos = (owner[None] == torch.arange(B, device=DEVICE)[:, None]) & valid[None]
        top1 = pos.gather(1, logits.argmax(1, keepdim=True)).float().mean()
        if args.enc_loss == "translation":
            # positives only: -log Σ_k exp(-|z - z_k|²/2σ²) over this pose's FR3 translations.
            # "be close to ANY valid translation", no repulsion from other poses (their
            # difference is the TCP, which g does not control and which is already exact).
            lp = logits.masked_fill(~pos, -1e9)
            loss = -torch.logsumexp(lp, 1) * two_s2           # scaled back to latent² units
            return loss.mean(), top1
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
        return loss.mean(), top1

    @torch.no_grad()
    def val(idx):
        W_tgt.eval()
        d = torch.cat([(W_tgt(X[i]).unsqueeze(1) - z_all[i].float()).norm(dim=-1)
                       .masked_fill(~ok_all[i], float("inf")).min(1).values
                       for i in [idx[j:j + 4096] for j in range(0, len(idx), 4096)]])
        W_tgt.train()
        return d

    @torch.no_grad()
    def val_canon(idx):
        W_tgt.eval()
        d = torch.cat([(W_tgt(X[i]) - canon["z"][i]).norm(dim=1)
                       for i in [idx[j:j + 4096] for j in range(0, len(idx), 4096)]])
        W_tgt.train()
        return d

    best, since, best_state = float("inf"), 0, None
    n_centres = max(1, args.enc_batch // (args.nbr_k + 1))
    for ep in range(args.enc_epochs):
        tot = acc = 0.0; nb = 0
        for _ in range(max(1, len(tr) // args.enc_batch)):
            c = tr[torch.randint(0, len(tr), (n_centres,), device=DEVICE)]
            idx = torch.cat([c, nbr[c].flatten()])
            idx = torch.unique(idx[in_tr[idx]])
            if ep < args.enc_warmup:          # canonical regression warm-up
                loss = ((W_tgt(X[idx]) - canon["z"][idx]) ** 2).sum(1).mean()
                top1 = torch.zeros(())
            elif args.enc_loss == "contrastive" and args.neg_scope == "pose":
                loss, top1, hn = info_nce_pose(idx)
            else:
                loss, top1 = info_nce(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); acc += top1.item(); nb += 1
        du, de = val(vu), val(ve)
        dcu, dce = val_canon(vu), val_canon(ve)
        m = (du if args.enc_loss == "translation" else dcu).mean().item()
        if ep == args.enc_warmup:
            best, since = float("inf"), 0
        sch.step(m)
        if m < best * 0.999:
            best, since = m, 0
            best_state = {k: v.clone() for k, v in W_tgt.state_dict().items()}
        else:
            since += 1
        if ep % 10 == 0 or since == 0 and ep % 5 == 0:
            print(f"[encoder ep {ep:4d}] {'warm-up' if ep < args.enc_warmup else args.enc_loss} loss={tot / nb:.4f}  top1={acc / nb:.1%}  "
                  f"val |z - nearest FR3 translation| workspace={m:.4f} (≈{m * MM_PER_LATENT:.1f} mm) "
                  f"p99={torch.quantile(du, 0.99):.4f}  expert={de.mean():.4f} "
                  f"(≈{de.mean() * MM_PER_LATENT:.1f} mm) | to expert-style translation: workspace "
                  f"{dcu.mean():.4f} (≈{dcu.mean() * MM_PER_LATENT:.1f} mm) expert {dce.mean():.4f} "
                  f"(≈{dce.mean() * MM_PER_LATENT:.1f} mm)  lr={opt.param_groups[0]['lr']:.1e}  since_best={since}")
        if since >= args.patience:
            break
    W_tgt.load_state_dict(best_state)
    du, de = val(vu), val(ve)
    dcu, dce = val_canon(vu), val_canon(ve)
    print(f"[encoder done] epoch {ep}: to nearest FR3 translation  workspace {du.mean():.4f} "
          f"(≈{du.mean() * MM_PER_LATENT:.1f} mm), expert {de.mean():.4f} (≈{de.mean() * MM_PER_LATENT:.1f} mm)")
    print(f"               to expert-style translation  workspace {dcu.mean():.4f} "
          f"(≈{dcu.mean() * MM_PER_LATENT:.1f} mm), expert {dce.mean():.4f} (≈{dce.mean() * MM_PER_LATENT:.1f} mm)")
    return (du if args.enc_loss == "translation" else dcu).mean().item()


def train_encoder(W_tgt, enc_tgt, pairs, canon, tr_idx, val_uni, val_exp, args):
    """Stage A, early stopping.
    MLPEmbedding     : regression W_new(x) -> canonical FR3 latent
    StructuredEncoder: regression g(x) -> canonical FR3 key-vectors [v_SE, v_SW]
    Reported metric in both cases: |W_new(x) - z_canon| in latent units and ≈ mm."""
    structured = isinstance(W_tgt, StructuredEncoder)
    nk = 9 if structured and W_tgt.learn_see else 6
    ok = canon["ok"]
    tr = tr_idx[ok[tr_idx]]
    vu, ve = val_uni[ok[val_uni]], val_exp[ok[val_exp]]
    with torch.no_grad():
        X = torch.cat([enc_tgt.forward(pairs["q_tgt"][i:i + 16384])[:, :EMB_IN]
                       for i in range(0, len(pairs["q_tgt"]), 16384)])
    Z, KV = canon["z"], canon["kv"]
    if structured:
        W_tgt.mu.copy_(X[tr].mean(0)); W_tgt.sd.copy_(X[tr].std(0) + 1e-6)
        params = list(W_tgt.g.parameters())
    else:
        params = list(W_tgt.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6)

    @torch.no_grad()
    def val(idx):
        W_tgt.eval(); d = (W_tgt(X[idx]) - Z[idx]).norm(dim=1)
        kv = (W_tgt.kv(X[idx])[:, :nk] - KV[idx, :nk]).norm(dim=1) if structured else None
        W_tgt.train()
        return d, kv

    best, since, best_state = float("inf"), 0, None
    for ep in range(args.enc_epochs):
        sh = tr[torch.randperm(len(tr), device=DEVICE)]
        tot, nb = 0.0, 0
        for b in range(0, len(sh) - 1024 + 1, 1024):
            i = sh[b:b + 1024]
            if structured:
                loss = ((W_tgt.kv(X[i])[:, :nk] - KV[i, :nk]) ** 2).sum(1).mean()
            else:
                loss = ((W_tgt(X[i]) - Z[i]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        (du, kvu), (de, _) = val(vu), val(ve)
        m = du.mean().item()
        sch.step(m)
        if m < best * 0.999:
            best, since = m, 0
            best_state = {k: v.clone() for k, v in W_tgt.state_dict().items()}
        else:
            since += 1
        if ep % 10 == 0 or since == 0 and ep % 5 == 0:
            extra = f"  kv err={kvu.mean():.4f}" if structured else ""
            print(f"[encoder ep {ep:4d}] train={tot / nb:.6f}  val workspace |dz| mean={m:.4f} "
                  f"(≈{m * MM_PER_LATENT:.1f} mm) p99={torch.quantile(du, 0.99):.4f}  "
                  f"expert={de.mean():.4f} (≈{de.mean() * MM_PER_LATENT:.1f} mm){extra}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}  since_best={since}")
        if since >= args.patience:
            break
    W_tgt.load_state_dict(best_state)
    (du, _), (de, _) = val(vu), val(ve)
    print(f"[encoder done] epoch {ep}: workspace |dz| mean={du.mean():.4f} (≈{du.mean() * MM_PER_LATENT:.1f} mm), "
          f"expert {de.mean():.4f} (≈{de.mean() * MM_PER_LATENT:.1f} mm)")
    return du.mean().item()


# ------------------------------- losses -------------------------------
def make_q_ref(q_target):
    n = q_target.shape[0]
    n_step = int(round(n * SYNTH_STEP_FRACTION))
    vmax = torch.as_tensor(STEP_RATIO * TGT_VEL_LIMIT / CONTROL_HZ, device=DEVICE)
    scale = torch.rand(n_step, 1, device=DEVICE)
    step = (torch.rand(n_step, TGT_DOF, device=DEVICE) * 2 - 1) * vmax * scale
    q_ref_step = q_target[:n_step] - step
    q_ref_noise = q_target[n_step:] + torch.randn_like(q_target[n_step:]) * Q_NOISE
    return torch.cat([q_ref_step, q_ref_noise], 0)


def ee_loss_mm(fk_pred, p_t, R_t):
    """L_EE in mm²: |Δp|² + (w_rot²/2)·||R - R*||_F²  ≈ |Δp|² + (w_rot·θ)²
    positions are FK metres -> ×1000; w_rot = W_ROT m/rad = 1000·W_ROT mm/rad."""
    p_p, R_p = pose(fk_pred)
    l = (((p_p - p_t) * 1000.0) ** 2).sum(1)
    if USE_ROT:
        l = l + ((W_ROT * 1000.0) ** 2) / 2.0 * ((R_p - R_t) ** 2).flatten(1).sum(1)
    return l.mean()


def mmd_rbf(x, y):
    xy = torch.cat([x, y], 0)
    d2 = torch.cdist(xy, xy) ** 2
    med = d2.detach().median().clamp(min=1e-8)
    k = sum(torch.exp(-d2 / (s * med)) for s in (0.25, 1.0, 4.0))
    n = x.shape[0]
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


RESIDUAL = False      # set from --residual in main()


DELTA_CTX = None      # (W_new, enc_tgt, scale) when --delta_input is on; set in main()


class DeltaInput(torch.nn.Module):
    """Feed the decoder the REQUESTED CHANGE instead of the absolute target latent:
         z_in = scale * (z_target - W_new(FK_new(q_ref)))
    A 1 mm request is only ~0.003 latent units inside a vector of norm ~1.2, so a decoder
    that sees the raw z cannot tell it from 'stay' and copies q_ref (dead zone ~10 mm).
    The difference is exactly zero when nothing should move and grows linearly with the
    requested motion; scaling makes 1-3 mm requests clearly visible.
    Same call signature as Decoder (z, m, q_prev) -> q; decoder.py is unchanged."""

    def __init__(self, dec, W_new, enc, scale):
        super().__init__()
        self.dec, self.W, self.enc, self.scale = dec, W_new, enc, float(scale)

    def z_in(self, z, q_ref):
        with torch.no_grad():
            z_ref = self.W(self.enc.forward(q_ref)[:, :EMB_IN])
        return (z - z_ref) * self.scale

    def forward(self, z, m, q_prev):
        return self.dec(self.z_in(z, q_prev), m, q_prev=q_prev)


def decode(dec, z, m, q_ref):
    """Residual option: q_hat = q_ref + Dec(z, m, q_ref) -> 'follow q_ref' is the
    default and the decoder only learns the correction. decoder.py is unchanged;
    the same wrapper must be used at deployment.
    Delta-input option (DELTA_CTX): the decoder sees scale*(z - W_new(FK(q_ref)))."""
    if DELTA_CTX is not None:
        W_new, enc, scale = DELTA_CTX
        with torch.no_grad():
            z_ref = W_new(enc.forward(q_ref)[:, :EMB_IN])
        z = (z - z_ref) * scale
    out = dec(z, m, q_prev=q_ref)
    return q_ref + out if RESIDUAL else out


def pick_one_positive(valid):
    w = valid.float() + 1e-9
    return torch.multinomial(w, 1).squeeze(1)


# ------------------------------ evaluation ------------------------------
@torch.no_grad()
def evaluate(pairs, idx, W_tgt, dec, enc_tgt, m_t, label="all"):
    W_tgt.eval(); dec.eval()
    q_target = pairs["q_tgt"][idx]
    n = q_target.shape[0]
    q_ref = make_q_ref(q_target)
    fk_t = enc_tgt.forward(q_target)
    p_t, R_t = pose(fk_t)
    z_new = W_tgt(fk_t[:, :EMB_IN])
    m_b = m_t.expand(n, -1)

    def ee_err(q_pred):
        p_p, R_p = pose(enc_tgt.forward(q_pred))
        e = (p_p - p_t).norm(dim=1) * 1000
        a = geo_angle((R_p * R_t).flatten(-2).sum(-1)) * 180 / np.pi if USE_ROT else None
        return e, a

    e_self, _ = ee_err(decode(dec, z_new, m_b, q_ref))
    j = pick_one_positive(pairs["valid"][idx])
    z_pos = pairs["z_src"][idx][torch.arange(n, device=DEVICE), j]
    q_cross = decode(dec, z_pos, m_b, q_ref)
    e_cross, a_cross = ee_err(q_cross)
    j_cross = (q_cross - q_target).norm(dim=1)
    j_self = (decode(dec, z_new, m_b, q_ref) - q_target).norm(dim=1)

    # top-1 latent retrieval among all val FR3 candidates
    z_c = pairs["z_src"][idx].flatten(0, 1)
    v_c = pairs["valid"][idx].flatten()
    p_c = pairs["p_src"][idx].flatten(0, 1)
    R_c = pairs["R_src"][idx].flatten(0, 1).view(-1, 3, 3) if USE_ROT else None
    hits = []
    for a0 in range(0, n, 1000):
        zs = z_new[a0:a0 + 1000]
        dz = torch.cdist(zs, z_c).masked_fill(~v_c[None], float("inf"))
        nn_idx = dz.argmin(1)
        Rq = R_t[a0:a0 + 1000] if USE_ROT else None
        Rn = R_c[nn_idx] if USE_ROT else None
        hits.append(pose_dist_paired(p_t[a0:a0 + 1000], Rq, p_c[nn_idx], Rn) < DELTA)
    top1 = torch.cat(hits).float().mean()

    def s(e):
        return (f"mean={e.mean():.2f} median={e.median():.2f} "
                f"p99={torch.quantile(e, 0.99):.2f}")
    print(f"  [val/{label}] self  EE (mm): {s(e_self)}")
    print(f"  [val/{label}] cross EE (mm): {s(e_cross)}")
    if USE_ROT:
        print(f"  [val/{label}] cross rot (deg): {s(a_cross)}")
    print(f"  [val/{label}] joint |q-q*| (rad)  self: {s(j_self)}   cross: {s(j_cross)}")
    print(f"  [val/{label}] cross branch switch (|q-q*| > 0.5 rad): {(j_cross > 0.5).float().mean():.1%}")
    print(f"  [val/{label}] latent top-1 retrieval: {top1:.1%}")
    W_tgt.train(); dec.train()
    return e_cross.mean().item()


@torch.no_grad()
def quick_cross_ee(pairs, idx, W_tgt, dec, enc_tgt, m_t, n_max=4000):
    """Cheap per-epoch monitor: mean cross EE (mm) on a fixed val subset."""
    dec.eval()
    idx = idx[:n_max]
    q_target = pairs["q_tgt"][idx]
    g = torch.Generator(device=DEVICE).manual_seed(0)          # same q_ref every epoch
    n = len(q_target); n_step = int(round(n * SYNTH_STEP_FRACTION))
    vmax = torch.as_tensor(STEP_RATIO * TGT_VEL_LIMIT / CONTROL_HZ, device=DEVICE)
    u = torch.rand(n_step, 1, device=DEVICE, generator=g)
    step = (torch.rand(n_step, TGT_DOF, device=DEVICE, generator=g) * 2 - 1) * vmax * u
    noise = torch.randn(n - n_step, TGT_DOF, device=DEVICE, generator=g) * Q_NOISE
    q_ref = torch.cat([q_target[:n_step] - step, q_target[n_step:] + noise], 0)
    j = pick_one_positive(pairs["valid"][idx])
    z_pos = pairs["z_src"][idx][torch.arange(n, device=DEVICE), j]
    q_hat = decode(dec, z_pos, m_t.expand(n, -1), q_ref)
    e = (enc_tgt.forward(q_hat)[:, :3] - enc_tgt.forward(q_target)[:, :3]).norm(dim=1) * 1000
    dec.train()
    return e.mean().item()


# -------------------------------- main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt", default="ur5e", help="new arm, as named in configs/robots.yaml")
    ap.add_argument("--n_tabletop", type=int, default=0,
                    help="add this many FR3 poses with the TCP just above the table, pointing down (any yaw) "
                         "to the pairs -- covers grasping / placing poses; new pairs cache")
    ap.add_argument("--pipeline", choices=["two_stage", "joint"], default="two_stage")
    ap.add_argument("--canon_ns", type=float, default=0.0, help="pull toward FR3 HOME_Q for canonical targets")
    ap.add_argument("--enc_epochs", type=int, default=2000)
    ap.add_argument("--encoder", choices=["structured", "mlp"], default="structured",
                    help="structured: TCP passed through W_FR3 exactly, only FR3 elbow/wrist key-vectors "
                         "are learned; mlp: plain MLPEmbedding regression onto the canonical latent")
    ap.add_argument("--enc_loss", choices=["translation", "contrastive", "regression"], default="translation",
                    help="translation: positives-only soft-min over all FR3 translations of a pose "
                         "(any valid translation is correct, no repulsion); "
                         "contrastive: multi-positive InfoNCE with hard negatives; "
                         "regression: L2 to the canonical (closest-to-HOME) FR3 translation")
    ap.add_argument("--encoder_only", action="store_true",
                    help="train the dictionary (encoder) only and skip the learned decoder")
    ap.add_argument("--use_expert_pairs", action="store_true",
                    help="also TRAIN on the 10k expert-state pairs (default: they are only used for "
                         "validation, so the dictionary does not depend on any task's expert data)")
    ap.add_argument("--neg_scope", choices=["pose", "batch"], default="pose",
                    help="contrastive negatives: 'pose' = other-elbow translations of the SAME TCP pose "
                         "(what g controls); 'batch' = translations of other poses (v9 behaviour)")
    ap.add_argument("--pull", type=float, default=1.0,
                    help="contrastive: weight of the absolute pull onto the positives (0 = pure InfoNCE)")
    ap.add_argument("--pos_radius", type=float, default=0.07,
                    help="contrastive positives: translations within this latent distance of the "
                         "expert-style one (0.07 ≈ elbow change of 0.3 rad the decision core tolerates)")
    ap.add_argument("--enc_warmup", type=int, default=10,
                    help="epochs of canonical regression before the translation/contrastive loss")
    ap.add_argument("--enc_sigma", type=float, default=0.03,
                    help="InfoNCE kernel width in latent units (0.03 ≈ 10 mm of TCP motion)")
    ap.add_argument("--enc_batch", type=int, default=512)
    ap.add_argument("--nbr_k", type=int, default=7, help="TCP neighbours per batch centre (hard negatives)")
    ap.add_argument("--fr3_seeds", type=int, default=8, help="FR3 IK seeds per pose (max translations per pose)")
    ap.add_argument("--learn_see", action="store_true",
                    help="structured encoder: also LEARN v_SEE instead of computing it from the TCP")
    ap.add_argument("--canon_mix", type=float, default=0.5,
                    help="two_stage: share of cross-decoding samples that use the canonical FR3 latent "
                         "(rest use the random-elbow FR3 latent of the pair)")
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--no_cross", action="store_true")
    ap.add_argument("--no_mmd", action="store_true")
    ap.add_argument("--freeze_tgt_W", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS,
                    help="epochs with joint-only decoder loss before the EE term ramps in")
    ap.add_argument("--l2_pair", type=float, default=0.0,
                    help="λ3: weight of a direct ||z_new - z_pair||² term on the exact IK pairs")
    ap.add_argument("--rel_loss", type=float, default=1.0,
                    help="λ_rel: relative-correction loss ||(q_hat-q_ref)-(q-q_ref)||² / (||q-q_ref||² + eps), "
                         "dimensionless (≈ remain²). Penalises 'copy q_ref' equally for small and large steps")
    ap.add_argument("--rel_eps", type=float, default=0.01,
                    help="rad; floor on the step size in the rel_loss denominator")
    ap.add_argument("--delta_mm", type=float, default=None,
                    help="override the positive threshold δ (mm) used by L_align")
    ap.add_argument("--dec_hidden", type=int, default=512, help="UR5e decoder width (same Decoder class)")
    ap.add_argument("--dec_depth", type=int, default=3, help="UR5e decoder hidden layers")
    ap.add_argument("--epochs", type=int, default=1000, help="maximum epochs (early stopping ends sooner)")
    ap.add_argument("--patience", type=int, default=30,
                    help="stop after this many epochs without improving the monitored val metric")
    ap.add_argument("--min_delta_mm", type=float, default=0.1, help="improvement smaller than this does not count")
    ap.add_argument("--lr_patience", type=int, default=10, help="halve lr after this many epochs without improvement")
    ap.add_argument("--lr_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--delta_input", action="store_true", default=True,
                    help="decoder input = scaled (z_target - W_new(FK(q_ref))) instead of z_target "
                         "(removes the small-step dead zone)")
    ap.add_argument("--no_delta_input", dest="delta_input", action="store_false")
    ap.add_argument("--residual", action="store_true",
                    help="decoder predicts a correction: q_hat = q_ref + Dec(...)")
    ap.add_argument("--no_cross_joint", action="store_true",
                    help="cross term supervises EE only (old behaviour)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--select", choices=["workspace", "expert"], default="workspace",
                    help="which val subset picks the best checkpoint")
    args = ap.parse_args()
    global RESIDUAL, DELTA
    RESIDUAL = args.residual
    if args.delta_mm is not None:
        DELTA = args.delta_mm / 1000.0
    print(f"[config] delta={DELTA*1000:.1f} mm  l2_pair={args.l2_pair}  rel_loss={args.rel_loss}  "
          f"decoder={args.dec_hidden}x{args.dec_depth}  epochs={args.epochs}")
    torch.manual_seed(0); np.random.seed(0)

    enc_src = FKEncoder(SRC_NAME, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    globals()["N_TABLETOP_SRC"] = args.n_tabletop
    enc_tgt = FKEncoder(args.tgt, ROBOTS_YAML, project_root=PROJECT_ROOT, device=DEVICE)
    setup_target(args.tgt, enc_tgt)
    if args.pipeline == "two_stage":
        args.freeze_tgt_W = True          # W_new is trained in stage A, frozen for the decoder
    check_rot_slice(enc_src, FR3_JOINT_LIMITS)
    offset = torch.tensor(TGT_BASE_OFFSET, device=DEVICE)

    ck = torch.load(SRC_CKPT, map_location=DEVICE)
    W_src = MLPEmbedding(EMB_IN, LATENT_DIM).to(DEVICE)
    W_src.load_state_dict(ck["W"]); W_src.eval()
    for p in W_src.parameters():
        p.requires_grad_(False)

    pairs = build_pairs(enc_src, enc_tgt, W_src, offset)
    N = pairs["q_tgt"].shape[0]
    src_ids = pairs["src_id"].unique()
    val_src = src_ids[torch.randperm(len(src_ids), device=DEVICE)[:int(len(src_ids) * VAL_FRAC)]]
    is_val = torch.isin(pairs["src_id"], val_src)
    val_idx = is_val.nonzero().squeeze(1)
    tr_idx = (~is_val).nonzero().squeeze(1)
    val_exp = val_idx[pairs["is_exp"][val_idx]]
    val_uni = val_idx[~pairs["is_exp"][val_idx]]
    if not args.use_expert_pairs:
        tr_idx = tr_idx[~pairs["is_exp"][tr_idx]]
        print(f"[data] task-independent dictionary: training on {len(tr_idx)} workspace pairs only "
              f"(expert-state pairs kept for validation)")
    n_val = len(val_idx)

    zs = pairs["z_src"][pairs["valid"]]
    sub = zs[torch.randperm(zs.shape[0], device=DEVICE)[:4096]]
    tau = TAU_SCALE * (torch.cdist(sub, sub) ** 2).median().item()
    print(f"[init] τ = {tau:.4g}")

    W_tgt = MLPEmbedding(EMB_IN, LATENT_DIM).to(DEVICE)
    W_tgt.load_state_dict(ck["W"])           # warm start from FR3
    canon, enc_val_dz, enc_cfg = None, None, {"encoder_type": "mlp"}
    if args.pipeline == "two_stage" and args.encoder == "structured":
        S_fr3, L_fr3 = fr3_shoulder(enc_src)
        W_tgt = StructuredEncoder(ck["W"], S_fr3, L_fr3, learn_see=args.learn_see).to(DEVICE)
        enc_cfg = {"encoder_type": "structured", "fr3_S": S_fr3.tolist(), "fr3_L": L_fr3,
                   "g_hidden": 256, "g_depth": 3, "learn_see": args.learn_see}
        print(f"[encoder] structured: FR3 shoulder S={np.round(S_fr3.cpu().numpy(), 4)}  L={L_fr3:.4f}")
    if args.pipeline == "two_stage":
        canon = canonical_targets(enc_src, enc_tgt, W_src, pairs, args.canon_ns, seeds=args.fr3_seeds)
        if args.enc_loss in ("contrastive", "translation"):
            enc_val_dz = train_encoder_contrastive(W_tgt, enc_tgt, pairs, canon, tr_idx, val_uni, val_exp, args)
        else:
            enc_val_dz = train_encoder(W_tgt, enc_tgt, pairs, canon, tr_idx, val_uni, val_exp, args)
        enc_cfg.update({"enc_loss": args.enc_loss, "enc_sigma": args.enc_sigma})
    dec = Decoder(LATENT_DIM, m_dim=M_DIM, out_dim=TGT_DOF, hidden=args.dec_hidden,
                  depth=args.dec_depth, use_q_prev=True).to(DEVICE)
    m_t = enc_tgt.m.to(DEVICE).unsqueeze(0)

    delta_scale = None
    if args.delta_input:
        # scale = 1 / typical per-dim |z_target - z_ref| for the q_ref steps seen in training
        with torch.no_grad():
            ii = tr_idx[torch.randperm(len(tr_idx), device=DEVICE)[:8192]]
            qt = pairs["q_tgt"][ii]; qr = make_q_ref(qt)
            jz = pick_one_positive(pairs["valid"][ii])
            zt = pairs["z_src"][ii][torch.arange(len(ii), device=DEVICE), jz]
            dz = zt - W_tgt(enc_tgt.forward(qr)[:, :EMB_IN])
            delta_scale = 1.0 / (dz.std(0).mean().item() + 1e-8)
        global DELTA_CTX
        DELTA_CTX = (W_tgt, enc_tgt, delta_scale)
        print(f"[decoder] delta input: z_in = {delta_scale:.1f} * (z - W_new(FK(q_ref)))  "
              f"(1 mm request ≈ {0.0029 * delta_scale / math.sqrt(LATENT_DIM):.3f} per dim after scaling)")
    params = list(dec.parameters())
    if args.freeze_tgt_W:
        for p in W_tgt.parameters():
            p.requires_grad_(False)
    else:
        params += list(W_tgt.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    # No fixed-length schedule: lr is halved when the monitored val metric stops
    # improving, and training stops after `patience` epochs without improvement
    # (counted only after warm-up + EE ramp, when the metric is meaningful).
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    since_best, history = 0, []
    hist_path = os.path.join(PROJECT_ROOT, "validation_out", f"stage2_{TGT_NAME}{args.tag}_hist.json")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)

    use_align = not args.no_align and not args.freeze_tgt_W
    use_mmd = not args.no_mmd and not args.freeze_tgt_W
    print(f"[config] align={use_align} cross={not args.no_cross} mmd={use_mmd} "
          f"freeze_tgt_W={args.freeze_tgt_W}  train={len(tr_idx)} val={n_val}")

    best, n_tr = float("inf"), len(tr_idx)
    ckpt_path = os.path.join(CKPT_DIR, f"stage2_align_{TGT_NAME}{args.tag}.pt")

    if args.encoder_only:
        # dictionary only: the arm is decoded by "meaning" lookup + its own IK
        # (decision_core.model.latent_to_meaning / tcp_ik), no learned decoder needed.
        torch.save({"W": W_tgt.state_dict(), "decoder": dec.state_dict(),
                    "latent_dim": LATENT_DIM, "tau": tau, "epoch": -1,
                    "config": {"residual": False, "dec_hidden": args.dec_hidden,
                               "dec_depth": args.dec_depth, "delta_scale": delta_scale,
                               "decoder_trained": False, "tgt": TGT_NAME, "dof": TGT_DOF,
                               "pipeline": args.pipeline,
                               "joint_limits": TGT_JOINT_LIMITS.tolist(),
                               "vel_limits": TGT_VEL_LIMIT.tolist(),
                               "continuous": TGT_WRAP.tolist(),
                               "encoder_val_dz": enc_val_dz, **enc_cfg, "args": vars(args)}},
                   ckpt_path)
        print(f"[done] dictionary (encoder) only -> {ckpt_path}  (decoder NOT trained; "
              f"evaluate with --ur_decoder meaning)")
        return

    for ep in range(args.epochs):
        sh = tr_idx[torch.randperm(n_tr, device=DEVICE)]
        acc = {"align": 0.0, "self": 0.0, "cross": 0.0, "mmd": 0.0}
        if ep < args.warmup_epochs:
            ee_w = 0.0
        else:
            ee_w = min(1.0, (ep - args.warmup_epochs + 1) / max(1, RAMP_EPOCHS))
        nb = 0
        for b0 in range(0, n_tr - BATCH + 1, BATCH):
            idx = sh[b0:b0 + BATCH]
            q_target = pairs["q_tgt"][idx]
            q_ref = make_q_ref(q_target)
            fk_t = enc_tgt.forward(q_target)
            p_t, R_t = pose(fk_t)
            p_t_src_frame = p_t + offset
            z_new = W_tgt(fk_t[:, :EMB_IN])
            m_b = m_t.expand(BATCH, -1)

            z_c = pairs["z_src"][idx].flatten(0, 1)
            v_c = pairs["valid"][idx].flatten()
            loss = torch.zeros((), device=DEVICE)

            # ---- L_align: multi-positive InfoNCE on latent distance ----
            if use_align:
                p_c = pairs["p_src"][idx].flatten(0, 1)
                R_c = pairs["R_src"][idx].flatten(0, 1).view(-1, 3, 3) if USE_ROT else None
                d_phys = pose_dist_pairwise(p_t_src_frame, R_t, p_c, R_c)
                pos = (d_phys < DELTA) & v_c[None]
                logits = -(torch.cdist(z_new, z_c) ** 2) / tau
                logits = logits.masked_fill(~v_c[None], -1e9)
                logp = logits - torch.logsumexp(logits, 1, keepdim=True)
                l_align = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
                l_align = l_align[pos.any(1)].mean()
                loss = loss + l_align
                acc["align"] += l_align.item()

            # ---- L_decoder: self and cross, joint term + (curriculum) EE term ----
            q_self = decode(dec, z_new, m_b, q_ref)
            l_self = ee_w * ee_loss_mm(enc_tgt.forward(q_self), p_t, R_t) + \
                ALPHA_J * ((q_self - q_target) ** 2).sum(1).mean()
            l_dec = l_self
            acc["self"] += l_self.item()

            j = pick_one_positive(pairs["valid"][idx])
            z_pos = pairs["z_src"][idx][torch.arange(BATCH, device=DEVICE), j]
            if canon is not None and args.canon_mix > 0:
                okb = canon["ok_all"][idx]
                jj = torch.multinomial(okb.float() + 1e-9, 1).squeeze(1)       # a random FR3 translation
                z_tr = canon["z_all"][idx, jj].float()
                use_c = (torch.rand(BATCH, device=DEVICE) < args.canon_mix) & okb.any(1)
                z_pos = torch.where(use_c.unsqueeze(1), z_tr, z_pos)
            if not args.no_cross:
                q_cross = decode(dec, z_pos, m_b, q_ref)
                l_cross = ee_w * ee_loss_mm(enc_tgt.forward(q_cross), p_t, R_t)
                if not args.no_cross_joint:
                    # q_ref was generated from THIS anchor's branch, so the joint
                    # answer is unique: the decoder must follow q_ref's branch.
                    l_cross = l_cross + ALPHA_J * ((q_cross - q_target) ** 2).sum(1).mean()
                l_dec = l_dec + l_cross
                acc["cross"] += l_cross.item()
                if args.rel_loss > 0:
                    # deployment asks for small steps (~0.02 rad / ~8 mm); with absolute losses,
                    # copying q_ref is almost free there. Normalising by the requested step makes
                    # "move the right fraction in the right direction" matter at every step size.
                    d_true = q_target - q_ref
                    d_hat = q_cross - q_ref
                    l_rel = (((d_hat - d_true) ** 2).sum(1) /
                             ((d_true ** 2).sum(1) + args.rel_eps ** 2)).mean()
                    loss = loss + args.rel_loss * l_rel
                    acc["rel"] = acc.get("rel", 0.0) + l_rel.item()
            loss = loss + LAMBDA1 * l_dec

            # ---- L_dist: MMD between UR5e latents and FR3 latents ----
            if args.l2_pair > 0 and not args.freeze_tgt_W:
                # the IK pairs are exact (< 3 mm, < 3°): pull each UR5e latent
                # onto its own FR3 partner, sharper than InfoNCE's δ-wide positives
                l_pair = ((z_new - z_pos) ** 2).sum(1).mean()
                loss = loss + args.l2_pair * l_pair
                acc["pair"] = acc.get("pair", 0.0) + l_pair.item()

            if use_mmd:
                l_mmd = mmd_rbf(z_new, z_pos)
                loss = loss + LAMBDA2 * l_mmd
                acc["mmd"] += l_mmd.item()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            nb += 1
        # ---- per-epoch monitor (cheap, fixed q_ref) ----
        mon_idx = val_uni if args.select == "workspace" else val_exp
        e = quick_cross_ee(pairs, mon_idx, W_tgt, dec, enc_tgt, m_t)
        ramp_done = ep >= args.warmup_epochs + RAMP_EPOCHS - 1
        if ep == args.warmup_epochs + RAMP_EPOCHS - 1:
            best = float("inf")      # full loss active from here: start the early-stopping race fresh
        improved = e < best - args.min_delta_mm
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": ep, "lr": lr_now, "ee_w": ee_w, "val_cross_ee_mm": e,
                        **{k: v / nb for k, v in acc.items()}})
        if improved or not ramp_done:
            if improved:
                best = e
                torch.save({"W": W_tgt.state_dict(), "decoder": dec.state_dict(),
                            "latent_dim": LATENT_DIM, "tau": tau, "epoch": ep,
                            "config": {"lambda1": LAMBDA1, "lambda2": LAMBDA2,
                                       "delta": DELTA, "w_rot": W_ROT,
                                       "use_rot": USE_ROT, "residual": RESIDUAL,
                                       "dec_hidden": args.dec_hidden, "dec_depth": args.dec_depth,
                                       "l2_pair": args.l2_pair, "rel_loss": args.rel_loss,
                                       "delta_scale": delta_scale,
                                       "tgt": TGT_NAME, "dof": TGT_DOF, "pipeline": args.pipeline,
                                       "joint_limits": TGT_JOINT_LIMITS.tolist(),
                                       "vel_limits": TGT_VEL_LIMIT.tolist(),
                                       "continuous": TGT_WRAP.tolist(),
                                       "encoder_val_dz": enc_val_dz, **enc_cfg,
                                       "args": vars(args)}},
                           ckpt_path)
            since_best = 0
        else:
            since_best += 1
        if ramp_done:
            sched.step(e)

        if ep % EVAL_EVERY == 0 or since_best == 0 and improved and ep % 5 == 0:
            msg = "  ".join(f"{k}={v / nb:.4f}" for k, v in acc.items())
            print(f"[ep {ep:3d}] lr={lr_now:.2e}  ee_w={ee_w:.2f}  {msg}  "
                  f"| monitor={e:.2f} mm best={best:.2f} since_best={since_best}")
            if ep % EVAL_EVERY == 0:
                evaluate(pairs, val_uni, W_tgt, dec, enc_tgt, m_t, "workspace")
                evaluate(pairs, val_exp, W_tgt, dec, enc_tgt, m_t, "expert")

        with open(hist_path, "w") as f:
            json.dump(history, f)

        if ramp_done and since_best >= args.patience:
            print(f"[stop] no improvement > {args.min_delta_mm} mm for {args.patience} epochs "
                  f"(epoch {ep}, best {best:.2f} mm)")
            break
    else:
        print(f"[stop] reached max epochs {args.epochs} -- may NOT be converged")

    # final report on the best checkpoint
    ck_best = torch.load(ckpt_path, map_location=DEVICE)
    W_tgt.load_state_dict(ck_best["W"]); dec.load_state_dict(ck_best["decoder"])
    print(f"\n[best checkpoint] epoch {ck_best['epoch']}  monitor={best:.2f} mm  -> {ckpt_path}")
    evaluate(pairs, val_uni, W_tgt, dec, enc_tgt, m_t, "workspace")
    evaluate(pairs, val_exp, W_tgt, dec, enc_tgt, m_t, "expert")
    print(f"[history] {hist_path}")

if __name__ == "__main__":
    main()