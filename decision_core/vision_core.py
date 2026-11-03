"""
Decision core with VISION object tokens (separate file; model.py and the chunk core are unchanged).

Object token = [ keypoint , appearance/state latent ]
  keypoint : where the image encoder finds the object in the IMAGE (soft-argmax of its heatmap,
             normalized u/W, v/H) + its visibility, given to the core as a Fourier position
             encoding. Pixel position found by the encoder itself: no world coordinates, no camera
             calibration, no labels at deployment (spatial-softmax keypoints, Finn et al. 2016,
             as fed to the policy in robomimic / Diffusion Policy).
  latent z : the encoder's per-object latent (shape, grasped or not, ...); optional (--feat kp).
  The two parts are embedded separately and summed, so the position cannot be washed out by z.

memory (history encoder, bidirectional like the chunk core):
    T arm tokens   = proprio latent_t + gripper embedding_t                    (unchanged)
    T cube tokens  = ObjEmbed(cube token_t)                                    (object state)
    + time embedding(t) + type embedding(arm | cube)
goal query (the goal is the query, it cannot be bypassed):
    K queries      = MLP([ObjEmbed(tape token at a frame where it is seen), ObjEmbed(cube now), finger])
                     + chunk position
outputs           = z_hat (B,K,128) for each arm's decoder + gripper logits (B,K)      (unchanged)

TCP token (use_tcp): where the arm's OWN TCP is in the image -- the TCP from the shared latent's FK,
projected with the calibrated camera (intrinsics + hand-eye extrinsics). Not detected in the image, so
it does not depend on what the gripper looks like (FR3 -> UR5e). Per frame:
    [kp_features(tcp_uv), REL_SCALE * (cube_uv - tcp_uv), REL_SCALE * (tape_uv - tcp_uv)]
so 'how far the gripper is from the cube / place point in the image' is an input, not something the
core has to work out from joint latents and pixels.

init_from_chunk(): copies the history blocks, chunk decoder, z_head and gripper embedding from a
trained chunk checkpoint (only if wanted -- a core that starts from the fixed-target pp2s prior
may keep ignoring the objects).
"""
import math

import torch
import torch.nn as nn

from decision_core.model import TrunkBlock, D_MODEL, N_LAYERS, N_HEADS, FFN_MULT, T_HISTORY, FINGER_SCALE

ARM, CUBE = 0, 1
LATENT = 128          # shared arm latent
REL_SCALE = 5.0
# coord = "xyz": object / TCP positions in the robot base frame (two calibrated cameras, triangulated),
# normalized as (p - XYZ_C) / XYZ_S (workspace -> about [-1, 1]); relative vectors cube - TCP, tape - TCP
# are given in units of REL_M metres, so 1 cm = 0.2
XYZ_C = (0.475, 0.025, 0.15)
XYZ_S = (0.30, 0.40, 0.30)
REL_M = 0.05


def xyz_denorm(x):
    """inverse of xyz_norm: the first 3 token features -> metres"""
    return x * torch.tensor(XYZ_S, device=x.device) + torch.tensor(XYZ_C, device=x.device)


def xyz_norm(p):
    p = torch.as_tensor(p, dtype=torch.float32)
    return (p - torch.tensor(XYZ_C, device=p.device)) / torch.tensor(XYZ_S, device=p.device)


def coord_features(x, vis, n_freq=6):
    """x (..., c) roughly in [-1, 1], vis (...) -> (..., c + 2*c*n_freq + 1)"""
    f = [x]
    for k in range(n_freq):
        f += [torch.sin((2 ** k) * math.pi * x), torch.cos((2 ** k) * math.pi * x)]
    return torch.cat(f + [vis.unsqueeze(-1)], -1)


def xyz_tokens(xyz, vis, n_freq=6):
    """3D positions (..., 3) [m] + visibility (...) -> token features (..., 3 + 6*n_freq + 1)"""
    vis = torch.as_tensor(vis, dtype=torch.float32)
    x = xyz_norm(xyz)
    return coord_features(x, vis.to(x.device), n_freq)


def project_uv(p, cam_pos, cam_xmat, K):
    """world points p (..., 3) -> normalized image coordinates (u/W, v/H) (..., 2), MuJoCo camera
    (looks along -z, y up); cam_pos (..., 3), cam_xmat (..., 9), K (..., 3, 3) with cx = W/2, cy = H/2"""
    Rm = cam_xmat.reshape(*cam_xmat.shape[:-1], 3, 3)
    pc = torch.einsum("...ji,...j->...i", Rm, p - cam_pos)
    depth = (-pc[..., 2]).clamp(min=1e-3)
    u = K[..., 0, 2] + K[..., 0, 0] * pc[..., 0] / depth
    v = K[..., 1, 2] - K[..., 1, 1] * pc[..., 1] / depth
    return torch.stack([u / (2 * K[..., 0, 2]), v / (2 * K[..., 1, 2])], -1)


def tcp_tokens(fk, q, cam_pos, cam_xmat, K, n_freq=6, uv_offset=None, coord="uv"):
    """arm joints q (..., 7) -> TCP token features (..., kp_dim): the TCP of the shared FK, projected.
    uv_offset (..., 2): optional calibration-error augmentation (normalized units)
    coord = "xyz": the TCP position itself (robot base frame), uv_offset = (..., 3) offset in metres"""
    shp = q.shape[:-1]
    tcp = fk.forward(q.reshape(-1, q.shape[-1]))[:, :3].reshape(*shp, 3)
    if coord == "xyz":
        if uv_offset is not None:
            tcp = tcp + uv_offset
        return xyz_tokens(tcp, torch.ones(shp, device=tcp.device, dtype=tcp.dtype), n_freq)
    uv = project_uv(tcp, cam_pos, cam_xmat, K)
    if uv_offset is not None:
        uv = uv + uv_offset
    return kp_features(uv, torch.ones(shp, device=uv.device, dtype=uv.dtype), n_freq)


def kp_features(uv, vis, n_freq=6):
    """uv (..., 2) in [0, 1], vis (...) -> (..., 2 + 4*n_freq + 1): [2uv-1, sin/cos(2^k pi uv), vis]"""
    return coord_features(uv * 2 - 1, vis, n_freq)


def kp_dim(n_freq, cdim=2):
    return cdim + 2 * cdim * n_freq + 1


def object_tokens(uv, vis, z, n_freq=6, use_z=True):
    """per-object token input: [keypoint features, z]  (numpy or torch in, torch out)"""
    uv, vis, z = (torch.as_tensor(t, dtype=torch.float32) for t in (uv, vis, z))
    k = kp_features(uv, vis, n_freq)
    return torch.cat([k, z], -1) if use_z else k


class ObjEmbed(nn.Module):
    def __init__(self, kp, zd, d=D_MODEL):
        super().__init__()
        self.kp, self.zd = kp, zd
        self.kp_mlp = nn.Sequential(nn.Linear(kp, d), nn.GELU(), nn.Linear(d, d))
        self.z_mlp = (nn.Sequential(nn.LayerNorm(zd), nn.Linear(zd, d), nn.GELU(), nn.Linear(d, d))
                      if zd > 0 else None)

    def forward(self, x):
        e = self.kp_mlp(x[..., :self.kp])
        if self.z_mlp is not None:
            e = e + self.z_mlp(x[..., self.kp:])
        return e


class VisionHistoryEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, t_history=T_HISTORY,
                 n_types=2, dropout=0.1, causal=False):
        super().__init__()
        self.t = t_history
        self.time_embed = nn.Parameter(torch.randn(1, t_history, d_model) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(n_types, d_model) * 0.02)
        self.blocks = nn.ModuleList([TrunkBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.causal = causal

    def mask(self, n_types, device, dtype):
        if not self.causal:
            return torch.zeros(self.t * n_types, self.t * n_types, device=device, dtype=dtype)
        t = torch.arange(self.t, device=device).repeat(n_types)
        m = torch.zeros(len(t), len(t), device=device, dtype=dtype)
        return m.masked_fill(t[None, :] > t[:, None], float("-inf"))

    def forward(self, streams):
        x = torch.cat([s + self.time_embed + self.type_embed[k] for k, s in enumerate(streams)], 1)
        mask = self.mask(len(streams), x.device, x.dtype)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)


def time_embedding(t, dim):
    """t (B,) in [0, 1] -> (B, dim) sinusoidal"""
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000.0)[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class DecisionCoreVision(nn.Module):
    """head = "mse" : the K queries regress the chunk directly (z_hat = z_head(h))
       head = "flow": flow matching on the NORMALIZED latent chunk x (K, 128):
                      x_t = (1-t) x0 + t x1, x0 ~ N(0, I), x1 = (z_true - z_mu) / z_sd; the queries get
                      W_in x_t + time embedding(t) and predict the velocity x1 - x0. Sampling integrates
                      dx/dt = v from noise in `flow_steps` Euler steps. The gripper head always reads a
                      noise-free pass of the same queries (it does not depend on the sample)."""

    def __init__(self, z_dim=64, n_freq=6, use_z=True, t_history=T_HISTORY, n_layers=N_LAYERS, n_heads=N_HEADS,
                 dropout=0.1, chunk=10, dec_layers=2, causal=False, residual=False, use_tcp=False,
                 d_model=D_MODEL, head="mse", flow_steps=10, coord="uv", anchor=False):
        super().__init__()
        self.coord = coord
        self.cdim = 3 if coord == "xyz" else 2
        self.register_buffer("rel_gain", torch.tensor(XYZ_S) / REL_M if coord == "xyz"
                             else torch.full((2,), REL_SCALE), persistent=False)   # old checkpoints load unchanged
        D = self.d = d_model           # width of the core; the arm latent stays 128-d (projected in / out)
        self.use_tcp = use_tcp
        self.residual = residual      # z_hat_k = z_now + delta_k: the error shrinks with the remaining motion
        self.head, self.flow_steps = head, flow_steps
        # anchor head (coord xyz): per chunk step, the TCP position RELATIVE to the observed cube (grasp) and to the
        # observed place point (placement), in units of REL_M. Used for the final approach: target = anchor + offset
        # is absolute, so it does not drift like a chain of residual steps, and its precision is that of the
        # triangulated anchor. Output (B, K, 6) = [offset to cube (3), offset to tape (3)] in self.last_anc
        self.anchor = anchor and coord == "xyz"
        self.last_anc = None
        if head == "flow" and residual:
            raise ValueError("flow head: residual output is not supported")
        self.kp = kp_dim(n_freq, self.cdim)
        self.z_dim = z_dim if use_z else 0
        self.n_freq, self.use_z, self.chunk = n_freq, use_z, chunk
        self.tok_dim = self.kp + self.z_dim
        self.encoder = VisionHistoryEncoder(D, n_layers, n_heads, t_history, 3 if use_tcp else 2, dropout, causal)
        if use_tcp:
            self.tcp_mlp = nn.Sequential(nn.Linear(self.kp + 2 * self.cdim, D), nn.GELU(), nn.Linear(D, D))
        self.grip_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, D))
        self.obj = ObjEmbed(self.kp, self.z_dim, D)
        self.goal_dim = (3 if use_tcp else 2) * D + 1
        self.query_proj = nn.Sequential(nn.Linear(self.goal_dim, D), nn.GELU(), nn.Linear(D, D))
        self.chunk_pos = nn.Parameter(torch.randn(1, chunk, D) * 0.02)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(D, n_heads, D * FFN_MULT, dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(dec_layers)])
        self.out_ln = nn.LayerNorm(D)
        self.z_head = nn.Linear(D, LATENT)
        self.arm_in = nn.Identity() if D == LATENT else nn.Linear(LATENT, D)
        self.grip_head = nn.Sequential(nn.Linear(D + self.goal_dim, D), nn.GELU(), nn.Linear(D, 1))
        if self.anchor:
            self.anc_head = nn.Sequential(nn.Linear(D + self.goal_dim, D), nn.GELU(), nn.Linear(D, 6))
        if head == "flow":
            self.x_in = nn.Linear(LATENT, D)
            self.t_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
            self.register_buffer("z_mu", torch.zeros(LATENT))
            self.register_buffer("z_sd", torch.ones(LATENT))

    # ------------------------------------------------------------------ pieces
    def encode(self, z_hist, finger_hist, cube_hist, tape_goal, tcp_hist=None):
        arm = self.arm_in(z_hist) + self.grip_embed((finger_hist * FINGER_SCALE).unsqueeze(-1))
        cube = self.obj(cube_hist)
        streams, extra = [arm, cube], []
        if self.use_tcp:
            c = self.cdim
            rel = torch.cat([(cube_hist[..., :c] - tcp_hist[..., :c]) * self.rel_gain,
                             (tape_goal[:, None, :c].expand_as(tcp_hist[..., :c]) - tcp_hist[..., :c]) * self.rel_gain], -1)
            tcp = self.tcp_mlp(torch.cat([tcp_hist, rel], -1))
            streams.append(tcp); extra = [tcp[:, -1]]
        mem = self.encoder(streams)
        goal = torch.cat([self.obj(tape_goal), cube[:, -1]] + extra +
                         [(finger_hist[:, -1] * FINGER_SCALE).unsqueeze(-1)], -1)
        return mem, goal

    def decode(self, q, mem):
        h = q
        for layer in self.decoder:
            h = layer(h, mem)
        return self.out_ln(h)

    def grip(self, h, goal):
        g = goal.unsqueeze(1).expand(-1, self.chunk, -1)
        return self.grip_head(torch.cat([h, g], -1)).squeeze(-1)

    def velocity(self, base, mem, x, t):
        """base (B,K,D) noise-free queries, x (B,K,128) normalized sample, t (B,) -> v (B,K,128)"""
        te = self.t_mlp(time_embedding(t, self.d))
        return self.z_head(self.decode(base + self.x_in(x) + te[:, None], mem))

    def normalize(self, z):
        return (z - self.z_mu) / self.z_sd

    def denormalize(self, x):
        return x * self.z_sd + self.z_mu

    # ------------------------------------------------------------------ main entry
    def forward(self, z_hist, finger_hist, cube_hist, tape_goal, tcp_hist=None, x_t=None, t=None, noise=None,
                steps=None):
        """z_hist (B,T,128) arm latents, finger_hist (B,T) metres, cube_hist (B,T,tok_dim) cube tokens
        (oldest -> newest), tape_goal (B,tok_dim) -> z_hat (B,K,128), gripper logits (B,K).
        flow head: with x_t / t given -> (velocity, gripper logits) for training; otherwise a SAMPLE
        (noise: initial x, default N(0, I); steps: Euler steps, default flow_steps)."""
        mem, goal = self.encode(z_hist, finger_hist, cube_hist, tape_goal, tcp_hist)
        base = self.query_proj(goal).unsqueeze(1) + self.chunk_pos
        h0 = self.decode(base, mem)
        logit = self.grip(h0, goal)
        if self.anchor:
            g = goal.unsqueeze(1).expand(-1, self.chunk, -1)
            self.last_anc = self.anc_head(torch.cat([h0, g], -1))
        if self.head != "flow":
            z = self.z_head(h0)
            if self.residual:
                z = z + z_hist[:, -1:]
            return z, logit
        if x_t is not None:
            return self.velocity(base, mem, x_t, t), logit
        B = z_hist.shape[0]
        x = torch.randn(B, self.chunk, LATENT, device=z_hist.device) if noise is None else noise
        n = steps or self.flow_steps
        for i in range(n):
            tt = torch.full((B,), i / n, device=z_hist.device)
            x = x + self.velocity(base, mem, x, tt) / n
        return self.denormalize(x), logit

    def init_from_chunk(self, path, device="cpu"):
        sd = torch.load(path, map_location=device, weights_only=False)["state_dict"]
        mine = self.state_dict()
        rename = {"encoder.pos_embed": "encoder.time_embed"}
        copied = []
        for k, v in sd.items():
            k2 = rename.get(k, k)
            if k2 in mine and mine[k2].shape == v.shape:
                mine[k2] = v
                copied.append(k2)
        self.load_state_dict(mine)
        return copied


def load_vision_core(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = DecisionCoreVision(c["z_dim"], c["n_freq"], c["use_z"], c["t_history"], c["n_layers"], c["n_heads"],
                           chunk=c["chunk"], dec_layers=c["dec_layers"], causal=c["causal"],
                           residual=c.get("residual", False), use_tcp=c.get("use_tcp", False),
                           d_model=c.get("d_model", D_MODEL), head=c.get("head", "mse"),
                           flow_steps=c.get("flow_steps", 10), coord=c.get("coord", "uv"),
                           anchor=c.get("anchor", False)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, c