"""
Decision core trunk.

Architecture v2 -- revised after the multi-goal ablation showed the FiLM +
readout-token version never learned to use goal (zeroing goal out entirely
gave the same loss). Informed by reading HPT's actual trunk (arxiv
2409.20537): HPT itself has NO goal-FiLM and NO readout token -- it just
pools the full output sequence and never had to solve "make an explicit
runtime goal signal impossible to ignore" (their task identity is implicit
in which stem/head pair is active, not a continuous 3D input). So this is
NOT a copy of HPT; it borrows HPT's "pool the trunk output" idea and
combines it with a mechanism HPT never needed: making goal the QUERY for
that pooling step, so goal can't be architecturally bypassed the way FiLM
could.

  - Input: T=8 proprio latent tokens (frozen Stage-1 embedding output,
    128d) + learned positional encoding. NO readout token in the sequence.
  - 10x pre-LN transformer blocks (up from 6 -- HPT-Small uses 16, but our
    dataset is far smaller, so this is a middle ground, not a copy),
    4 heads, d_model=128, FFN 4x expansion, plain residuals throughout --
    NO FiLM anywhere in these blocks.
  - Causal mask only (T x T): proprio_i attends to proprio_0..proprio_i.
  - Goal input is 6d: [goal_xyz, goal_xyz - current_ee_pos] (unchanged from
    before -- this part wasn't the problem).
  - Final step: GoalQueryPooling. goal_input -> small MLP -> ONE query
    vector. Cross-attention: that query attends over the trunk's full
    T-token output (keys/values). The result IS z_hat. There is no path
    to z_hat that doesn't go through this goal-conditioned attention --
    goal cannot be architecturally ignored the way FiLM's soft (1+gamma)
    modulation could be (and, empirically, was).
  - z_hat (128d) feeds the FROZEN Stage-1 decoder together with q_ref to
    produce q_pred (7d, absolute next joint angles).
  - Trunk + GoalQueryPooling are the only trainable parts. Stage-1
    FKEncoder/Embedding/Decoder are all loaded frozen (load_frozen_stage1).
  - q_ref noise during training (Stage-1 eps=0.03 regime) is unchanged
    from before -- disabled automatically at eval time via model.eval().

Decoder input (delta input): a retrained FR3 decoder (scripts/train_fr3_decoder.py,
checkpoint stage1_franka_dec2.pt) does not take z_hat itself but
    delta_scale * (z_hat - W_FR3(FK(q_ref)))
i.e. "how far to move" instead of "where to go". Every place that calls the
decoder builds its input with decoder_input(...) below; for the original
stage1_franka.pt (no delta_scale) decoder_input returns z_hat unchanged.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 128
N_LAYERS = 10          # up from 6; HPT-Small uses 16 but that's tuned for a
                        # much larger dataset than ours
N_HEADS = 4
FFN_MULT = 4
T_HISTORY = 8
GOAL_DIM = 6           # [goal_xyz(3), goal_xyz - current_ee_pos(3)] -- unchanged
ARM_DOF = 7
QREF_NOISE_STD = 0.03  # matches Stage-1's own q_ref=q+eps training regime
DEFAULT_STAGE1_CKPT = "checkpoints/stage1_franka.pt"


class TrunkBlock(nn.Module):
    """One plain pre-LN transformer block: self-attn + FFN, both plain
    residuals. No FiLM anywhere -- goal conditioning happens ONLY at the
    final GoalQueryPooling step, never inside these blocks."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 ffn_mult: int = FFN_MULT, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)

        h = self.ln2(x)
        x = x + self.ffn_dropout(self.ffn(h))
        return x


def build_causal_mask(t_history: int, device=None) -> torch.Tensor:
    """Standard causal mask over the T proprio tokens only -- no readout
    token in the sequence anymore, so this is just a plain T x T causal mask."""
    return torch.triu(
        torch.full((t_history, t_history), float("-inf"), device=device), diagonal=1)


class GoalQueryPooling(nn.Module):
    """
    goal (B, goal_dim) -> small MLP -> one query vector (B,1,d_model).
    Cross-attention: query attends over the trunk's T-token output
    (B,T,d_model) as keys/values -> pooled result (B,d_model) = z_hat.
    Goal is architecturally load-bearing here: there is no way to compute
    z_hat without it, unlike FiLM's easily-ignorable soft modulation.
    """

    def __init__(self, goal_dim: int = GOAL_DIM, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, hidden: int = 64):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(goal_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # NO dropout here (deliberately, unlike TrunkBlock): this attention's
        # output is compared directly against a different-goal version of
        # itself in train.py's contrastive loss. Dropout here would give the
        # model a free, goal-independent source of noise to satisfy that
        # margin with -- confirmed empirically: train-time dist looked good
        # (~4.2) but collapsed to ~0 the moment eval mode turned dropout off,
        # proving query_proj had learned nothing about goal itself.
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, goal: torch.Tensor, trunk_out: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(goal).unsqueeze(1)          # (B,1,d_model)
        pooled, _ = self.attn(query, trunk_out, trunk_out, need_weights=False)  # (B,1,d_model)
        return self.out_ln(pooled.squeeze(1))                # (B,d_model)


class DecisionCoreTrunk(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, t_history: int = T_HISTORY,
                 dropout: float = 0.1, goal_dim: int = GOAL_DIM):
        super().__init__()
        self.d_model = d_model
        self.t_history = t_history

        self.pos_embed = nn.Parameter(torch.randn(1, t_history, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            TrunkBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.pre_pool_ln = nn.LayerNorm(d_model)
        self.goal_pool = GoalQueryPooling(goal_dim, d_model, n_heads)

        self.register_buffer(
            "attn_mask", build_causal_mask(t_history), persistent=False)

    def forward(self, z_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        z_hist: (B, T, d_model) -- frozen encoder's proprio latents, oldest->newest
        goal:   (B, goal_dim)
        returns z_hat: (B, d_model)
        """
        return self.pool(self.encode_history(z_hist), goal)

    def encode_history(self, z_hist: torch.Tensor) -> torch.Tensor:
        """
        The goal-independent part: runs the causal self-attention blocks
        over z_hist only. Exposed separately (not just inlined in forward)
        so callers computing z_hat under two different goals for the SAME
        z_hist (e.g. train.py's contrastive loss) can share ONE pass through
        these blocks -- calling this twice under dropout would inject two
        independent noise draws, making any resulting z_hat difference
        partly an artifact of dropout rather than purely the goal's effect.
        """
        x = z_hist + self.pos_embed
        mask = self.attn_mask.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return self.pre_pool_ln(x)

    def pool(self, trunk_out: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """The goal-dependent part: one cross-attention pooling step."""
        return self.goal_pool(goal, trunk_out)


# ============================== Frozen Stage-1 wrapper ======================

def load_frozen_stage1(device: str = "cpu", ckpt_path: str = None):
    """
    Loads FKEncoder (analytic, no weights) + MLPEmbedding + Decoder, all
    frozen. Returns (fk, embedding, decoder, m_const) -- m_const is the robot's
    fixed morphology vector, needed by the decoder's forward signature.

    ckpt_path: default checkpoints/stage1_franka.pt (original Stage-1 decoder).
    checkpoints/stage1_franka_dec2.pt = SAME embedding W + retrained delta-input
    decoder; its scale is stored on the decoder as `decoder.delta_scale`
    (None for the original), which decoder_input() reads.
    """
    import sys
    PROJECT_ROOT = os.path.expanduser("~/cross_embody")
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from fk_encoder.fk_encoder import FKEncoder
    from layer2.embedding import MLPEmbedding
    from layer2.decoder import Decoder

    ROBOTS_YAML = os.path.join(PROJECT_ROOT, "configs/robots.yaml")
    CKPT_PATH = ckpt_path or DEFAULT_STAGE1_CKPT
    if not os.path.isabs(CKPT_PATH):
        CKPT_PATH = os.path.join(PROJECT_ROOT, CKPT_PATH)
    EMB_IN = 18

    fk = FKEncoder("fr3", ROBOTS_YAML, project_root=PROJECT_ROOT, device=device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = ckpt.get("config", {})

    embedding = MLPEmbedding(EMB_IN, ckpt["latent_dim"]).to(device)
    embedding.load_state_dict(ckpt["W"])
    embedding.eval()

    decoder = Decoder(ckpt["latent_dim"], m_dim=3, out_dim=ARM_DOF,
                       hidden=cfg.get("dec_hidden", 256), depth=cfg.get("dec_depth", 2),
                       use_q_prev=True).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    decoder.delta_scale = cfg.get("delta_scale")      # None for the original decoder

    for p in list(embedding.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    print(f"[stage1] {CKPT_PATH}  decoder={cfg.get('dec_hidden', 256)}x{cfg.get('dec_depth', 2)}  "
          f"delta_input={decoder.delta_scale is not None}")
    m_const = fk.m.to(device).unsqueeze(0)  # (1,3), broadcast per-batch at call time
    return fk, embedding, decoder, m_const


def decoder_input(z_hat: torch.Tensor, q_ref: torch.Tensor, fk, embedding, decoder) -> torch.Tensor:
    """What to feed the FR3 decoder for target latent z_hat at current joints q_ref.
    Original decoder: z_hat. Delta-input decoder: delta_scale * (z_hat - W(FK(q_ref)))."""
    scale = getattr(decoder, "delta_scale", None)
    if scale is None:
        return z_hat
    with torch.no_grad():
        z_ref = embedding(fk.forward(q_ref)[:, :18])
    return (z_hat - z_ref) * scale


@torch.no_grad()
def inverse_decode(z_hat: torch.Tensor, q_ref: torch.Tensor, fk, embedding,
                   iters: int = 8, damping: float = 1e-4, max_step: float = 0.3,
                   eps: float = 1e-4, lo=None, hi=None) -> torch.Tensor:
    """Decoder defined as the INVERSE OF THE ENCODER (no learned weights):
           q* = argmin_q || W(FK(q)) - z_hat ||²,  started at q_ref
    solved with damped Gauss-Newton; Jacobian of W∘FK by finite differences.
    Works for any robot given its FK + encoder (FR3: W_FR3; new arm: W_new).
    Starting at q_ref keeps the solution in q_ref's IK branch, and small requested
    steps are executed exactly -- there is no learned dead zone."""
    q = q_ref.clone()
    B, d = q.shape
    eye = torch.eye(d, device=q.device)
    for _ in range(iters):
        qq = q.unsqueeze(1).repeat(1, d + 1, 1)
        qq[:, 1:] += eps * eye
        z = embedding(fk.forward(qq.reshape(-1, d))[:, :18]).view(B, d + 1, -1)
        r = z[:, 0] - z_hat                                    # (B,128)
        J = ((z[:, 1:] - z[:, :1]) / eps).transpose(1, 2)      # (B,128,d)
        JT = J.transpose(1, 2)
        dq = -torch.linalg.solve(JT @ J + damping * eye, (JT @ r.unsqueeze(-1))).squeeze(-1)
        n = dq.norm(dim=1, keepdim=True).clamp(min=1e-12)
        q = q + dq * torch.clamp(max_step / n, max=1.0)
        if lo is not None:
            q = torch.max(torch.min(q, hi), lo)
    return q


@torch.no_grad()
def latent_to_meaning(z_hat: torch.Tensor, f0: torch.Tensor, W, S, L,
                      iters: int = 10, damping: float = 1e-4, eps: float = 1e-4,
                      solve_kv: bool = True):
    """Dictionary lookup: which FR3-style description [TCP position p, TCP rotation r6,
    elbow/wrist key-vectors v_SE, v_SW] does z_hat say?  Solves
        u* = argmin_u || W_FR3([p, r6, v_SE, v_SW, (p - S)/L]) - z_hat ||²
    (v_SEE follows from p exactly), started at f0 = the arm's current description.
    Returns (p*, r6*, kv*)."""
    kv0 = f0[:, 9:15].clone()
    u = f0[:, :15].clone() if solve_kv else f0[:, :9].clone()
    B, d = u.shape
    eye = torch.eye(d, device=u.device)

    def feats(x):
        if not solve_kv:                     # arm shape held at the current description:
            x = torch.cat([x, kv0.unsqueeze(1).expand(-1, x.shape[0] // B, -1).reshape(-1, 6)
                           if x.shape[0] != B else kv0], 1)      # every latent change -> TCP
        return torch.cat([x, (x[:, :3] - S) / L], 1)

    for _ in range(iters):
        uu = u.unsqueeze(1).repeat(1, d + 1, 1)
        uu[:, 1:] += eps * eye
        z = W(feats(uu.reshape(-1, d))).view(B, d + 1, -1)
        r = z[:, 0] - z_hat
        J = ((z[:, 1:] - z[:, :1]) / eps).transpose(1, 2)
        JT = J.transpose(1, 2)
        u = u - torch.linalg.solve(JT @ J + damping * eye, JT @ r.unsqueeze(-1)).squeeze(-1)
    return u[:, :3], u[:, 3:9], (u[:, 9:15] if solve_kv else kv0)


@torch.no_grad()
def tcp_ik(fk, q_ref: torch.Tensor, p_t: torch.Tensor, r6_t: torch.Tensor, iters: int = 10,
           w_rot: float = 0.3, damping: float = 1e-4, eps: float = 1e-4, max_step: float = 0.3):
    """The arm's OWN realization of a TCP pose: damped Gauss-Newton IK from q_ref
    (stays in q_ref's branch), target given as position + rot6d."""
    q = q_ref.clone()
    B, d = q.shape
    eye = torch.eye(d, device=q.device)
    tgt = torch.cat([p_t, w_rot * r6_t], 1)
    for _ in range(iters):
        qq = q.unsqueeze(1).repeat(1, d + 1, 1)
        qq[:, 1:] += eps * eye
        f = fk.forward(qq.reshape(-1, d)).view(B, d + 1, -1)
        f = torch.cat([f[..., :3], w_rot * f[..., 3:9]], -1)
        r = f[:, 0] - tgt
        J = ((f[:, 1:] - f[:, :1]) / eps).transpose(1, 2)
        JT = J.transpose(1, 2)
        dq = -torch.linalg.solve(JT @ J + damping * eye, JT @ r.unsqueeze(-1)).squeeze(-1)
        n = dq.norm(dim=1, keepdim=True).clamp(min=1e-12)
        q = q + dq * torch.clamp(max_step / n, max=1.0)
    return q


def add_qref_noise(q_ref: torch.Tensor, std: float) -> torch.Tensor:
    """Gaussian noise on q_ref, matching Stage 1's own q_ref=q+eps regime."""
    if std <= 0:
        return q_ref
    return q_ref + torch.randn_like(q_ref) * std


def compute_goal_input(fk, goal: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """
    goal (B,3), q_ref (B,7) -> (B,6) = [goal_xyz, goal_xyz - current_ee_pos].
    Shared by DecisionCoreModel.forward and rollout_eval.py's manual loop --
    single source of truth so train and rollout can never silently diverge
    on how this signal is built.
    """
    ee_pos = fk.forward(q_ref)[:, :3]
    return torch.cat([goal, goal - ee_pos], dim=-1)


class DecisionCoreModel(nn.Module):
    """
    Full model used at train/eval time: trunk (trainable) + frozen
    Stage-1 FKEncoder + decoder (+ embedding, needed only to build the
    delta-input decoder's input).
    """

    def __init__(self, fk, decoder: nn.Module, m_const: torch.Tensor,
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, t_history: int = T_HISTORY,
                 dropout: float = 0.1, qref_noise_std: float = QREF_NOISE_STD,
                 embedding: nn.Module = None):
        super().__init__()
        self.trunk = DecisionCoreTrunk(d_model, n_layers, n_heads, t_history, dropout)
        self.fk = fk                    # frozen, analytic, no parameters
        self.decoder = decoder          # frozen, eval mode, no grad
        self.embedding = embedding      # frozen W_FR3 (only used for delta input)
        self.qref_noise_std = qref_noise_std
        self.register_buffer("m_const", m_const, persistent=False)

    def forward(self, z_hist: torch.Tensor, goal: torch.Tensor,
                q_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        goal here is the raw 3d xyz -- expanded to 6d internally.
        Returns (q_pred, z_hat):
          q_pred: (B, ARM_DOF) -- absolute predicted next joint angles
          z_hat:  (B, D_MODEL) -- trunk's own latent prediction, for the
                                   auxiliary z_hat-vs-z_true loss
        """
        if self.training:
            q_ref = add_qref_noise(q_ref, self.qref_noise_std)

        goal_input = compute_goal_input(self.fk, goal, q_ref)  # (B,6)
        z_hat = self.trunk(z_hist, goal_input)
        B = z_hat.shape[0]
        m_batch = self.m_const.expand(B, -1)
        # decoder's OWN parameters are frozen, but the forward pass must stay
        # differentiable so the loss on q_pred can backprop through z_hat into
        # the trunk. Do NOT wrap this call in torch.no_grad().
        z_in = decoder_input(z_hat, q_ref, self.fk, self.embedding, self.decoder)
        q_pred = self.decoder(z_in, m_batch, q_prev=q_ref)
        return q_pred, z_hat


# ============================== TCP baseline ==============================
TCP_DIM = 9   # [p (3), rot6d (6)] = fk21[:9]


class TCPDecisionCore(nn.Module):
    """Baseline for the 'does the FR3 latent add anything over the TCP?' question.
    Same trunk (DecisionCoreTrunk, same size, same goal-query pooling), but the input
    history is the TCP pose [p, r6] of each step (normalized) instead of the FR3 latent,
    and the output is the next TCP change (normalized). Any arm executes it with its own
    IK (decision_core.model.tcp_ik); no embedding, no dictionary, no latent decoder."""

    def __init__(self, t_history: int = T_HISTORY, n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(TCP_DIM, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.trunk = DecisionCoreTrunk(D_MODEL, n_layers, n_heads, t_history, dropout)
        self.out = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, TCP_DIM))
        for name in ("f_mu", "f_sd", "d_sd"):
            self.register_buffer(name, torch.ones(TCP_DIM) if name != "f_mu" else torch.zeros(TCP_DIM))

    def forward(self, f_hist_norm: torch.Tensor, goal_input: torch.Tensor) -> torch.Tensor:
        """f_hist_norm (B,T,9) normalized TCP history -> normalized TCP change (B,9)"""
        return self.out(self.trunk(self.inp(f_hist_norm), goal_input))

    def norm(self, f):
        return (f - self.f_mu) / self.f_sd

    @torch.no_grad()
    def next_tcp(self, f_hist_raw: torch.Tensor, goal_input: torch.Tensor) -> torch.Tensor:
        """raw TCP history (B,T,9) -> raw target TCP (B,9) = current + predicted change"""
        return f_hist_raw[:, -1] + self(self.norm(f_hist_raw), goal_input) * self.d_sd


def load_tcp_core(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)   # our own checkpoint
    c = ck["config"]
    m = TCPDecisionCore(c["t_history"], c.get("n_layers", N_LAYERS), c.get("n_heads", N_HEADS)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


# ============================== pick-and-place ==============================
GOAL_DIM_PP = 13          # [cube, cube - ee, place, place - ee, finger opening]
FINGER_SCALE = 25.0       # Franka finger joint 0..0.04 m -> 0..1


def compute_goal_input_pp(fk, cube, place, q_ref, finger, psi_goal=None, psi_now=None):
    """cube (B,3), place (B,3), q_ref (B,7), finger (B,) -> (B,13);
    with psi_goal (B,2) = target swivel angles (pick, place) -> (B,17): + sin/cos of each;
    with psi_now (B,) too -> (B,19): + the posture ERRORS wrap(psi_goal - psi_now), exactly like the
    position errors cube - ee / place - ee. psi_now is measured by each arm on its own key points."""
    ee = fk.forward(q_ref)[:, :3]
    g = torch.cat([cube, cube - ee, place, place - ee, (finger * FINGER_SCALE).unsqueeze(-1)], -1)
    if psi_goal is not None:
        g = torch.cat([g, torch.sin(psi_goal), torch.cos(psi_goal)], -1)
        if psi_now is not None:
            d = psi_goal - psi_now.unsqueeze(-1)
            g = torch.cat([g, torch.atan2(torch.sin(d), torch.cos(d))], -1)
    return g


class DecisionCorePP(nn.Module):
    """Decision core for pick-and-place: the same trunk, plus
      - the gripper opening of every history step added to its latent token,
      - goal input = cube and place point (and their offsets from the TCP) + gripper opening,
      - a gripper head: logit of 'gripper closed' for the current step.
    The arm output stays a latent z_hat, decoded by each arm as before."""

    def __init__(self, t_history: int = T_HISTORY, n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 dropout: float = 0.1, goal_dim: int = GOAL_DIM_PP):
        super().__init__()
        self.goal_dim = goal_dim                        # 13, or 17 with posture goals (sin/cos psi1, psi2)
        self.trunk = DecisionCoreTrunk(D_MODEL, n_layers, n_heads, t_history, dropout, goal_dim=goal_dim)
        self.grip_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, D_MODEL))
        self.grip_head = nn.Sequential(nn.Linear(D_MODEL + goal_dim, D_MODEL), nn.GELU(),
                                       nn.Linear(D_MODEL, 1))

    def forward(self, z_hist, finger_hist, goal_input):
        """z_hist (B,T,128), finger_hist (B,T) metres, goal_input (B,13) -> z_hat (B,128), grip logit (B,)"""
        x = z_hist + self.grip_embed((finger_hist * FINGER_SCALE).unsqueeze(-1))
        z_hat = self.trunk(x, goal_input)
        return z_hat, self.grip_head(torch.cat([z_hat, goal_input], -1)).squeeze(-1)


def load_pp_core(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)   # our own checkpoint
    c = ck["config"]
    m = DecisionCorePP(c["t_history"], c.get("n_layers", N_LAYERS), c.get("n_heads", N_HEADS),
                       goal_dim=c.get("goal_dim", GOAL_DIM_PP)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, c