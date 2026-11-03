"""
Retrain the FR3 DECODER with the frozen Stage-1 embedding W_FR3.

Why: the decision core now outputs on-manifold latents (z_hat = W_FR3(FK(next pose))).
The Stage-1 decoder only executes ~20% of a requested step and has a ~5 mm floor on
small steps (diag_correction: remain ≈ 0.8), so the latent decision core moves FR3 far
too slowly (MuJoCo min-DTG SR 25%). The old decision core hid this by over-asking
(the FR3-only "dialect"). This decoder is trained like the UR5e one:

  input   : scale * (z_target - W_FR3(FK(q_ref)))   (DeltaInput -- small requests visible)
  data    : uniform FR3 poses over the whole above-table workspace (q_ref = velocity-
            limited steps or q + noise) + real expert (q_t, q_t+1) pairs
  loss    : TCP pose (mm²) + joint term + relative-correction loss, joint-only warm-up
  stopping: early stopping, patience 30

W_FR3 is NOT modified: the latent space, cached latents, the UR5e dictionary and the
decision core's inputs/targets all stay valid. Output:
  checkpoints/stage1_franka_dec2.pt  (same "W" + new "decoder" + config.delta_scale)
Use it everywhere with:  export FR3_STAGE1_CKPT=checkpoints/stage1_franka_dec2.pt

Run:  python scripts/train_fr3_decoder.py
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import torch

R = os.path.expanduser("~/cross_embody")
for p in (R, os.path.join(R, "scripts")):
    sys.path.insert(0, p)
import train_stage2_align as T
from fk_encoder.fk_encoder import FKEncoder
from layer2.embedding import MLPEmbedding
from layer2.decoder import Decoder

DEV = T.DEVICE
OUT = os.path.join(R, "checkpoints", "stage1_franka_dec2.pt")
EXPERT_DIR = os.path.join(R, "fr3_pick_gen", "datasets", "expert_v1", "data", "chunk-000")
FR3_VEL = torch.tensor([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], device=DEV)
CONTROL_HZ, STEP_RATIO, Q_NOISE = 30.0, 0.5, 0.03
N_UNIFORM, BATCH, REAL_FRAC = 200000, 512, 0.3
HIDDEN, DEPTH = 512, 3
LAMBDA1, ALPHA_J, REL = 0.01, 1000.0, 1.0
WARMUP, RAMP, PATIENCE, MAX_EPOCHS = 10, 10, 30, 1000
torch.manual_seed(0)


def sample_fr3(enc, n):
    lim = torch.tensor(T.FR3_JOINT_LIMITS, device=DEV)
    q = lim[:, 0] + torch.rand(n * 3, 7, device=DEV) * (lim[:, 1] - lim[:, 0])
    with torch.no_grad():
        keep = enc.forward(q)[:, 2] > 0.0
    return q[keep][:n]


def synth_qref(q):
    n = len(q); h = n // 2
    u = torch.rand(h, 1, device=DEV)
    step = (torch.rand(h, 7, device=DEV) * 2 - 1) * (STEP_RATIO * FR3_VEL / CONTROL_HZ) * u
    return torch.cat([q[:h] - step, q[h:] + torch.randn_like(q[h:]) * Q_NOISE], 0)


def load_expert_pairs():
    qr, qt = [], []
    for f in sorted(glob.glob(os.path.join(EXPERT_DIR, "episode_*.parquet"))):
        s = np.stack(pd.read_parquet(f)["observation.state"].values)[:, :7]
        qr.append(s[:-1]); qt.append(s[1:])
    return (torch.tensor(np.concatenate(qr), dtype=torch.float32, device=DEV),
            torch.tensor(np.concatenate(qt), dtype=torch.float32, device=DEV))


def main():
    enc = FKEncoder("fr3", T.ROBOTS_YAML, project_root=R, device=DEV)
    T.check_rot_slice(enc, T.FR3_JOINT_LIMITS)
    ck = torch.load(T.SRC_CKPT, map_location=DEV)
    W = MLPEmbedding(T.EMB_IN, T.LATENT_DIM).to(DEV); W.load_state_dict(ck["W"]); W.eval()
    for p in W.parameters():
        p.requires_grad_(False)
    m = enc.m.to(DEV).unsqueeze(0)

    Qu = sample_fr3(enc, N_UNIFORM)
    Er, Et = load_expert_pairs()
    pu, pe = torch.randperm(len(Qu), device=DEV), torch.randperm(len(Er), device=DEV)
    vu, tu = pu[:10000], pu[10000:]
    ve, te = pe[:3000], pe[3000:]
    print(f"[data] uniform {len(tu)} train / {len(vu)} val   expert pairs {len(te)} / {len(ve)}")

    with torch.no_grad():                         # delta scale from typical training requests
        q = Qu[tu[:8192]]; qr = synth_qref(q)
        dz = W(enc.forward(q)[:, :T.EMB_IN]) - W(enc.forward(qr)[:, :T.EMB_IN])
        scale = 1.0 / (dz.std(0).mean().item() + 1e-8)
    dec = Decoder(T.LATENT_DIM, m_dim=3, out_dim=7, hidden=HIDDEN, depth=DEPTH, use_q_prev=True).to(DEV)
    net = T.DeltaInput(dec, W, enc, scale)
    print(f"[decoder] delta input scale={scale:.1f}  decoder={HIDDEN}x{DEPTH}")
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)

    def batch():
        nr = int(BATCH * REAL_FRAC)
        iu = tu[torch.randint(0, len(tu), (BATCH - nr,), device=DEV)]
        ie = te[torch.randint(0, len(te), (nr,), device=DEV)]
        q = torch.cat([Qu[iu], Et[ie]]); qr = torch.cat([synth_qref(Qu[iu]), Er[ie]])
        return q, qr

    @torch.no_grad()
    def evaluate():
        dec.eval()
        out = {}
        for name, q, qr in [("uniform", Qu[vu], synth_qref(Qu[vu])), ("expert", Et[ve], Er[ve])]:
            z = W(enc.forward(q)[:, :T.EMB_IN]); qh = net(z, m.expand(len(q), -1), qr)
            out[name] = ((enc.forward(qh)[:, :3] - enc.forward(q)[:, :3]).norm(dim=1) * 1000)
        # small-step test: request a 0.01 rad move, how much is executed?
        q = Qu[vu[:4000]]; u = torch.randn_like(q); u = u / u.norm(dim=1, keepdim=True)
        qr = q + 0.01 * u
        qh = net(W(enc.forward(q)[:, :T.EMB_IN]), m.expand(len(q), -1), qr)
        remain = (qh - q).norm(dim=1) / 0.01
        cos = torch.nn.functional.cosine_similarity(qh - qr, q - qr, dim=1)
        dec.train()
        return out, remain.median().item(), cos.mean().item()

    best, since, hist = float("inf"), 0, []
    steps = len(tu) // BATCH
    for ep in range(MAX_EPOCHS):
        ee_w = 0.0 if ep < WARMUP else min(1.0, (ep - WARMUP + 1) / RAMP)
        tot = 0.0
        for _ in range(steps):
            q, qr = batch()
            p_t, R_t = T.pose(enc.forward(q))
            z = W(enc.forward(q)[:, :T.EMB_IN])
            qh = net(z, m.expand(len(q), -1), qr)
            l_dec = ee_w * T.ee_loss_mm(enc.forward(qh), p_t, R_t) + ALPHA_J * ((qh - q) ** 2).sum(1).mean()
            d_true, d_hat = q - qr, qh - qr
            l_rel = (((d_hat - d_true) ** 2).sum(1) / ((d_true ** 2).sum(1) + 0.01 ** 2)).mean()
            loss = LAMBDA1 * l_dec + REL * l_rel
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step()
            tot += loss.item()
        ev, remain, cos = evaluate()
        mon = 0.5 * (ev["uniform"].mean().item() + ev["expert"].mean().item())
        ramp_done = ep >= WARMUP + RAMP - 1
        if ep == WARMUP + RAMP - 1:
            best = float("inf")
        if mon < best - 0.05:
            best, since = mon, 0
            torch.save({"W": ck["W"], "decoder": dec.state_dict(), "latent_dim": T.LATENT_DIM,
                        "epoch": ep, "config": {"delta_scale": scale, "dec_hidden": HIDDEN,
                                                "dec_depth": DEPTH, "source": "train_fr3_decoder.py"}}, OUT)
        elif ramp_done:
            since += 1
        if ramp_done:
            sch.step(mon)
        hist.append({"epoch": ep, "loss": tot / steps, "uniform_mm": ev["uniform"].mean().item(),
                     "expert_mm": ev["expert"].mean().item(), "remain": remain, "cos": cos})
        if ep % 5 == 0 or since == 0:
            print(f"[ep {ep:3d}] loss={tot / steps:.4f} ee_w={ee_w:.1f} | EE uniform mean="
                  f"{ev['uniform'].mean():.2f} med={ev['uniform'].median():.2f} mm  expert mean="
                  f"{ev['expert'].mean():.2f} med={ev['expert'].median():.2f} mm | 0.01 rad request: "
                  f"remain={remain:.2f} cos={cos:.2f} | best={best:.2f} since={since} "
                  f"lr={opt.param_groups[0]['lr']:.1e}")
        if ramp_done and since >= PATIENCE:
            print(f"[stop] epoch {ep}, best monitor {best:.2f} mm")
            break
    os.makedirs(os.path.join(R, "validation_out"), exist_ok=True)
    with open(os.path.join(R, "validation_out", "fr3_decoder_dec2_hist.json"), "w") as f:
        json.dump(hist, f)
    print(f"[done] {OUT}\n  use it: export FR3_STAGE1_CKPT={OUT}")


if __name__ == "__main__":
    main()