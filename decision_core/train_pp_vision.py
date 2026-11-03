"""
VISION decision core for pick-and-place, FROZEN image encoder (object tokens precomputed), so an epoch
takes minutes. Same recipe as train_pp_chunk.py (next-K-state latent targets, state-based gripper
labels, null-space augmentation); no object coordinates: the cube enters as per-frame object tokens
[keypoint, latent] and the place target as the tape token in the goal query.

    z_true_k = W_FR3(FK(q_{t+k})),  k = 1..K
    loss = mean_k ||z_hat_k - z_true_k||^2  +  grip_lambda * mean_k BCE_k              (as train_pp_chunk)

Extra:
  --hist_drop  some samples get a motion-free arm history (all older steps = the current one + padding
               noise, like an episode start), so the core cannot extrapolate the arm and must read the cube
  swap test    every epoch on validation: the cube / tape tokens are shuffled across the batch; the change
               of the chunk's last latent (x347 ~ mm of TCP) says how much the decision depends on the images.
               Near 0 = the core ignores the images.

Needs, for --dataset:
  data_latent[_ns]/chunk-000/episode_*.parquet       (precompute_latents_pp.py)
  images/ep_*/obj_tokens.npz                          (vision/render_episodes.py + encode_episode_tokens.py)

Run:
  python decision_core/train_pp_vision.py --dataset expert_pp_v3                       # core from scratch
  python decision_core/train_pp_vision.py --dataset expert_pp_v3 --feat kp --tag _kp   # keypoints only
  python decision_core/train_pp_vision.py --dataset expert_pp_v3 --init_chunk checkpoints/decision_core_chunk_K10_bi_l10_h4_s42_pp2s.pt --tag _initpp2s
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

R = os.path.expanduser("~/cross_embody")
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "fr3_pick_gen", "scripts"))
from decision_core.model import load_frozen_stage1
from decision_core.vision_core import DecisionCoreVision, tcp_tokens, xyz_denorm, REL_M
from decision_core.latent_readout import fk_error, rot6_angle
from decision_core.vision_dataset import build_vision_datasets, build_extra_dataset
from torch.utils.data import ConcatDataset
import decision_core_dataset as _dcd
from decision_core_dataset import T_HISTORY, PADDING_NOISE_STD

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="expert_pp_v3")
ap.add_argument("--tokens", default="obj_tokens", help="name of the token files (encode_episode_tokens --name)")
ap.add_argument("--coord", choices=["uv", "xyz"], default="uv",
                help="xyz: object positions from the two fixed cameras (triangulated, robot base frame; "
                     "vision/render_encode_2cam.py) and the TCP position from FK instead of image keypoints")
ap.add_argument("--tok_dir", default="tokens2", help="coord xyz: token directory under the dataset")
ap.add_argument("--pos_noise_mm", type=float, default=1.0, help="coord xyz: noise on the object positions [mm]")
ap.add_argument("--tcp_noise_mm", type=float, default=2.0,
                help="coord xyz: per-sample offset of the TCP position [mm] (camera-robot calibration error)")
ap.add_argument("--feat", choices=["kp+z", "kp"], default="kp+z", help="object token: keypoint + latent, or keypoint only")
ap.add_argument("--n_freq", type=int, default=6, help="Fourier frequencies of the keypoint encoding")
ap.add_argument("--chunk", type=int, default=10)
ap.add_argument("--dec_layers", type=int, default=2)
ap.add_argument("--ns_prob", type=float, default=0.5)
ap.add_argument("--grip_lambda", type=float, default=1.0)
ap.add_argument("--hist_drop", type=float, default=0.3)
ap.add_argument("--hist_still", type=float, default=0.0,
                help="fraction of samples whose older history = EXACT copies of the current latent (no padding "
                     "noise): an arm that has stopped away from the target. hist_drop always adds the padding noise, "
                     "so without this a stopped arm is a state the core never saw (the residual core stalls there)")
ap.add_argument("--pair_swap", type=float, default=0.3)
ap.add_argument("--uv_noise_px", type=float, default=1.0, help="keypoint noise (pixels of the 768x432 image)")
ap.add_argument("--z_noise", type=float, default=0.05)
ap.add_argument("--vis_source", choices=["vis", "conf"], default="vis",
                help="token visibility: the encoder's vis head, or the label-free heatmap confidence (kp_conf.py); "
                     "the hold rule (--hold_thr) then uses the same signal")
ap.add_argument("--jump_px", type=float, default=0.0,
                help="object-permanence filter: while the gripper is open, a cube keypoint that jumps more than this "
                     "(px) from the last accepted one is held back until it stays there 5 frames (0 = off)")
ap.add_argument("--hold_thr", type=float, default=0.5,
                help="cube visibility below this -> keep its last seen keypoint (0 = off); evaluation does the same")
ap.add_argument("--tok_drop", type=float, default=0.1, help="per history frame: treat the cube as unseen")
ap.add_argument("--hist_noise", type=float, default=0.05,
                help="std of Gaussian noise on the arm-latent history (targets unchanged): states slightly off the "
                     "expert path, so the core learns to steer back instead of extrapolating")
ap.add_argument("--max_epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_layers", type=int, default=10)
ap.add_argument("--n_heads", type=int, default=4)
ap.add_argument("--history_attn", choices=["bi", "causal"], default="bi")
ap.add_argument("--init_chunk", default="", help="copy the matching weights of a trained chunk core (default: from scratch)")
ap.add_argument("--extra_dataset", nargs="*", default=[],
                help="extra TRAIN-only datasets, e.g. DAgger episodes (dagger_pp_vision.py); validation stays on --dataset")
ap.add_argument("--extra_weight", type=float, default=1.0,
                help="sampling weight of an extra-dataset sample relative to a base sample")
ap.add_argument("--residual", action="store_true",
                help="predict the chunk as the CURRENT arm latent + offsets (precision scales with the remaining "
                     "motion instead of a fixed floor from regressing absolute poses over the whole workspace)")
ap.add_argument("--oracle_kp", action="store_true",
                help="ORACLE perception (diagnostic upper bound): cube / tape keypoints = exact projections from the "
                     "render, visibility 1, keypoint-only token. Separates what the core can do from encoder errors")
ap.add_argument("--t_history", type=int, default=T_HISTORY, help="arm / object history length (default 8)")
ap.add_argument("--d_model", type=int, default=128, help="width of the core (128 = original; the latent stays 128-d)")
ap.add_argument("--use_tcp", action="store_true",
                help="TCP token: the arm's TCP (shared FK) projected into the image with the calibrated camera")
ap.add_argument("--tcp_noise_px", type=float, default=3.0,
                help="calibration-error augmentation: a constant random image offset of the TCP token per sample (px)")
ap.add_argument("--head", choices=["mse", "flow"], default="mse",
                help="mse: regress the chunk; flow: flow matching on the normalized latent chunk (multimodal)")
ap.add_argument("--flow_steps", type=int, default=10, help="Euler steps when sampling (validation / deployment)")
ap.add_argument("--tcp_loss", type=float, default=0.0,
                help="weight of the TASK-SPACE term: TCP position error implied by the latent error (exact first-order "
                     "map through the frozen embedding, latent_readout.tcp_error), in cm^2; 0 = off")
ap.add_argument("--rot_deg_per_cm", type=float, default=2.0,
                help="TCP loss also covers the gripper ORIENTATION: this many degrees cost as much as 1 cm of position "
                     "(0 = position only)")
ap.add_argument("--near_w", type=float, default=1.0,
                help="loss weight of samples whose TCP is within --near_mm (xy) of the cube (gripper open) or of the "
                     "place point (gripper closed) -- the frames that decide grasp / placement precision")
ap.add_argument("--near_mm", type=float, default=50.0)
ap.add_argument("--anchor", action="store_true",
                help="coord xyz: add the ANCHOR head -- per chunk step the TCP position relative to the observed cube "
                     "and to the observed place point; the final approach then uses anchor + offset (absolute, no drift)")
ap.add_argument("--anc_loss", type=float, default=1.0, help="weight of the anchor-head loss")
ap.add_argument("--anc_mm", type=float, default=80.0,
                help="anchor loss only on samples within this xy of the anchor (TCP-cube with the gripper open, "
                     "cube-place with it closed)")
ap.add_argument("--init_vision", default="", help="start from a trained vision core (fine-tuning, e.g. the v2 checkpoint)")
ap.add_argument("--tag", default="")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--patience", type=int, default=30)
ap.add_argument("--select", choices=["aux", "val", "last", "near", "tcp", "anc"], default="aux",
                help="best checkpoint by the arm-latent error (aux) or by aux + gripper BCE (val); the gripper BCE "
                     "grows from overconfidence on a few frames while its accuracy stays, so aux is the default; last = save "
                     "every epoch (DAgger fine-tuning: the expert-only validation cannot show what DAgger fixes)")
ap.add_argument("--grip_smooth", type=float, default=0.05, help="label smoothing of the gripper targets")
ap.add_argument("--gpu_mem_frac", type=float, default=0.5)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda" and args.gpu_mem_frac < 1:
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
WARMUP = max(200, int(2000 * 32 / args.batch))
FEAT = "oracle" if args.oracle_kp else args.feat.replace("+", "")
if args.coord == "xyz":
    FEAT = "xyz" + ("oracle" if args.oracle_kp else "")
CKPT = os.path.join(R, "checkpoints", f"decision_core_vision_{FEAT}_K{args.chunk}_{args.history_attn}_l{args.n_layers}"
                                      f"_h{args.n_heads}_s{args.seed}{args.tag}.pt")
torch.manual_seed(args.seed); np.random.seed(args.seed)
KEYS = ["z_hist", "finger_hist", "q_chunk", "grip_chunk", "cube_hist", "tape_goal", "q_hist", "cam_pos", "cam_xmat", "cam_K",
        "obj_now", "place_now", "closed_now"]


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in KEYS}


def lr_lambda(step, total):
    if step < WARMUP:
        return step / WARMUP
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - WARMUP) / max(1, total - WARMUP))))


def main():
    fk, emb, _, _ = load_frozen_stage1(device=DEV)
    _dcd.NS_PROB = args.ns_prob
    _dcd.CHUNK = args.chunk
    use_z = args.feat == "kp+z" and not args.oracle_kp
    TH = args.t_history
    if args.coord == "xyz":
        use_z = False
    xkw = dict(coord=args.coord, tok_dir=args.tok_dir)
    tr, va = build_vision_datasets(args.dataset, TH, **xkw, pos_noise_mm=args.pos_noise_mm, oracle=args.oracle_kp, vis_source=args.vis_source, jump_px=args.jump_px, name=args.tokens, n_freq=args.n_freq, use_z=use_z,
                                   pair_swap=args.pair_swap, uv_noise_px=args.uv_noise_px, z_noise=args.z_noise,
                                   hold_thr=args.hold_thr, tok_drop=args.tok_drop)
    weights = list(tr.weights)
    if args.extra_dataset:
        kw = dict(**xkw, pos_noise_mm=args.pos_noise_mm, vis_source=args.vis_source, jump_px=args.jump_px, name=args.tokens, n_freq=args.n_freq, use_z=use_z, pair_swap=args.pair_swap,
                  uv_noise_px=args.uv_noise_px, z_noise=args.z_noise, hold_thr=args.hold_thr, tok_drop=args.tok_drop)
        parts = [tr]
        for name in args.extra_dataset:
            ex = build_extra_dataset(name, TH, z_std=tr.z_std, oracle=args.oracle_kp, **kw)
            if ex.ckpt != tr.ckpt:
                raise ValueError(f"{name} tokens encoded with {ex.ckpt}, base with {tr.ckpt}")
            parts.append(ex); weights += [w * args.extra_weight for w in ex.weights]
        z_dim, ckpt_enc = tr.z_dim, tr.ckpt
        tr = ConcatDataset(parts); tr.z_dim, tr.ckpt = z_dim, ckpt_enc
    tl = DataLoader(tr, batch_size=args.batch, sampler=WeightedRandomSampler(weights, len(tr), replacement=True),
                    num_workers=args.workers, drop_last=True, pin_memory=True, collate_fn=collate,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    persistent_workers=args.workers > 0, pin_memory=True, collate_fn=collate, drop_last=True)
    K = args.chunk
    model = DecisionCoreVision(tr.z_dim, args.n_freq, use_z, TH, args.n_layers, args.n_heads, chunk=K,
                               dec_layers=args.dec_layers, causal=args.history_attn == "causal",
                               residual=args.residual, use_tcp=args.use_tcp, d_model=args.d_model,
                               head=args.head, flow_steps=args.flow_steps, coord=args.coord,
                               anchor=args.anchor).to(DEV)
    readout = True
    print(f"[loss] head {args.head}  tcp_loss {args.tcp_loss}  near_w {args.near_w} (< {args.near_mm:.0f} mm)  "
          f"TCP mm via the embedding Jacobian")
    if args.head == "flow" and not args.init_vision:          # latent statistics for the flow normalization
        zs = []
        with torch.no_grad():
            for i, b in enumerate(tl):
                qc = b["q_chunk"].to(DEV)
                zs.append(emb(fk.forward(qc.reshape(-1, 7))[:, :18]))
                if i >= 30:
                    break
        zs = torch.cat(zs)
        model.z_mu.copy_(zs.mean(0)); model.z_sd.copy_(zs.std(0).clamp(min=1e-3))
        print(f"[flow] latent normalization from {len(zs)} chunk states: |mu| {model.z_mu.norm():.3f}, "
              f"sd median {model.z_sd.median():.4f}")
    if args.init_vision:
        p = args.init_vision if os.path.isabs(args.init_vision) else os.path.join(R, args.init_vision)
        miss, unexp = model.load_state_dict(torch.load(p, map_location=DEV, weights_only=False)["state_dict"],
                                            strict=False)
        if unexp or any(not k.startswith("anc_head") for k in miss):
            raise RuntimeError(f"--init_vision {p}: missing {miss}, unexpected {unexp}")
        print(f"[init] weights from {p}" + (f" (new: {len(miss)} anchor-head tensors)" if miss else ""))
    if args.init_chunk:
        p = args.init_chunk if os.path.isabs(args.init_chunk) else os.path.join(R, args.init_chunk)
        print(f"[init] {len(model.init_from_chunk(p, DEV))} / {len(model.state_dict())} tensors from {p}")
    print(f"[vision] token = {args.feat} (keypoint {model.kp} + z {model.z_dim})  K={K}  encoder {tr.ckpt}\n"
          f"         params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M  hist_drop={args.hist_drop}  "
          f"pair_swap={args.pair_swap}  -> {CKPT}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.max_epochs * len(tl)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total))

    def step(b, train):
        z_hist = b["z_hist"].to(DEV); f_hist = b["finger_hist"].to(DEV)
        cube, tape = b["cube_hist"].to(DEV), b["tape_goal"].to(DEV)
        B = z_hist.shape[0]
        tcp = None
        if args.use_tcp:
            off = None
            if train and args.coord == "xyz" and args.tcp_noise_mm > 0:
                off = (torch.randn(B, 1, 3, device=DEV) * args.tcp_noise_mm / 1000.0).expand(B, z_hist.shape[1], 3)
            elif train and args.tcp_noise_px > 0:
                off = (torch.randn(B, 1, 2, device=DEV) * args.tcp_noise_px
                       / torch.tensor([768.0, 432.0], device=DEV)).expand(B, z_hist.shape[1], 2)
            with torch.no_grad():
                tcp = tcp_tokens(fk, b["q_hist"].to(DEV), b["cam_pos"].to(DEV), b["cam_xmat"].to(DEV),
                                 b["cam_K"].to(DEV), args.n_freq, off, coord=args.coord)
        if train and args.hist_drop > 0:
            dm = torch.rand(B, device=DEV) < args.hist_drop
            if dm.any():
                z_hist = z_hist.clone(); f_hist = f_hist.clone()
                z_hist[dm, :-1] = z_hist[dm, -1:] + torch.randn_like(z_hist[dm, :-1]) * PADDING_NOISE_STD
                f_hist[dm, :-1] = f_hist[dm, -1:]
                if tcp is not None:
                    tcp = tcp.clone(); tcp[dm, :-1] = tcp[dm, -1:]
        if train and args.hist_still > 0:
            sm = torch.rand(B, device=DEV) < args.hist_still
            if sm.any():
                z_hist = z_hist.clone(); f_hist = f_hist.clone()
                z_hist[sm, :-1] = z_hist[sm, -1:]
                f_hist[sm, :-1] = f_hist[sm, -1:]
                if tcp is not None:
                    tcp = tcp.clone(); tcp[sm, :-1] = tcp[sm, -1:]
        if train and args.hist_noise > 0:
            z_hist = z_hist + torch.randn_like(z_hist) * args.hist_noise
        qc = b["q_chunk"].to(DEV)
        with torch.no_grad():
            fq = fk.forward(qc.reshape(-1, 7))
            f_true = fq[:, :18].view(B, K, 18)
            z_true = emb(fq[:, :18]).view(B, K, -1)
            tcp_true = fq[:, :3].view(B, K, 3)
            tcp_now = fk.forward(b["q_hist"][:, -1].to(DEV))[:, :3]
            closed = b["closed_now"].to(DEV) > 0.5
            tgt = torch.where(closed[:, None], b["place_now"].to(DEV), b["obj_now"].to(DEV))
            near = (tcp_now[:, :2] - tgt[:, :2]).norm(dim=-1) < args.near_mm / 1000
            w = torch.where(near, torch.full_like(tcp_now[:, 0], args.near_w), torch.ones_like(tcp_now[:, 0]))
            w = w / w.mean()
        wmean = lambda per: (w * per).mean()
        if args.head == "flow":
            x1 = model.normalize(z_true)
            x0 = torch.randn_like(x1)
            tt = torch.sigmoid(torch.randn(B, device=DEV))            # logit-normal: more weight to mid t
            xt = (1 - tt)[:, None, None] * x0 + tt[:, None, None] * x1
            v_hat, logit = model(z_hist, f_hist, cube, tape, tcp, x_t=xt, t=tt)
            fit_loss = wmean(((v_hat - (x1 - x0)) ** 2).sum(-1).mean(-1))
            z_est = model.denormalize(xt + (1 - tt)[:, None, None] * v_hat)   # one-step estimate of the chunk
            if train:
                z_hat = z_est
            else:                                                     # validation: real (deterministic) samples
                z_hat, _ = model(z_hist, f_hist, cube, tape, tcp, noise=torch.zeros_like(x1))
        else:
            z_hat, logit = model(z_hist, f_hist, cube, tape, tcp)
            z_est = z_hat
            fit_loss = wmean(((z_hat - z_true) ** 2).sum(-1).mean(-1))
        anc_l = torch.zeros((), device=DEV); anc_c = torch.zeros((), device=DEV); anc_t = torch.zeros((), device=DEV)
        if args.anchor and model.last_anc is not None:
            anc = model.last_anc                                                 # (B, K, 6), units of REL_M
            with torch.no_grad():
                cube_obs = xyz_denorm(cube[:, -1, :3]); tape_obs = xyz_denorm(tape[:, :3])   # what the core sees
                off_c = (tcp_true - cube_obs[:, None]) / REL_M
                off_t = (tcp_true - tape_obs[:, None]) / REL_M
                obj, plc = b["obj_now"].to(DEV), b["place_now"].to(DEV)
                m_c = (~closed) & ((tcp_now[:, :2] - obj[:, :2]).norm(dim=-1) < args.anc_mm / 1000)
                m_t = closed & ((obj[:, :2] - plc[:, :2]).norm(dim=-1) < args.anc_mm / 1000)
            hub = lambda a, t: F.smooth_l1_loss(a, t, reduction="none", beta=0.2).sum(-1).mean(-1)   # (B,)
            l_c, l_t = hub(anc[..., :3], off_c), hub(anc[..., 3:], off_t)
            anc_l = ((l_c * m_c).sum() + (l_t * m_t).sum()) / (m_c.sum() + m_t.sum()).clamp(min=1)
            with torch.no_grad():                                                # anchored TCP error, last chunk step
                e_c = (anc[:, -1, :3] - off_c[:, -1]).norm(dim=-1) * REL_M * 1000
                e_t = (anc[:, -1, 3:] - off_t[:, -1]).norm(dim=-1) * REL_M * 1000
                anc_c = e_c[m_c].mean() if m_c.any() else torch.zeros((), device=DEV)
                anc_t = e_t[m_t].mean() if m_t.any() else torch.zeros((), device=DEV)
        aux = ((z_hat - z_true) ** 2).sum(-1).mean()
        tcp_l = torch.zeros((), device=DEV); tcp_mm = torch.zeros(K, device=DEV); near_mm = torch.zeros((), device=DEV)
        rot_deg = torch.zeros(K, device=DEV); near_deg = torch.zeros((), device=DEV)
        if readout is not None:
            dfk = fk_error(emb, f_true, z_est if train else z_hat, z_true)            # (B, K, 18)
            err = dfk[..., :3].norm(dim=-1)                                           # (B, K) metres
            ang = rot6_angle(dfk[..., 3:9])                                           # (B, K) rad
            per = ((err / 0.01) ** 2).mean(-1)                                        # cm^2
            if args.rot_deg_per_cm > 0:
                per = per + ((ang / math.radians(args.rot_deg_per_cm)) ** 2).mean(-1)   # same units (1 cm ~ rot_deg)
            rot_deg = torch.rad2deg(ang).mean(0)
            near_deg = torch.rad2deg(ang[near, -1]).mean() if near.any() else torch.zeros((), device=DEV)
            if args.head == "flow" and train:
                per = per * tt                    # the one-step estimate is only meaningful near t = 1
            tcp_l = wmean(per)
            tcp_mm = 1000 * err.mean(0)
            near_mm = 1000 * err[near, -1].mean() if near.any() else torch.zeros((), device=DEV)
        gc = b["grip_chunk"].to(DEV)
        gs = gc * (1 - 2 * args.grip_smooth) + args.grip_smooth if train else gc
        grip = F.binary_cross_entropy_with_logits(logit, gs)
        acc1 = ((logit[:, 0] > 0).float() == gc[:, 0]).float().mean()
        gap = (z_hat - z_true).norm(dim=-1)
        swap = torch.zeros((), device=DEV)
        if not train:                                     # does the decision depend on the images?
            perm = torch.roll(torch.arange(B, device=DEV), B // 2)
            z_sw, _ = (model(z_hist, f_hist, cube[perm], tape[perm], tcp, noise=torch.zeros_like(z_hat))
                       if args.head == "flow" else model(z_hist, f_hist, cube[perm], tape[perm], tcp))
            swap = (z_sw[:, -1] - z_hat[:, -1]).norm(dim=-1).mean()
        still = (z_hist[:, -1:] - z_true).norm(dim=-1)                 # 'stay where you are' baseline
        loss = fit_loss + args.grip_lambda * grip + args.tcp_loss * tcp_l + args.anc_loss * anc_l
        return (loss, aux, grip, acc1, gap[:, 0].mean(), gap[:, -1].mean(), swap,
                still[:, 0].mean(), still[:, -1].mean(), tcp_mm[0], tcp_mm[-1], near_mm, near.float().mean(),
                rot_deg[0], rot_deg[-1], near_deg, anc_c, anc_t)

    best, since = float("inf"), 0
    for ep in range(args.max_epochs):
        t0 = time.time(); model.train(); tot = 0.0
        for b in tl:
            loss, *_ = step(b, True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item()
        model.eval(); m = np.zeros(18)
        with torch.no_grad():
            for b in vl:
                m += np.array([x.item() for x in step(b, False)])
        m /= len(vl)
        msg = (f"[ep {ep:3d}] ({time.time() - t0:5.1f}s) train={tot / len(tl):.4f}  val={m[0]:.4f}  aux={m[1]:.4f}  "
               f"zgap k=1 {m[4]:.3f} (≈{m[4] * 347:.0f} mm, stay-still {m[7] * 347:.0f})  k={K} {m[5]:.3f} "
               f"(≈{m[5] * 347:.0f} mm, stay-still {m[8] * 347:.0f})  "
               f"grip acc(k=1)={100 * m[3]:.1f}%  | swap test: Δ ≈{m[6] * 347:.0f} mm")
        if readout is not None:
            msg += (f"\n          TCP (readout, true mm): k=1 {m[9]:.1f}  k={K} {m[10]:.1f}  | near target "
                    f"({100 * m[12]:.0f}% of samples) k={K} {m[11]:.1f} mm\n          gripper orientation (deg): k=1 "
                    f"{m[13]:.2f}  k={K} {m[14]:.2f}  | near target k={K} {m[15]:.2f}")
        if args.anchor:
            msg += (f"\n          ANCHORED TCP (k={K}, mm): to the cube {m[16]:.1f}  to the place point {m[17]:.1f}  "
                    f"(compare: residual readout near target {m[11]:.1f})")
        score = {"aux": m[1], "val": m[1] + args.grip_lambda * m[2], "last": m[1],
                 "near": m[11],                              # TCP mm, last chunk step, samples near cube / place point
                 "tcp": m[10],                               # TCP mm, last chunk step, all samples
                 "anc": m[11] + m[16] + m[17]}[args.select]  # residual near target + anchored (cube, place point)
        if args.select == "last" or score < best * 0.995:
            best, since = score, 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_loss": float(m[0]),
                        "config": {"input": "vision_tokens", "t_history": TH, "oracle_kp": args.oracle_kp, "n_layers": args.n_layers,
                                   "n_heads": args.n_heads, "chunk": K, "dec_layers": args.dec_layers,
                                   "causal": args.history_attn == "causal", "z_dim": tr.z_dim, "n_freq": args.n_freq,
                                   "use_z": use_z, "encoder_ckpt": tr.ckpt, "tokens": args.tokens,
                                   "dataset": args.dataset, "ns_prob": args.ns_prob, "hist_drop": args.hist_drop,
                                   "init_chunk": args.init_chunk, "hold_thr": args.hold_thr,
                                   "tok_drop": args.tok_drop, "hist_noise": args.hist_noise,
                                   "extra_dataset": args.extra_dataset, "init_vision": args.init_vision,
                                   "residual": args.residual, "hist_still": args.hist_still,
                                   "use_tcp": args.use_tcp, "tcp_noise_px": args.tcp_noise_px,
                                   "d_model": args.d_model, "vis_source": args.vis_source,
                                   "jump_px": args.jump_px, "head": args.head, "flow_steps": args.flow_steps,
                                   "tcp_loss": args.tcp_loss, "near_w": args.near_w, "near_mm": args.near_mm,
                                   "rot_deg_per_cm": args.rot_deg_per_cm, "coord": args.coord,
                                   "tok_dir": args.tok_dir, "pos_noise_mm": args.pos_noise_mm,
                                   "tcp_noise_mm": args.tcp_noise_mm, "anchor": args.anchor}}, CKPT)
            msg += "  -> best"
        else:
            since += 1
        print(msg, flush=True)
        if since >= args.patience:
            print(f"[stop] no improvement for {args.patience} epochs")
            break
    print(f"[done] {CKPT}")


if __name__ == "__main__":
    main()