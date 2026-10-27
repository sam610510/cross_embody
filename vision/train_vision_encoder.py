#!/usr/bin/env python3
"""Train / evaluate the object-position encoder (vision_plan step 3).

Train on FR3-only renders, evaluate on held-out FR3 renders and (zero-shot) on UR5e renders:
  python train_vision_encoder.py --train data/vis_fr3_train --val data/vis_fr3_val --out runs/vis_v1
  python train_vision_encoder.py --eval_only --ckpt runs/vis_v1/best.pt --val data/vis_ur5e_test

Metrics (per object): pixel error and world error [mm] on visible samples, broken down by
arm posture mode and by EE-to-cube distance, plus visibility accuracy. World positions come
from back-projecting (u, v) onto the known object-centre height with the calibrated camera.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vision_encoder import ObjectEncoder, heatmap_target, uv_to_world

OBJS = ("cube", "tape")


# --------------------------------------------------------------------------- data
def load_dir(d):
    parts = sorted(glob.glob(os.path.join(d, "labels_*.npz")))
    if not parts:
        raise SystemExit(f"no labels_*.npz in {d}")
    loaded = [dict(np.load(p)) for p in parts]
    L = {k: np.concatenate([l[k] for l in loaded]) for k in loaded[0]}
    L["dir"] = np.array([d] * len(L["idx"]))
    meta = json.load(open(os.path.join(d, "meta.json")))
    return L, meta


def load_dirs(dirs):
    Ls, metas = zip(*[load_dir(d) for d in dirs])
    L = {k: np.concatenate([l[k] for l in Ls]) for k in Ls[0]}
    return L, metas[0]


class VisDataset(Dataset):
    def __init__(self, L, roi, in_hw, train, vis_thr):
        self.L, self.roi, self.in_hw, self.train, self.vis_thr = L, roi, in_hw, train, vis_thr
        x0, y0, w, h = roi
        self.scale = np.array([in_hw[1] / w, in_hw[0] / h])
        self.off = np.array([x0, y0])

    def __len__(self):
        return len(self.L["idx"])

    def __getitem__(self, i):
        L = self.L
        path = os.path.join(L["dir"][i], "rgb", f"{L['idx'][i]:06d}.png")
        img = Image.open(path).convert("RGB")
        x0, y0, w, h = self.roi
        if (x0, y0, w, h) != (0, 0, img.width, img.height):
            img = img.crop((x0, y0, x0 + w, y0 + h))
        if img.size != (self.in_hw[1], self.in_hw[0]):
            img = img.resize((self.in_hw[1], self.in_hw[0]), Image.BILINEAR)
        img = np.asarray(img, np.float32) / 255.0
        uv = np.stack([(L[f"{k}_uv"][i] - self.off) * self.scale for k in OBJS]).astype(np.float32)
        vis = np.array([L[f"{k}_vis"][i] for k in OBJS], np.float32)
        if self.train:
            img = self.augment(img, uv)
        return torch.from_numpy(img).permute(2, 0, 1).contiguous(), torch.from_numpy(uv), torch.from_numpy(vis), i

    def augment(self, img, uv):
        rng = np.random.default_rng()
        H, W, _ = img.shape
        # photometric: brightness, contrast, per-channel gain, saturation, noise, occasional blur
        img = img * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08)
        mean = img.mean()
        img = (img - mean) * rng.uniform(0.7, 1.3) + mean
        img = img * rng.uniform(0.9, 1.1, 3)
        gray = img.mean(-1, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0.6, 1.4)
        if rng.random() < 0.3:
            k = np.array([1, 2, 1], np.float32) / 4
            img = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0, img)
            img = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, img)
        img = img + rng.normal(0, rng.uniform(0, 0.03), img.shape).astype(np.float32)
        # random erasing (Zhong et al.); a patch never covers an object centre, so labels stay valid
        if rng.random() < 0.5:
            for _ in range(rng.integers(1, 4)):
                area = rng.uniform(0.01, 0.08) * H * W
                ar = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
                eh, ew = int(min(H - 1, math.sqrt(area * ar))), int(min(W - 1, math.sqrt(area / ar)))
                ey, ex = rng.integers(0, H - eh), rng.integers(0, W - ew)
                if any(ex - 6 <= u <= ex + ew + 6 and ey - 6 <= v <= ey + eh + 6 for u, v in uv):
                    continue
                img[ey:ey + eh, ex:ex + ew] = rng.uniform(0, 1, (eh, ew, 3)) if rng.random() < 0.5 \
                    else rng.uniform(0, 1, 3)
        return np.clip(img, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- metrics
@torch.no_grad()
def evaluate(model, loader, L, meta, ds, dev, vis_thr, amp):
    model.eval()
    uv_all = np.zeros((len(L["idx"]), len(OBJS), 2))
    vp_all = np.zeros((len(L["idx"]), len(OBJS)))
    for img, _, _, i in loader:
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            out = model(img.to(dev, non_blocking=True).float())
        uv_all[i.numpy()] = out["uv"].float().cpu().numpy()
        vp_all[i.numpy()] = torch.sigmoid(out["vis_logit"].float()).cpu().numpy()
    uv_full = uv_all / ds.scale + ds.off
    rep = {}
    modes = meta["modes"]
    for k, name in enumerate(OBJS):
        vis = L[f"{name}_vis"] > vis_thr
        px = np.linalg.norm(uv_full[:, k] - L[f"{name}_uv"], axis=1)
        z = L[f"{name}_pos"][:, 2]
        pw = uv_to_world(uv_full[:, k], L["K"], L["cam_pos"], L["cam_xmat"], z)
        mm = 1000 * np.linalg.norm(pw - L[f"{name}_pos"], axis=1)
        e = mm[vis]
        r = {"n_visible": int(vis.sum()), "px_median": float(np.median(px[vis])),
             "mm_median": float(np.median(e)), "mm_mean": float(e.mean()),
             "mm_p95": float(np.percentile(e, 95)), "mm_max": float(e.max()),
             "vis_acc": float(np.mean((vp_all[:, k] > 0.5) == vis))}
        r["by_mode_mm_median"] = {modes[m]: float(np.median(mm[vis & (L["mode"] == m)]))
                                  for m in range(len(modes)) if np.any(vis & (L["mode"] == m))}
        dist = L["ee_cube_dist"]
        bins = [(0, 0.10), (0.10, 0.25), (0.25, 9)]
        r["by_ee_cube_dist_mm_median"] = {f"{lo:.2f}-{hi:.2f}m": float(np.median(mm[vis & (dist >= lo) & (dist < hi)]))
                                         for lo, hi in bins if np.any(vis & (dist >= lo) & (dist < hi))}
        part = vis & (L[f"{name}_vis"] <= 0.9)
        if part.any():
            r["partly_occluded_mm_median"] = float(np.median(mm[part]))
        rep[name] = r
    return rep, uv_full, vp_all


def fmt(rep):
    s = []
    for k, r in rep.items():
        s.append(f"{k}: {r['mm_median']:.1f}/{r['mm_mean']:.1f}/{r['mm_p95']:.1f} mm (med/mean/p95), "
                 f"{r['px_median']:.2f} px, vis_acc {r['vis_acc']:.3f}")
    return " | ".join(s)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="runs/vis")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="x0 y0 w h crop of the full image")
    ap.add_argument("--in_hw", type=int, nargs=2, default=None, help="network input H W (default: ROI size)")
    ap.add_argument("--norm", default="gn", choices=["gn", "bn"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--w_hm", type=float, default=1.0, help="heatmap (Gaussian target) loss weight")
    ap.add_argument("--w_vis", type=float, default=0.2)
    ap.add_argument("--huber_px", type=float, default=2.0)
    ap.add_argument("--vis_thr", type=float, default=0.1, help="visible fraction above which (u,v) is supervised")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_aug", action="store_true", help="debug: disable train-time augmentation")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (debug / ablation)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    amp = dev.type == "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    Lv, meta = load_dirs(a.val)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False) if a.ckpt else None
    if ck is not None:
        roi, in_hw, norm = ck["roi"], ck["in_hw"], ck["norm"]
    else:
        roi = a.roi or [0, 0, meta["width"], meta["height"]]
        in_hw = a.in_hw or [roi[3], roi[2]]
        norm = a.norm
    assert in_hw[0] % 16 == 0 and in_hw[1] % 16 == 0, "input H, W must be divisible by 16"

    model = ObjectEncoder(n_obj=len(OBJS), norm=norm, pretrained=ck is None and not a.no_pretrained).to(dev)
    model = model.to(memory_format=torch.channels_last)
    if ck is not None:
        model.load_state_dict(ck["model"])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device {dev} | params {n_par / 1e6:.2f} M | roi {roi} | input {in_hw}")

    ds_v = VisDataset(Lv, roi, in_hw, False, a.vis_thr)
    dl_v = DataLoader(ds_v, a.batch, shuffle=False, num_workers=a.workers, pin_memory=amp)

    if a.eval_only:
        rep, uv, vp = evaluate(model, dl_v, Lv, meta, ds_v, dev, a.vis_thr, amp)
        print(fmt(rep))
        print(json.dumps(rep, indent=1))
        tag = os.path.basename(os.path.normpath(a.val[0]))
        json.dump(rep, open(os.path.join(a.out, f"eval_{tag}.json"), "w"), indent=1)
        np.savez(os.path.join(a.out, f"pred_{tag}.npz"), uv=uv, vis_prob=vp, idx=Lv["idx"])
        return

    Lt, _ = load_dirs(a.train)
    ds_t = VisDataset(Lt, roi, in_hw, not a.no_aug, a.vis_thr)
    dl_t = DataLoader(ds_t, a.batch, shuffle=True, num_workers=a.workers, pin_memory=amp,
                      drop_last=True, persistent_workers=a.workers > 0)
    print(f"train {len(ds_t)} | val {len(ds_v)}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.max_epochs * len(dl_t)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    best, bad, step, hist = float("inf"), 0, 0, []
    for ep in range(a.max_epochs):
        model.train()
        t0, run = time.time(), np.zeros(3)
        for img, uv, vis, _ in dl_t:
            img = img.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            uv, vis = uv.to(dev), vis.to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                out = model(img)
            m = (vis > a.vis_thr).float()                                    # (B, K)
            l_uv = (F.huber_loss(out["uv"], uv, delta=a.huber_px, reduction="none").sum(-1) * m).sum() \
                / m.sum().clamp_min(1)
            h, w = out["logits"].shape[-2:]
            tgt = heatmap_target(uv, h, w, model.STRIDE)
            logp = out["logits"].flatten(2).log_softmax(-1)
            l_hm = ((-(tgt.flatten(2) * logp).sum(-1)) * m).sum() / m.sum().clamp_min(1)
            l_vis = F.binary_cross_entropy_with_logits(out["vis_logit"], m)
            loss = l_uv + a.w_hm * l_hm + a.w_vis * l_vis
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += [l_uv.item(), l_hm.item(), l_vis.item()]
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        run /= max(1, len(dl_t) if not a.max_steps else step)
        rep, _, _ = evaluate(model, dl_v, Lv, meta, ds_v, dev, a.vis_thr, amp)
        score = rep["cube"]["mm_median"] + rep["tape"]["mm_median"]
        hist.append({"epoch": ep, "loss_uv": run[0], "loss_hm": run[1], "loss_vis": run[2], **rep})
        flag = ""
        if score < best:
            best, bad, flag = score, 0, " *"
            torch.save({"model": model.state_dict(), "roi": roi, "in_hw": in_hw, "norm": norm,
                        "objs": OBJS, "epoch": ep, "val": rep, "args": vars(a)}, os.path.join(a.out, "best.pt"))
        else:
            bad += 1
        print(f"ep {ep:3d} {time.time() - t0:5.0f}s | uv {run[0]:.3f} hm {run[1]:.3f} vis {run[2]:.3f} | "
              f"{fmt(rep)}{flag}", flush=True)
        json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        if bad >= a.patience or (a.max_steps and step >= a.max_steps):
            break
    print(f"best val (cube+tape median mm): {best:.2f} -> {os.path.join(a.out, 'best.pt')}")


if __name__ == "__main__":
    main()
