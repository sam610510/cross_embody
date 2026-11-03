#!/usr/bin/env python3
"""Encode every rendered episode frame (render_episodes.py) into OBJECT TOKEN inputs, frozen encoder.

Per episode writes images/ep_XXXXXX/<--name>.npz:
  uv      (F, 2, 2) float32  soft-argmax keypoint of cube / tape, normalized to [0, 1] (u / W, v / H)
  vis     (F, 2)    float16  visibility probability (the encoder's vis head)
  conf    (F, 2)    float16  heatmap confidence (kp_conf.heatmap_conf): mass of the densest 5x5 window, label-free
  z       (F, 2, D) float16  appearance/state latent (D = 0 for a keypoint-only encoder)
  *_pair  same for the paired renders (other arm colour) at pair_frames
  ckpt    encoder checkpoint used (decision core training and evaluation must use the same one)

Also reports the keypoint error against the render labels (meta.npz cube_uv / tape_uv) on frames
where the object is visible -- the token is only as good as this.

  python encode_episode_tokens.py --images ~/cross_embody/fr3_pick_gen/datasets/expert_pp_v3/images \
      --ckpt ~/cross_embody/checkpoints/decision_core_vision_e2e_K10_bi_l10_h4_fusion_s42_vis_e2e.pt
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from object_latent import load_any_encoder
from kp_conf import heatmap_conf


class Frames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(Image.open(self.paths[i]).convert("RGB"), np.uint8)).permute(2, 0, 1)


@torch.no_grad()
def encode(enc, in_hw, paths, dev, batch, workers):
    UV, V, Z, C = [], [], [], []
    for x in DataLoader(Frames(paths), batch, num_workers=workers):
        x = x.to(dev).float() / 255.0
        H0, W0 = x.shape[-2:]
        if tuple(x.shape[-2:]) != in_hw:
            x = F.interpolate(x, size=in_hw, mode="bilinear", align_corners=False)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = enc(x)
        uv = o["uv"].float() / torch.tensor([in_hw[1], in_hw[0]], device=dev)
        UV.append(uv.cpu()); V.append(torch.sigmoid(o["vis_logit"].float()).cpu())
        C.append(heatmap_conf(o["prob"]).cpu())
        Z.append(o["z"].float().cpu() if "z" in o else torch.zeros(len(x), 2, 0))
    return (torch.cat(UV).numpy().astype(np.float32), torch.cat(V).numpy().astype(np.float16),
            torch.cat(Z).numpy().astype(np.float16), torch.cat(C).numpy().astype(np.float16))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="obj_tokens")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu_mem_frac", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and a.gpu_mem_frac < 1:
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem_frac)
    enc, in_hw, has_z = load_any_encoder(os.path.expanduser(a.ckpt), dev)
    print(f"[encoder] {a.ckpt}  input {in_hw}  latent z: {'yes' if has_z else 'no (keypoints only)'}")
    eps = sorted(glob.glob(os.path.join(os.path.expanduser(a.images), "ep_*")))
    errs = {0: [], 1: []}
    cerr = []                                     # (conf, px error) of the cube, all frames: is conf a good alarm?
    for k, ep in enumerate(eps):
        out = os.path.join(ep, f"{a.name}.npz")
        rgb = sorted(glob.glob(os.path.join(ep, "rgb", "*.jpg")))
        pair = sorted(glob.glob(os.path.join(ep, "pair", "*.jpg")))
        if os.path.exists(out) and not a.overwrite:
            continue
        uv, v, z, c = encode(enc, in_hw, rgb, dev, a.batch, a.workers)
        if pair:
            uvp, vp, zp, cp = encode(enc, in_hw, pair, dev, a.batch, a.workers)
        else:
            uvp, vp, zp, cp = (np.zeros((0, 2, 2), np.float32), np.zeros((0, 2), np.float16),
                               np.zeros((0, 2, z.shape[-1]), np.float16), np.zeros((0, 2), np.float16))
        np.savez(out, uv=uv, vis=v, z=z, conf=c, uv_pair=uvp, vis_pair=vp, z_pair=zp, conf_pair=cp,
                 pair_frames=np.array([int(os.path.basename(p)[:5]) for p in pair], int),
                 ckpt=os.path.abspath(os.path.expanduser(a.ckpt)))
        mt = np.load(os.path.join(ep, "meta.npz"))
        W, H = Image.open(rgb[0]).size
        for o, key in ((0, "cube_uv"), (1, "tape_uv")):
            px = np.linalg.norm((uv[:, o] - mt[key] / np.array([W, H])) * np.array([W, H]), axis=1)
            errs[o].append(px[v[:, o].astype(np.float32) > 0.5])
            if o == 0:
                cerr.append(np.stack([c[:, 0].astype(np.float32), px], 1))
        if k % 50 == 0 or k == len(eps) - 1:
            st = lambda e: (f"median {np.median(e):.2f} px (p95 {np.percentile(e, 95):.1f}, n={len(e)})" if len(e)
                            else "no visible frames")
            print(f"[{k + 1}/{len(eps)}] keypoint error on visible frames so far: cube {st(np.concatenate(errs[0]))}, "
                  f"tape {st(np.concatenate(errs[1]))}", flush=True)
    if cerr:
        ce = np.concatenate(cerr); good, bad = ce[ce[:, 1] < 3, 0], ce[ce[:, 1] > 10, 0]
        print(f"[conf] cube frames with error < 3 px: n={len(good)}, conf median {np.median(good):.2f} "
              f"(p5 {np.percentile(good, 5):.2f}) | error > 10 px: n={len(bad)}, conf median "
              f"{np.median(bad) if len(bad) else float('nan'):.2f} (p95 {np.percentile(bad, 95) if len(bad) else float('nan'):.2f})")
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            keep = ce[:, 0] >= thr
            print(f"   conf >= {thr:.1f}: keeps {100 * keep.mean():5.1f}% of frames, of the > 10 px errors "
                  f"{100 * np.mean(keep[ce[:, 1] > 10]) if (ce[:, 1] > 10).any() else 0:5.1f}% pass (lower = better alarm)")
    print("done")


if __name__ == "__main__":
    main()