#!/usr/bin/env python3
"""Sanity-check a generated vision dataset.

1. Label consistency: projected (u, v) vs. centroid of the object's segmentation
   pixels on fully visible samples. The tape is flat, so its centroid should sit
   within ~1 px of the projected centre; the cube centroid is biased by a few px
   because the visible faces (top + sides) are not centred on the 3D centre.
2. Statistics: visibility, posture modes, EE-to-cube distance.
3. A montage PNG with the labels drawn on the images (green = cube, yellow = tape,
   red ring = label marked as hidden).

  python check_vision_data.py --data data/vis_fr3_train --n_show 16
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw


def load_labels(data):
    parts = sorted(glob.glob(os.path.join(data, "labels_*.npz")))
    if not parts:
        raise SystemExit(f"no labels_*.npz in {data}")
    loaded = [dict(np.load(p)) for p in parts]
    return {k: np.concatenate([l[k] for l in loaded]) for k in loaded[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_show", type=int, default=16)
    ap.add_argument("--max_check", type=int, default=2000)
    ap.add_argument("--vis_thr", type=float, default=0.1)
    a = ap.parse_args()

    L = load_labels(a.data)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    seg_ids = meta["seg_ids"]
    n = len(L["idx"])
    print(f"{n} samples, image {meta['width']}x{meta['height']}")
    print("posture modes:", dict(zip(meta["modes"], np.bincount(L["mode"], minlength=len(meta["modes"])).tolist())))

    for key in ("cube", "tape"):
        v = L[f"{key}_vis"]
        print(f"{key}: vis mean {v.mean():.2f} | >0.9 {np.mean(v > 0.9):.2f} | hidden(<= {a.vis_thr}) "
              f"{np.mean(v <= a.vis_thr):.2f} | npix median {np.median(L[f'{key}_npix']):.0f}")
        uv = L[f"{key}_uv"]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < meta["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < meta["height"])
        print(f"      uv inside image {inside.mean():.3f}")

    # label consistency on fully visible samples
    for key in ("cube", "tape"):
        sel = np.where(L[f"{key}_vis"] > 0.98)[0][: a.max_check]
        errs = []
        for i in sel:
            seg = np.array(Image.open(os.path.join(a.data, "seg", f"{L['idx'][i]:06d}.png")))
            ys, xs = np.nonzero(seg == seg_ids[key])
            if len(xs) == 0:
                continue
            c = np.array([xs.mean() + 0.5, ys.mean() + 0.5])   # pixel centres
            errs.append(np.linalg.norm(c - L[f"{key}_uv"][i]))
        if errs:
            e = np.array(errs)
            print(f"{key}: |projected uv - seg centroid| on {len(e)} visible samples: "
                  f"median {np.median(e):.2f} px, p95 {np.percentile(e, 95):.2f} px, max {e.max():.2f} px")

    d = L["ee_cube_dist"]
    print(f"EE-cube distance: <0.10 m {np.mean(d < 0.10):.2f} | 0.10-0.25 m {np.mean((d >= 0.10) & (d < 0.25)):.2f} "
          f"| >=0.25 m {np.mean(d >= 0.25):.2f}")

    # montage
    k = min(a.n_show, n)
    cols = 4
    rows = int(np.ceil(k / cols))
    W, H = meta["width"] // 2, meta["height"] // 2
    canvas = Image.new("RGB", (cols * W, rows * H))
    for j in range(k):
        i = j * n // k
        img = Image.open(os.path.join(a.data, "rgb", f"{L['idx'][i]:06d}.png")).convert("RGB")
        dr = ImageDraw.Draw(img)
        for key, col in (("cube", (0, 255, 0)), ("tape", (255, 220, 0))):
            u, v = L[f"{key}_uv"][i]
            hidden = L[f"{key}_vis"][i] <= a.vis_thr
            dr.line([(u - 8, v), (u + 8, v)], fill=col, width=2)
            dr.line([(u, v - 8), (u, v + 8)], fill=col, width=2)
            if hidden:
                dr.ellipse([u - 14, v - 14, u + 14, v + 14], outline=(255, 0, 0), width=3)
        mode = meta["modes"][L["mode"][i]]
        dr.text((6, 6), f"#{L['idx'][i]} {mode}  vis c={L['cube_vis'][i]:.2f} t={L['tape_vis'][i]:.2f}",
                fill=(255, 255, 255))
        canvas.paste(img.resize((W, H)), ((j % cols) * W, (j // cols) * H))
    out = os.path.join(a.data, "montage.png")
    canvas.save(out)
    print("montage:", out)


if __name__ == "__main__":
    main()
