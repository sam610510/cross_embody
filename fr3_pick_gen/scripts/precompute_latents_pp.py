"""
Adds the frozen Stage-1 FR3 latent of every frame (W_FR3(FK(q))) to a dataset, like
precompute_latents.py does for the reaching data.

  python fr3_pick_gen/scripts/precompute_latents_pp.py --dataset expert_pp_v1
reads   datasets/<dataset>/data/chunk-000/episode_*.parquet
writes  datasets/<dataset>/data_latent/chunk-000/episode_*.parquet  (+ column "latent")
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import torch

R = os.path.expanduser("~/cross_embody")
sys.path.insert(0, R)
from decision_core.model import load_frozen_stage1

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="expert_pp_v1")
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
root = os.path.join(R, "fr3_pick_gen", "datasets", args.dataset)
src, dst = os.path.join(root, "data", "chunk-000"), os.path.join(root, "data_latent", "chunk-000")


@torch.no_grad()
def main():
    fk, emb, _, _ = load_frozen_stage1(device=DEV)
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "episode_*.parquet")))
    assert files, f"no episodes in {src}"
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        q = torch.tensor(np.stack(df["observation.state"].values)[:, :7], dtype=torch.float32, device=DEV)
        df["latent"] = list(emb(fk.forward(q)[:, :18]).cpu().numpy())
        df.to_parquet(os.path.join(dst, os.path.basename(f)))
        if i % 50 == 0:
            print(f"[{i + 1}/{len(files)}] {os.path.basename(f)}  {len(df)} frames")
    print(f"[done] {len(files)} episodes -> {dst}")


if __name__ == "__main__":
    main()