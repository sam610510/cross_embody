#!/usr/bin/env python3
"""
Tidy ~/cross_embody -- MOVES ONLY, nothing is deleted. Dry run by default.

  python tidy.py                 # print the plan (source -> destination), change nothing
  python tidy.py --apply         # do it; every move is appended to archive/MOVES.txt
  python tidy.py --undo          # move everything in archive/MOVES.txt back (newest first)

What stays where it is (the current main line and what the code reads by fixed paths):
  checkpoints   stage1_franka*, latent_readout*, vision_encoder*, the final vision core _x3s4 and the anchor
                ablation core _x3a, and every checkpoint matching --keep (regex)
  datasets      the datasets named in the config of _x3s4 (found automatically), expert_pp_se*,
                expert_pp_v2s (protocol of 2026-10-22), and --keep_ds
  vision        vis2*, vis2ep* data and runs/vis2_fusion_ep (the encoder of the main line)
  code          all .py / .sh files (only *.bak* copies are moved)

Where the rest goes -- ALL under archive/ (one place; delete sub-folders there when sure):
  archive/checkpoints/protocol_2026-10/    decision_core_l*_h4 / chunk / tcp cores (protocol, layer ablation)
  archive/checkpoints/ur5e_align_2026-10/  stage2_align_ur5e*, stage2_posture_ur5e*
  archive/checkpoints/explore/             older vision cores (kp / kpz / oracle / earlier xyz series)
  archive/cache/                           posture_pairs_ur5e* (rebuilt by train_stage2_posture.py)
  archive/datasets/                        older expert / DAgger datasets
  archive/vision_data/, archive/vision_runs/   older vision data and encoder runs
  archive/logs/<series>/                   top-folder logs NOT behind a reported number
  logs/<series>/                           top-folder logs behind reported numbers (x3s, x3a, eval_x3w3, eval_x3x4)
  archive/root_misc/                       images and dumps in the top folder
  archive/code_bak/<folder>/               *.bak* copies of code files
Files changed in the last 15 minutes are left alone (a run may still be writing them).
"""
import argparse, glob, os, re, shutil, sys, time

R = os.path.expanduser("~/cross_embody")
AR = os.path.join(R, "archive")                            # EVERYTHING that is moved goes under here
LOG = os.path.join(AR, "MOVES.txt")
ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--undo", action="store_true")
ap.add_argument("--keep", nargs="*", default=[], help="extra checkpoint name patterns (regex) to keep")
ap.add_argument("--keep_ds", nargs="*", default=[], help="extra dataset names to keep")
a = ap.parse_args()


def rel(p):
    return os.path.relpath(p, R)


if a.undo:
    if not os.path.exists(LOG):
        sys.exit("no archive/MOVES.txt")
    lines = [l.rstrip("\n").split("\t") for l in open(LOG) if "\t" in l]
    for src, dst in reversed(lines):
        s, d = os.path.join(R, src), os.path.join(R, dst)
        if os.path.exists(d) and not os.path.exists(s):
            os.makedirs(os.path.dirname(s), exist_ok=True); shutil.move(d, s); print(f"back  {dst} -> {src}")
    os.rename(LOG, LOG + f".undone_{int(time.time())}")
    sys.exit(0)

now = time.time()
fresh = lambda p: now - os.path.getmtime(p) < 15 * 60
plan = []                                                   # (src, dst)


def move(src, dst_dir):
    if not os.path.exists(src) or fresh(src):
        return
    plan.append((src, os.path.join(dst_dir, os.path.basename(src))))


# ---------------------------------------------------------------- checkpoints
CK = os.path.join(R, "checkpoints")
# only the LAST model of each line that is still used: x3s4 (final), x3s3 (holdout / UR5e results), x3a (anchor ablation)
# checkpoints: the final model (_x3s4) and the anchor ablation model (_x3a, ON / OFF on the same checkpoint)
KEEP_CK = [r"^stage1_franka", r"^latent_readout", r"^vision_encoder",
           r"^decision_core_vision_xyz_.*_x3(s4|a)\.pt$"] + a.keep
DS_FROM = [r"_x3s4\.pt$"] + a.keep         # datasets are kept only if THESE checkpoints were trained on them
kept_ck = []
for p in sorted(glob.glob(os.path.join(CK, "*.pt"))):
    n = os.path.basename(p)
    if any(re.search(k, n) for k in KEEP_CK):
        kept_ck.append(p); continue
    if re.match(r"stage2_(align|posture)_ur5e", n):
        move(p, os.path.join(AR, "checkpoints", "ur5e_align_2026-10"))
    elif re.match(r"decision_core_(l\d+_h\d+|chunk_|tcp_|h\d)", n):
        move(p, os.path.join(AR, "checkpoints", "protocol_2026-10"))
    else:
        move(p, os.path.join(AR, "checkpoints", "explore"))
for p in glob.glob(os.path.join(R, "cache", "posture_pairs_ur5e*")):
    move(p, os.path.join(AR, "cache"))

# ---------------------------------------------------------------- datasets (keep what the kept checkpoints name)
DS = os.path.join(R, "fr3_pick_gen", "datasets")
names = {os.path.basename(p) for p in glob.glob(os.path.join(DS, "*")) if os.path.isdir(p)}
keep_ds = {n for n in names if n.startswith("expert_pp_se")} | {"expert_pp_v2s"} | set(a.keep_ds)
try:
    import torch

    def walk(x):
        if isinstance(x, str):
            for n in names:
                if re.search(rf"(^|[/\s,]){re.escape(n)}($|[/\s,])", x):
                    keep_ds.add(n)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    for p in [p for p in kept_ck if any(re.search(k, os.path.basename(p)) for k in DS_FROM)]:
        try:
            walk(torch.load(p, map_location="cpu", weights_only=False).get("config", {}))
        except Exception as ex:
            print(f"  (could not read the config of {rel(p)}: {ex})")
except ImportError:
    print("  (no torch: datasets named in checkpoint configs are not detected -- run inside the conda env)")
for n in sorted(names - keep_ds):
    move(os.path.join(DS, n), os.path.join(AR, "datasets"))

# ---------------------------------------------------------------- vision
for p in glob.glob(os.path.join(R, "vision", "data", "*")):
    if os.path.isdir(p) and not re.match(r"vis[23]", os.path.basename(p)):
        move(p, os.path.join(AR, "vision_data"))
for p in glob.glob(os.path.join(R, "vision", "runs", "*")):
    if os.path.isdir(p) and os.path.basename(p) != "vis2_fusion_ep" and not re.match(r"vis3", os.path.basename(p)):
        move(p, os.path.join(AR, "vision_runs"))

# ---------------------------------------------------------------- top folder: logs, images, dumps


def series(n):
    m = re.search(r"(x3[a-z])", n)
    if m:
        return m.group(1)
    if n.startswith("dagger_loop"):
        return "dagger_loops"
    m = re.search(r"_(b[a-z])[A-Z_]", n)
    if m:
        return "b_series"
    return "misc"


def cited(n):
    """logs behind numbers in the project docs (results tables) stay in logs/; the rest goes to archive/logs/"""
    return bool(re.search(r"x3s|x3a|eval_x3w3|eval_x3x4", n))


for p in sorted(glob.glob(os.path.join(R, "*"))):
    n = os.path.basename(p)
    if not os.path.isfile(p):
        continue
    if re.match(r"(requirements|README|LICENSE|MOVES)", n, re.I):    # project files, not logs
        continue
    if re.search(r"\.(log|log~|txt)$", n):
        move(p, os.path.join(R if cited(n) else AR, "logs", series(n)))
    elif re.search(r"\.(png|jpg|jpeg|TXT)$", n):
        move(p, os.path.join(AR, "root_misc"))

# ---------------------------------------------------------------- code backups
for folder in ("decision_core", "scripts", "vision", os.path.join("fr3_pick_gen", "scripts")):
    for p in glob.glob(os.path.join(R, folder, "*")):
        if re.search(r"\.bak|_bak\b|\.orig$", os.path.basename(p)) and os.path.isfile(p):
            move(p, os.path.join(AR, "code_bak", folder))

# ---------------------------------------------------------------- report / apply
print(f"kept checkpoints ({len(kept_ck)}): " + ", ".join(os.path.basename(p) for p in kept_ck))
print(f"kept datasets: {', '.join(sorted(keep_ds & names))}")
groups = {}
for s, d in plan:
    groups.setdefault(rel(os.path.dirname(d)), []).append(os.path.basename(s))
for g, items in sorted(groups.items()):
    print(f"\n-> {g}/  ({len(items)})")
    for it in items[:12]:
        print(f"     {it}")
    if len(items) > 12:
        print(f"     ... +{len(items) - 12} more")
print(f"\n{len(plan)} moves planned." + ("" if a.apply else "  Dry run: nothing changed. Run with --apply to do it."))
if a.apply:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        for s, d in plan:
            if os.path.exists(d):
                print(f"  skip (exists): {rel(d)}"); continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.move(s, d)
            f.write(f"{rel(s)}\t{rel(d)}\n")
    print(f"done. moves recorded in {rel(LOG)}  (undo: python tidy.py --undo)")