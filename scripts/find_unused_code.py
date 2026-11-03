#!/usr/bin/env python3
"""
Which code files does the MAIN PIPELINE actually use? Everything else (diag_*, check_*, test scripts, old versions)
is listed as unused and can be moved to archive/code_unused/. Dry run by default.

Used = reachable from the entry points below by
  - Python imports (import x / from x import y, also inside functions),
  - a string literal equal to a module name (importlib / --scene_module build_scene_arm, ...),
  - a "python path/to/file.py" or "bash path/to/file.sh" inside a used .sh file / a used .py file.

  python scripts/find_unused_code.py                 # list used / unused
  python scripts/find_unused_code.py --protocol      # also keep the 10/22 protocol pipeline (run_protocol.py ...)
  python scripts/find_unused_code.py --apply         # move the unused ones to archive/code_unused/<folder>/
                                                     # (recorded in archive/MOVES.txt; tidy.py --undo moves them back)
"""
import argparse, ast, glob, os, re, shutil

R = os.path.expanduser("~/cross_embody")
DIRS = ["decision_core", "scripts", "vision", os.path.join("fr3_pick_gen", "scripts"), "fk_encoder"]   # not the top folder
ENTRY = [
    # vision pick-and-place, state expert, DAgger, evaluation
    "decision_core/eval_pp_vision.py", "decision_core/train_pp_vision.py", "decision_core/dagger_pp_vision.py",
    "decision_core/state_expert.py", "decision_core/anchor_blend.py", "decision_core/new_arm.py",
    "decision_core/dagger_loop_v6.sh",
    # vision data + image encoder
    "vision/gen_vision_data.py", "vision/gen_episode_vis_data.py", "vision/train_vision_encoder.py",
    "vision/render_encode_2cam.py", "vision/encode_episode_tokens.py", "vision/triangulate.py",
    # expert data / scenes
    "fr3_pick_gen/scripts/build_scene.py", "fr3_pick_gen/scripts/build_scene_arm.py",
    "fr3_pick_gen/scripts/generate_fr3_pickplace_data.py", "fr3_pick_gen/scripts/generate_fr3_expert_data.py",
    "fr3_pick_gen/scripts/precompute_latents_pp.py", "fr3_pick_gen/scripts/precompute_nullspace.py",   # run by hand
    "env_x3.sh",
    # Stage 1 / Stage 2 / new arm / housekeeping
    "scripts/train_stage1.py", "scripts/train_fr3_decoder.py",
    "scripts/train_stage2_posture.py", "scripts/add_arm.sh", "scripts/tidy.py", "scripts/find_unused_code.py",
]
PROTOCOL = ["scripts/run_protocol.py", "scripts/eval_pp_ur5e.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--protocol", action="store_true", help="keep the 10/22 protocol pipeline too")
ap.add_argument("--entry", nargs="*", default=[], help="more entry points (paths relative to ~/cross_embody)")
a = ap.parse_args()

files = {}                                                       # module name -> [paths]
allf = []
for d in DIRS:
    for pat in ("*.py", "*.sh"):
        for p in glob.glob(os.path.join(R, d, pat)):
            allf.append(p)
            if p.endswith(".py"):
                files.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(p)
for p in glob.glob(os.path.join(R, "fk_encoder", "*.py")):
    files.setdefault("fk_encoder", []).append(p)                 # the package as a whole

entries = [os.path.join(R, e) for e in ENTRY + (PROTOCOL if a.protocol else []) + a.entry]
missing = [os.path.relpath(e, R) for e in entries if not os.path.exists(e)]
used, todo = set(), [e for e in entries if os.path.exists(e)]


def refs(p):
    out = set()
    src = open(p, errors="replace").read()
    for m in re.finditer(r"([\w./$~-]+\.(?:py|sh))\b", src):     # any path to a .py / .sh (calls, variables, defaults)
        q = m.group(1).lstrip("-")
        for base in (R, os.path.dirname(p)):
            c = os.path.normpath(os.path.join(base, q.replace("$HOME/cross_embody/", "").replace("~/cross_embody/", "")))
            if os.path.exists(c):
                out.add(c)
    if p.endswith(".py"):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return out
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names |= {x.name for x in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                names.add(n.module)
                names |= {f"{n.module}.{x.name}" for x in n.names}
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and re.fullmatch(r"[A-Za-z_]\w*", n.value):
                names.add(n.value)
        for nm in names:
            for part in nm.split("."):
                for q in files.get(part, []):
                    out.add(q)
    return out


while todo:
    p = todo.pop()
    if p in used:
        continue
    used.add(p)
    todo += [q for q in refs(p) if q not in used]

unused = sorted(p for p in set(allf) - used if not os.path.basename(p).startswith("__init__"))
print(f"used ({len(used)}):")
for p in sorted(used):
    print(f"   {os.path.relpath(p, R)}")
print(f"\nUNUSED ({len(unused)}) -> archive/code_unused/:")
for p in unused:
    print(f"   {os.path.relpath(p, R)}")
if missing:
    print(f"\n(entry points not found, ignored: {', '.join(missing)})")
if not a.apply:
    print("\nDry run: nothing changed. --apply moves the unused files.")
else:
    log = os.path.join(R, "archive", "MOVES.txt")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a") as f:
        for p in unused:
            d = os.path.join(R, "archive", "code_unused", os.path.relpath(p, R))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.move(p, d)
            f.write(f"{os.path.relpath(p, R)}\t{os.path.relpath(d, R)}\n")
    print(f"moved {len(unused)} files (undo: python scripts/tidy.py --undo)")