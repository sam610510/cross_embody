#!/usr/bin/env bash
# Add a new arm to the FR3-trained system -- fixed procedure, nothing tuned in between.
#
#   1. FK check      the arm's FK encoder (configs/robots.yaml) == the physics model in the scene (TCP, 20 random poses)
#   2. Stage 2       posture-aware contrastive alignment into the FR3 latent (train_stage2_posture.py), fixed recipe
#   3. zero-shot     the FR3 vision decision core + image encoder, unchanged, on the new arm:
#                    test seed 1, 50 episodes: random targets / holdout cube / holdout place
#
# Nothing here reads FR3 or new-arm demonstrations; the decision core and the image encoder are not touched.
#
#   cd ~/cross_embody; source env_x3.sh
#   ARM=ur5e_menagerie TAG=_clean SEED=0 bash scripts/add_arm.sh 2>&1 | tee add_arm_clean.log
set -e
cd ~/cross_embody
export MUJOCO_GL=egl
ARM=${ARM:-ur5e_menagerie}             # robots.yaml entry
TAG=${TAG:-_clean}                     # checkpoint tag: checkpoints/stage2_posture_${ARM}${TAG}.pt
SEED=${SEED:-0}
CK=${CK:-checkpoints/decision_core_vision_xyz_K10_bi_l10_h8_s42_x3s4.pt}   # FR3 vision core (frozen)
EVAL_EPS=${EVAL_EPS:-50}
EXTRA=${EXTRA:-}                       # extra eval flags, fixed for the whole run (e.g. "--anc_vmax 0.3")
# the fixed Stage 2 recipe (as for the reported UR5e results: elbow-up poses, keypoint posture term)
S2_FLAGS=${S2_FLAGS:---elbow_up --s_psi_deg 15}
PCK=checkpoints/stage2_posture_${ARM}${TAG}.pt
P="--pick_gen $HOME/cross_embody/fr3_pick_gen/scripts --scene_module build_scene --cam_name d435"
A="--ee_site tcp_site --cube object --table_geom table_top --width 768 --height 432"

echo "==== add_arm  ARM=$ARM TAG=$TAG SEED=$SEED  core=$CK  stage2: $S2_FLAGS  eval extra: '$EXTRA'  $(date)"

echo "==== 1/3 FK check"
NEW_ARM=$ARM python - <<EOF
import os, sys, numpy as np, torch, mujoco
R = os.path.expanduser("~/cross_embody")
for p in (R, os.path.join(R, "scripts"), os.path.join(R, "fr3_pick_gen", "scripts")):
    sys.path.insert(0, p)
import train_stage2_align as T
from fk_encoder.fk_encoder import FKEncoder
import build_scene_arm as B
cfg = B.robot_cfg(); dof = int(cfg["arm_dof"]); gp = B.PREFIX + B.gripper_prefix(cfg)
m = B.build_model(); d = mujoco.MjData(m)
enc = FKEncoder("$ARM", T.ROBOTS_YAML, project_root=R, device="cpu"); T.setup_target("$ARM", enc)
mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, B.MOUNT)
def under(b):
    while b > 0:
        if b == mount: return True
        b = m.body_parentid[b]
    return False
hinge = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
arm_j = [j for j in range(m.njnt) if under(m.jnt_bodyid[j]) and int(m.jnt_type[j]) in hinge
         and not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(gp)][:dof]
print("arm joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_j])
qadr = [m.jnt_qposadr[j] for j in arm_j]
lim = np.asarray(T.TGT_JOINT_LIMITS, float).reshape(dof, 2).clip(-np.pi, np.pi)
site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, B.TCP_SITE)
rng = np.random.default_rng(0); err = []
for _ in range(20):
    q = rng.uniform(lim[:, 0], lim[:, 1]); d.qpos[qadr] = q; mujoco.mj_kinematics(m, d)
    p = enc.forward(torch.tensor(q[None], dtype=torch.float32))[0, :3].numpy()
    err.append(np.linalg.norm(p - d.site_xpos[site]) * 1000)
print(f"TCP |FK encoder - physics| over 20 random poses: max {max(err):.2f} mm, median {np.median(err):.2f} mm")
if max(err) > 2.0:
    raise SystemExit("FK check FAILED (> 2 mm): fix configs/robots.yaml before aligning")
EOF

echo "==== 2/3 Stage 2 alignment -> $PCK"
if [ -f "$PCK" ]; then
  echo "!! $PCK exists -- refusing to overwrite (move it or use another TAG)"; exit 1
fi
python scripts/train_stage2_posture.py --robot "$ARM" --tag "$TAG" --seed "$SEED" --cache_tag "_s$SEED$TAG" $S2_FLAGS \
    2>&1 | tee "logs/stage2_posture_${ARM}${TAG}.log"

echo "==== 3/3 zero-shot vision eval ($EVAL_EPS episodes, test seed 1)"
E="python decision_core/eval_pp_vision.py $P $A --ckpt $CK --test_scene --episodes $EVAL_EPS --robot $ARM --posture_ckpt $PCK $EXTRA"
G="behaviour|grasped episodes|vision:|goal:|==>|start poses had"
$E                                             2>&1 | tee "eval_${ARM}${TAG}.log"            | grep -E "$G"
$E --target_split holdout --holdout_which cube  2>&1 | tee "eval_${ARM}${TAG}_hold_cube.log"  | grep -E "$G"
$E --target_split holdout --holdout_which place 2>&1 | tee "eval_${ARM}${TAG}_hold_place.log" | grep -E "$G"
echo "==== done $(date)"