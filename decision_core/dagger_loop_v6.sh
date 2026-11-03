#!/usr/bin/env bash
# Iterative DAgger for the vision core: every round the CURRENT model collects its own errors,
# the expert relabels them, and training continues from the current model (only the core's own data).
#
#   round 0:  (INIT_CKPT=scratch, default) train EPOCHS epochs from scratch on expert_pp_v3 -> quick eval
#   round r:  collect (current ckpt) -> precompute latents -> render d435 -> encode tokens
#             -> train EPOCHS more epochs (init from current ckpt, all rounds' data) -> quick eval
#
#   absolute output, from scratch (round 0 = 15 epochs, then DAgger with that model):
#   cd ~/cross_embody && conda activate cross_embody
#   RESIDUAL=0 PREFIX=expert_pp_v3_adag TAG=_btL bash decision_core/dagger_loop_v4.sh 2>&1 | tee dagger_loop_abs.log
#
# Resume after a crash: same command (finished round 0 / collected datasets are skipped), or from round 3:
#   RESIDUAL=0 PREFIX=expert_pp_v3_adag TAG=_btL START_ROUND=3 INIT_CKPT=checkpoints/decision_core_vision_kp_K10_bi_l10_h8_s42_btL2.pt bash decision_core/dagger_loop_v4.sh
set -euo pipefail
cd ~/cross_embody
export MUJOCO_GL=egl

# ---------------- settings ----------------
ROUNDS=${ROUNDS:-4}                 # DAgger rounds
BASE=${BASE:-expert_pp_v3}          # base demonstrations (state expert: expert_pp_se)
ANCHOR=${ANCHOR:-0}                 # 1 = train with the anchor head (final approach from anchor + offset)
EPS=${EPS:-300}                     # kept episodes per round
EPOCHS=${EPOCHS:-15}                # training epochs per DAgger round
EPOCHS0=${EPOCHS0:-$EPOCHS}         # epochs of round 0 (the from-scratch model that collects round 1)
START_ROUND=${START_ROUND:-1}
INIT_CKPT=${INIT_CKPT:-scratch}     # "scratch" = round 0 trains a fresh model; or a checkpoint to start round 1 from
PREFIX=${PREFIX:-expert_pp_v3_rdag}   # datasets: expert_pp_v3_rdag1, _rdag2, ...
TAG=${TAG:-_brL}                    # checkpoints: ..._brL1.pt, ..._brL2.pt, ...
ENC=checkpoints/vision_encoder_v3.pt
GEN_ARGS=${GEN_ARGS:-""}
DAGGER_ARGS=${DAGGER_ARGS:-""}      # extra flags for dagger_pp_vision.py, e.g. "--p_post 0.8 --post_max 200"
WORKERS=${WORKERS:-8}
PRECOMPUTE=${PRECOMPUTE:-fr3_pick_gen/scripts/precompute_latents_pp.py}
EVAL_EPS=${EVAL_EPS:-50}            # eval after every round (same scenes each round -> comparable); 0 = skip
RESIDUAL=${RESIDUAL:-1}             # 1 = residual output, 0 = absolute output
COORD=${COORD:-uv}                  # xyz = two fixed cameras, triangulated (vision/render_encode_2cam.py)
ENC2=${ENC2:-vision/runs/vis2_fusion/best.pt}   # encoder of the two-camera set-up (COORD=xyz)

P="--pick_gen $HOME/cross_embody/fr3_pick_gen/scripts --scene_module build_scene --cam_name d435"
A="--ee_site tcp_site --cube object --table_geom table_top --width 768 --height 432"
COMMON="--dataset $BASE --feat kp --vis_source conf --hold_thr 0.75 --use_tcp --t_history 2 --n_heads 8 \
 --hist_noise 0 --hist_still 0.2 --tok_drop 0 --gpu_mem_frac 0.3 \
 $( [ "$RESIDUAL" = 1 ] && echo --residual || true ) $( [ "$COORD" = xyz ] && echo --coord xyz || true ) --tcp_loss 0.001 --rot_deg_per_cm 2 --near_w 4 $( [ "$ANCHOR" = 1 ] && echo --anchor --anc_mm 150 --select anc || echo --select near )"
DS=fr3_pick_gen/datasets

# ---------------- checks before hours of work ----------------
[ "$INIT_CKPT" = scratch ] || [ -f "$INIT_CKPT" ] || { echo "missing $INIT_CKPT"; exit 1; }
[ -f "$PRECOMPUTE" ] || { echo "missing $PRECOMPUTE -- set PRECOMPUTE=<path to precompute_latents_pp.py>"; exit 1; }
python "$PRECOMPUTE" --help | grep -q -- "--dataset" || {
  echo "$PRECOMPUTE has no --dataset flag; edit the precompute line below to match its --help"; exit 1; }
grep -q rot_deg_per_cm decision_core/train_pp_vision.py || { echo "old train_pp_vision.py (no rot_deg_per_cm)"; exit 1; }

FEATN=$( [ "$COORD" = xyz ] && echo xyz || echo kp )      # name part of the checkpoint (train_pp_vision.py)
ckpt_of() { echo "checkpoints/decision_core_vision_${FEATN}_K10_bi_l10_h8_s42${TAG}$1.pt"; }
quick_eval() {
  [ "$EVAL_EPS" -gt 0 ] || return 0                       # EVAL_EPS=0: no eval between rounds
  echo "---- eval ($EVAL_EPS test-scene episodes): $1"
  python decision_core/eval_pp_vision.py $P $A --ckpt "$1" --test_scene --episodes "$EVAL_EPS" 2>&1 \
      | tee "eval${TAG}$2.log" | grep -E "behaviour|yaw chain|==>" | tail -n 3 \
      || echo "!! eval of $1 failed (see eval${TAG}$2.log) -- continuing with the next round"
}

CUR=$INIT_CKPT
if [ "$CUR" = scratch ]; then
  CUR=$(ckpt_of 0)
  if ! grep -q "^\[done\]" "train${TAG}0.log" 2>/dev/null; then      # a ckpt alone may be from an interrupted run
    echo "================ round 0: train $EPOCHS0 epochs from scratch -> $CUR  ($(date '+%m-%d %H:%M'))"
    python decision_core/train_pp_vision.py $COMMON --max_epochs "$EPOCHS0" --tag "${TAG}0" 2>&1 | tee "train${TAG}0.log"
    [ -f "$CUR" ] || { echo "training did not write $CUR"; exit 1; }
    quick_eval "$CUR" 0
  else
    echo "[skip] round 0 already trained: $CUR"
  fi
fi

for r in $(seq "$START_ROUND" "$ROUNDS"); do
  NAME=${PREFIX}${r}
  echo "================ round $r: policy $CUR -> dataset $NAME  ($(date '+%m-%d %H:%M'))"

  if [ ! -f "$DS/$NAME/meta/info.json" ]; then           # written only when the collection finished
    python decision_core/dagger_pp_vision.py --ckpt "$CUR" --episodes "$EPS" --out_name "$NAME" \
        --seed $((100 + r)) $P $A $DAGGER_ARGS ${GEN_ARGS:+--gen_args "$GEN_ARGS"}
    # (an interrupted collection is NOT resumed: delete $DS/$NAME and rerun to start that round clean)
  else
    echo "[skip] $NAME already collected"
  fi

  echo "---- precompute latents"
  python "$PRECOMPUTE" --dataset "$NAME"

  if [ "$COORD" = xyz ]; then
    echo "---- render + encode + triangulate (two cameras)"
    (cd vision && python render_encode_2cam.py --pick_gen $HOME/cross_embody/fr3_pick_gen/scripts --scene_module build_scene \
        --encoder "$HOME/cross_embody/$ENC2" --episodes "$HOME/cross_embody/$DS/$NAME/data/chunk-000" \
        --out "$HOME/cross_embody/$DS/$NAME/tokens2" --workers 4)
  else
    echo "---- render"
    python vision/render_episodes.py $P $A --episodes "$HOME/cross_embody/$DS/$NAME/data/chunk-000" \
        --out "$HOME/cross_embody/$DS/$NAME/images" --workers "$WORKERS"
    echo "---- encode"
    python vision/encode_episode_tokens.py --images "$HOME/cross_embody/$DS/$NAME/images" --ckpt "$ENC" \
        --workers "$WORKERS"
  fi

  echo "---- train $EPOCHS epochs"
  EXTRA=$(for i in $(seq 1 "$r"); do printf "%s%s " "$PREFIX" "$i"; done)
  python decision_core/train_pp_vision.py $COMMON --extra_dataset $EXTRA --init_vision "$CUR" \
      --max_epochs "$EPOCHS" --tag "${TAG}${r}" 2>&1 | tee "train${TAG}${r}.log"
  CUR=$(ckpt_of "$r")
  [ -f "$CUR" ] || { echo "training did not write $CUR"; exit 1; }

  quick_eval "$CUR" "$r"
done
echo "done. last policy: $CUR"