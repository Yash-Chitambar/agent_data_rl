#!/usr/bin/env bash
# K7 evaluation set (docs/real/K7_20261114.md) for one insertion policy. Sim only. n = 1024 per cell (replay: 256 per run).
# Usage: scripts/eval/k7_evalset.sh <policy.pt> <tag> <gpu> [groups]
#   groups (default all): k2 mate6 mate10 sig2 sig3 place depth0 depth6 depth10 replay
# Common env for every policy: rig-spot layout (v2: PC nominal = real port 2 (231, 394) mm, 4 deg; PC +/-5 mm, +/-1.5 deg),
# no mating force unless the group sets it, training-only prior scale off (k2_matrix does this itself).
# JSON summaries -> results/real_align/k7/<tag>_<group>.json; logs -> runs/k7_eval/logs/.
set -u
POL=$1; TAG=$2; GPU=$3; shift 3
GROUPS_=${*:-k2 mate6 mate10 sig2 sig3 place depth0 depth6 depth10 replay}
cd "$(dirname "$0")/../.."
PY=${PY:-/home/yashchitambar/peg_test_bimanual/.venv/bin/python}
OUT=results/real_align/k7; LOG=runs/k7_eval/logs; mkdir -p $OUT $LOG
RIG='"task_layout": "results/placement/task_layout_real_v2.json", "pc_pose_range": [0.005, 0.005, 1.5]'
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONPATH=.
for g in $GROUPS_; do
  case $g in
    k2)     $PY scripts/eval/k2_matrix.py --policy $POL --plant real --env_extra "{$RIG, \"mate_force_n\": null}" --out $OUT/${TAG}_k2.json ;;
    mate6)  $PY scripts/eval/k2_matrix.py --policy $POL --plant real --env_extra "{$RIG, \"mate_force_n\": [6.0, 6.0]}" --out $OUT/${TAG}_mate6.json ;;
    mate10) $PY scripts/eval/k2_matrix.py --policy $POL --plant real --env_extra "{$RIG, \"mate_force_n\": [10.0, 10.0]}" --out $OUT/${TAG}_mate10.json ;;
    sig2)   $PY scripts/eval/k2_matrix.py --policy $POL --plant real --cells real_ho,ring3,mix --env_extra "{$RIG, \"mate_force_n\": null, \"vision_prior_sigma_scale\": [2.0, 2.0]}" --out $OUT/${TAG}_sig2.json ;;
    sig3)   $PY scripts/eval/k2_matrix.py --policy $POL --plant real --cells real_ho,ring3,mix --env_extra "{$RIG, \"mate_force_n\": null, \"vision_prior_sigma_scale\": [3.0, 3.0]}" --out $OUT/${TAG}_sig3.json ;;
    place)  $PY scripts/eval/k7_placement.py --policy $POL --out $OUT/${TAG}_place.json ;;
    depth0|depth6|depth10) $PY scripts/eval/k7_depth.py --policy $POL --mate ${g#depth} --out $OUT/${TAG}_${g}.json ;;
    replay) $PY scripts/eval/replay_real_handovers.py --policy $POL --tag $TAG --real-left-lead --env_extra "{$RIG, \"mate_force_n\": null, \"vision_prior_sigma_scale\": null}" --out $OUT/${TAG}_replay.json ;;
    *) echo "unknown group $g"; continue ;;
  esac 2>&1 | grep -v linesearch > $LOG/${TAG}_$g.log
  echo "$(date +%H:%M) $TAG $g exit ${PIPESTATUS[0]}"
done
