#!/usr/bin/env bash
# Plan §11.5: replay 20 successes and up to 20 failures of the USB-C policy for a visual check of
# physics artifacts (plug through the tongue, sideways snaps). Qualitative.
# usage: GPU=2 scripts/analysis/run_replay_check.sh <run> <tag> [env_json]  -> results/E/replay_check/<tag>*
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU:-0}
RUN=$1; TAG=$2; EJ=${3:-}; O=results/E/replay_check
.venv/bin/python -m blind_insertion.eval --policy runs/$RUN/last.pt --env_file blind_insertion/configs/phase_e_usbc_spheres.json \
  --family usbc --n_variants 320 --episodes 2048 --peg_offset 0.0039216 ${EJ:+--env_json "$EJ"} \
  --demo_k 20 --out $O/$TAG.parquet 2>&1 | grep success
for k in "20 0 s" "0 5 f"; do set -- $k
  .venv/bin/python -m blind_insertion.viz.demos --rollouts $O/$TAG.rollouts.npz --out $O/${TAG}_$3 \
    --title "USB-C $TAG replay check" --k_success $1 --k_fail $2 --ncol 5 2>&1 | grep wrote
done
