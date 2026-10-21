#!/usr/bin/env bash
# Evaluate a USB-C checkpoint on each curriculum stage (2,048 matched starts, deterministic).
# usage: scripts/usbc_stage_eval.sh <run_name> <tag> [env_file]
# The start offset is passed as a half-width: median radial offset / 0.765 (train.py MEDIAN_RADIAL_PER_HALFWIDTH).
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2; CF=${3:-blind_insertion/configs/phase_e_usbc_spheres.json}
PY=.venv/bin/python; OUT=results/E/stages; mkdir -p $OUT
st() {  # stage clearance halfwidth [extra args]
  local s=$1 c=$2 hw=$3; shift 3
  [ -s $OUT/${TAG}_stage$s.parquet ] && { echo "skip stage $s"; return; }
  $PY -m blind_insertion.eval --policy runs/$RUN/last.pt --env_file $CF --family usbc --n_variants 320 \
    --episodes 2048 --clearance $c --peg_offset $hw "$@" --out $OUT/${TAG}_stage$s.parquet 2>&1 | grep -v "^Module" | tail -1
}
st 1 3e-4 6.54e-4
st 2 1.5e-4 1.307e-3
st 3 8e-5 1.961e-3
st 4 5e-5 2.614e-3
st 5 2e-5,1e-4 3.922e-3 --log_worlds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
