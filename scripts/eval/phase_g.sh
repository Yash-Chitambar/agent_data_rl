#!/usr/bin/env bash
# Phase G (gear meshing) evals for one run: in-dist (8192 matched starts, as the paper's
# Fig. 11 gear bars) + the 6 gear OOD bins of Table 6 (1024 each).
# usage: scripts/eval/phase_g.sh <run_name> <tag e.g. blind_s0>
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2
CK=runs/$RUN/last.pt
PY=.venv/bin/python
$PY -m blind_insertion.eval --task gear --policy $CK --episodes 8192 --log_worlds 0,1,2,3,4,5,6,7 --out results/G/eval_$TAG.parquet
for b in tighter taller much_taller mild_size extreme_size random_tooth; do
  $PY -m blind_insertion.eval --task gear --policy $CK --episodes 1024 --ood $b --out results/G/ood_${TAG}_$b.parquet
done
