#!/usr/bin/env bash
# Phase B evals for one run: in-dist (2048 matched starts) + 6 OOD bins (1024 each) [+ probe].
# usage: scripts/eval/phase_b.sh <run_name> <tag e.g. blind_s0> [probe]
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2
CK=runs/$RUN/last.pt
PY=.venv/bin/python
$PY -m blind_insertion.eval --policy $CK --episodes 2048 --log_worlds 0,1,2,3,4,5,6,7 --out results/B/eval_$TAG.parquet
for b in tighter slender very_slender mild_size extreme_size random_convex; do
  $PY -m blind_insertion.eval --policy $CK --episodes 1024 --ood $b --out results/B/ood_${TAG%_s*}_${TAG#*_}_$b.parquet
done
if [ "$3" == "probe" ]; then
  $PY -m blind_insertion.analysis.probes --policy $CK --out results/B/probe_$TAG
fi
