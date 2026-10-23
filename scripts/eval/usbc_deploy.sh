#!/usr/bin/env bash
# USB-C deployment-recipe tests under stiff friction: latency x backlash grid + latch
# usage: scripts/eval/usbc_deploy.sh <run_name> <tag> -> results/E/robust/<tag>_stiff_lat<k>_bl<b>.parquet
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2; S='"mj_opt": {"cone": 1, "impratio": 10}'
for c in "0 0" "1 0" "2 0" "1 1" "1 3" "2 1" "2 3"; do
  set -- $c; out=results/E/robust/${TAG}_stiff_lat$1_bl$2.parquet; [ -f "$out" ] && continue
  .venv/bin/python -m blind_insertion.eval --policy runs/$RUN/last.pt --env_file blind_insertion/configs/phase_e_usbc_spheres.json \
    --family usbc --n_variants 320 --episodes 2048 --peg_offset 0.0039216 \
    --env_json "{$S, \"latency\": $1, \"backlash_arcmin\": $2}" --out "$out" 2>&1 | grep success | sed "s#^#lat$1 bl$2: #"
done
