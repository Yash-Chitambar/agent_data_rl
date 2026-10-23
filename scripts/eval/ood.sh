#!/usr/bin/env bash
# OOD bins (builder.OOD_BINS, 1,024 matched starts each) for a checkpoint with an env file,
# e.g. Phase C-R4 on the YAM. Mirrors the OOD loop of scripts/eval/phase_b.sh.
# usage: scripts/eval/ood.sh <run_name> <tag> <phase> <env_file> [env_json]
#   -> results/<phase>/ood_<tag>_<bin>.parquet
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/server/env.sh
RUN=$1; TAG=$2; PH=$3; EF=$4; EJ=${5:-}
CK=runs/$RUN/last.pt
[ -f "$CK" ] || { echo "missing $CK" >&2; exit 1; }
mkdir -p "results/$PH"
for b in tighter slender very_slender mild_size extreme_size random_convex; do
  "$PY" -m blind_insertion.eval --policy "$CK" --env_file "$EF" ${EJ:+--env_json "$EJ"} \
    --episodes 1024 --ood "$b" --out "results/$PH/ood_${TAG}_$b.parquet"
done
