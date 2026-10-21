#!/usr/bin/env bash
# Phase F evals for a visual USB-C checkpoint (plan §12.4 F1/F2/F4); the modality and fusion
# come from the checkpoint.
# usage: scripts/server/eval/eval_phase_f.sh <run_name> <tag> [env_file]
#   F1: final-stage distribution, 2,048 matched starts   -> results/F/eval_<tag>_final.parquet
#   F2: capture radius, median offset 0.5..15 mm, 1,024 each -> results/F/capture_<tag>_off<mm>.parquet
#   F4: sensor degradation (depth noise x2, depth dropout 20%, no camera/light randomization in
#       training -> shifted light, mount offset 5 mm)       -> results/F/degrade_<tag>_<name>.parquet
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/server/env.sh
RUN=$1; TAG=$2; EF=${3:-blind_insertion/configs/phase_f_usbc.json}
CK=runs/$RUN/last.pt
[ -f "$CK" ] || { echo "missing $CK" >&2; exit 1; }
half() { python3 -c "print(round($1 / 1000 / 0.765, 7))"; }
E=("$PY" -m blind_insertion.eval --policy "$CK" --env_file "$EF" --family usbc --n_variants 320)
mkdir -p results/F
"${E[@]}" --episodes 2048 --peg_offset "$(half 3)" --log_worlds 0,1,2,3 --out "results/F/eval_${TAG}_final.parquet"
for o in 0.5 1 2 3 5 8 12 15; do
  "${E[@]}" --episodes 1024 --peg_offset "$(half $o)" --out "results/F/capture_${TAG}_off${o}mm.parquet"
done
DM='"depth_model": {"min": 0.07, "max": 0.3, "noise": 0.0005, "dropout": 0.02, "edge_dropout": 0.5, "edge": 0.005}'
declare -A DEG=(
  [depthnoise2x]='{"depth_model": {"min": 0.07, "max": 0.3, "noise": 0.001, "dropout": 0.02, "edge_dropout": 0.5, "edge": 0.005}}'
  [dropout20]='{"depth_model": {"min": 0.07, "max": 0.3, "noise": 0.0005, "dropout": 0.2, "edge_dropout": 0.5, "edge": 0.005}}'
  [mount5mm]="{\"cam_rand\": {\"cam_pos\": 0.005, \"cam_rot_deg\": 2, \"light\": true, \"colors\": true}, $DM}"
)
for k in "${!DEG[@]}"; do
  "${E[@]}" --episodes 1024 --peg_offset "$(half 3)" --env_json "${DEG[$k]}" --out "results/F/degrade_${TAG}_$k.parquet"
done
