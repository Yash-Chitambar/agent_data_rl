#!/usr/bin/env bash
# Phase E evals for a USB-C checkpoint (plan §11.4).
# usage: scripts/eval/phase_e.sh <run_name> <tag> <full|indist> [--env_json J]
#            [--env_file F] [--variant_kw K]
#   indist: final-stage distribution (clearance 20-100 um log-uniform, median start offset
#           3 mm), 2,048 matched starts             -> results/E/eval_<tag>_final.parquet
#   full:   indist + E2 capture radius (median offset 0.5..5 mm, 1,024 each)
#                                                  -> results/E/capture_<tag>_off<mm>.parquet
#           + E3 clearance sweep at 2 mm median offset incl. the §11.5 +20/-10 um checks
#             around 50 um (1,024 each)            -> results/E/clear_<tag>_c<m>.parquet
# --env_file defaults to the sphere-pad USB-C config (box pads drop the drive, report_usbc/);
# --env_json must repeat any EnvCfg override used in training (e.g. {"history": 10});
# --variant_kw is passed to builder.sample_usbc (E4 chamfer: {"chamfer": [2e-4, 5e-4]}).
# --peg_offset is the per-axis uniform half-width a; median radial offset = 0.765 a
# (train.MEDIAN_RADIAL_PER_HALFWIDTH), the same conversion the training curriculum uses.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/server/env.sh
RUN=$1; TAG=$2; MODE=${3:-full}; shift 3 || true
EF=blind_insertion/configs/phase_e_usbc_spheres.json; EJ=""; VK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --env_json) EJ=$2; shift 2;;
    --env_file) EF=$2; shift 2;;
    --variant_kw) VK=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
CK=runs/$RUN/last.pt
[ -f "$CK" ] || { echo "missing $CK" >&2; exit 1; }
half() { python3 -c "print(round($1 / 1000 / 0.765, 7))"; }   # median mm -> half-width m
E=("$PY" -m blind_insertion.eval --policy "$CK" --env_file "$EF" --family usbc --n_variants 320
   ${EJ:+--env_json "$EJ"} ${VK:+--variant_kw "$VK"})
mkdir -p results/E
"${E[@]}" --episodes 2048 --peg_offset "$(half 3)" --log_worlds 0,1,2,3 --out "results/E/eval_${TAG}_final.parquet"
[ "$MODE" = full ] || exit 0
for o in 0.5 1 2 3 4 5 6 8; do
  "${E[@]}" --episodes 1024 --peg_offset "$(half $o)" --out "results/E/capture_${TAG}_off${o}mm.parquet"
done
for c in 2e-5 4e-5 5e-5 7e-5 1e-4 2e-4; do
  "${E[@]}" --episodes 1024 --clearance "$c" --peg_offset "$(half 2)" --out "results/E/clear_${TAG}_c$c.parquet"
done
