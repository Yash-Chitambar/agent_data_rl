#!/usr/bin/env bash
# USB-C policy under the Phase C robustness factors (plan §9.3 brittleness on the USB-C task):
# final-stage distribution (20-100 um, median start 3 mm), lab-table mount, sphere pads,
# 2,048 matched starts per condition -> results/E/robust/<tag>_<cond>.parquet
# usage: scripts/usbc_brittleness.sh <run_name> <tag> [env_file]
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2; EF=${3:-blind_insertion/configs/phase_e_usbc_spheres.json}
H=$(python3 -c "print(round(3e-3/0.765, 7))")
E=(.venv/bin/python -m blind_insertion.eval --policy runs/$RUN/last.pt --env_file $EF --family usbc --n_variants 320 --episodes 2048 --peg_offset $H)
mkdir -p results/E/robust
declare -A C=([base]='' [lat1]='{"latency": 1}' [lat2]='{"latency": 2}' [bl1]='{"backlash_arcmin": 1}'
              [bl3]='{"backlash_arcmin": 3}' [qn5e-4]='{"q_noise": 5e-4}' [quant14]='{"q_quant_bits": 14}')
for k in base lat1 lat2 bl1 bl3 qn5e-4 quant14; do
  out=results/E/robust/${TAG}_$k.parquet
  [ -f "$out" ] && continue
  "${E[@]}" ${C[$k]:+--env_json "${C[$k]}"} --out "$out" 2>&1 | grep success
done
