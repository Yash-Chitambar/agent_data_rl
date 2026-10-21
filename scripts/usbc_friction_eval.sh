#!/usr/bin/env bash
# USB-C policy under default (pyramidal, impratio 1) and stiff (elliptic, impratio 10) friction,
# with backlash and the latch; final stage, lab table, 2,048 starts each
# usage: scripts/usbc_friction_eval.sh <run_name> <tag> -> results/E/robust/<tag>_<cond>.parquet
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2; S='"mj_opt": {"cone": 1, "impratio": 10}'
declare -A C=([soft]='{}' [stiff]="{$S}" [stiff_bl1]="{$S, \"backlash_arcmin\": 1}" [stiff_bl3]="{$S, \"backlash_arcmin\": 3}"
              [stiff_lat1]="{$S, \"latency\": 1}" [soft_bl1]='{"backlash_arcmin": 1}')
for k in soft stiff stiff_bl1 stiff_bl3 stiff_lat1 soft_bl1; do
  out=results/E/robust/${TAG}_$k.parquet; [ -f "$out" ] && continue
  .venv/bin/python -m blind_insertion.eval --policy runs/$RUN/last.pt --env_file blind_insertion/configs/phase_e_usbc_spheres.json \
    --family usbc --n_variants 320 --episodes 2048 --peg_offset 0.0039216 --env_json "${C[$k]}" --out "$out" 2>&1 | grep success | sed "s#^#$k: #"
done
for k in latch_soft latch_stiff; do
  out=results/E/robust/${TAG}_$k.parquet; [ -f "$out" ] && continue
  ej='{}'; [ $k = latch_stiff ] && ej="{$S}"
  .venv/bin/python -m blind_insertion.eval --policy runs/$RUN/last.pt --env_file blind_insertion/configs/phase_e_usbc_latch.json \
    --family usbc --n_variants 320 --episodes 2048 --peg_offset 0.0039216 --env_json "$ej" --out "$out" 2>&1 | grep success | sed "s#^#$k: #"
done
