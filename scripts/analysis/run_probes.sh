#!/usr/bin/env bash
# Plan C-R3 / D §10.2 / E7: linear probe + latent swap of the history latent for the YAM peg,
# the Phase D sizes at 0.05 mm, and the USB-C policies. One job per GPU (GPUS="0 1 2 3 4").
# usage: scripts/analysis/run_probes.sh  -> results/{C,D,E}/probe/<tag>.{json,png}
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
GPUS=(${GPUS:-0 1 2 3 4})
NR='"noise_rand":{"q_noise":[0,0.0001,0.0005],"q_quant_bits":[0,16,14],"backlash_arcmin":[0,1,3,5],"latency":[0,1,2]}'
SPH='"pad_spheres":4,"nconmax":128,"njmax":1024'
P=".venv/bin/python -m blind_insertion.analysis.probes"
JOBS=(
  "C/probe/yam_base    5    runs/C-peg-proprio-base-s0             --env_file blind_insertion/configs/phase_c_yam.json --n_variants 512"
  "D/probe/o60_c0.05   5    runs/D-peg-proprio-o60_c0.05_scaled-s0     --env_file blind_insertion/configs/phase_c_yam.json --n_variants 512 --family scale:0.06:5e-05 --env_json {$NR,\"peg_offset\":0.013072}"
  "D/probe/o30_c0.05   2.5  runs/D-peg-proprio-o30_c0.05_scaled-s0     --env_file blind_insertion/configs/phase_c_yam.json --n_variants 512 --family scale:0.03:5e-05 --env_json {$NR,\"peg_offset\":0.006536}"
  "D/probe/o15_c0.05   1.25 runs/D-peg-proprio-o15_c0.05_scaled_sph-s0 --env_file blind_insertion/configs/phase_c_yam.json --n_variants 512 --family scale:0.015:5e-05 --env_json {$NR,\"peg_offset\":0.003268,$SPH}"
  "D/probe/o8_c0.05    0.67 runs/D-peg-proprio-o8_c0.05_scaled_sph-s0  --env_file blind_insertion/configs/phase_c_yam.json --n_variants 512 --family scale:0.008:5e-05 --env_json {$NR,\"peg_offset\":0.001743,$SPH}"
  "E/probe/usbc_base   1    runs/E-usbc-proprio-curr-s0                --env_file blind_insertion/configs/phase_e_usbc_spheres.json --family usbc --n_variants 320 --peg_offset 0.0039216"
  "E/probe/usbc_h10    1    runs/E-usbc-proprio-h10-s0                 --env_file blind_insertion/configs/phase_e_usbc_spheres.json --family usbc --n_variants 320 --peg_offset 0.0039216 --env_json {\"history\":10}"
  "E/probe/usbc_nocontact 1 runs/E-usbc-proprio-nocontact-s0           --env_file blind_insertion/configs/phase_e_usbc_spheres.json --family usbc --n_variants 320 --peg_offset 0.0039216"
)
i=0
for j in "${JOBS[@]}"; do
  read -r tag near run rest <<<"$j"
  out=results/$tag; mkdir -p "$(dirname "$out")"
  [ -f "$out.json" ] && continue
  g=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$g $P --policy $run/last.pt --near_mm $near --out "$out" $rest > "$out.log" 2>&1 &
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
grep -h "probe\[" results/{C,D,E}/probe/*.log
