#!/usr/bin/env bash
# Success / failure demonstration videos for every experiment with a local checkpoint.
# Each case: eval.py --demo_k (records all episodes, keeps k per outcome) -> results/demos/data/,
# then viz/demos.py renders a captioned grid -> results/demos/<case>.mp4 / _sheet.png / .json.
# usage: scripts/report/make_demos.sh [case-regex]     (cases whose video exists are skipped)
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
PY=.venv/bin/python; D=results/demos; mkdir -p $D/data
ONLY=${1:-.}
SPH=blind_insertion/configs/phase_e_usbc_spheres.json
H3=0.0039216   # USB-C median start 3 mm (half-width = median / 0.765)
case_() {  # name "title" eval-args...
  local name=$1 title=$2; shift 2
  [[ "$name" =~ $ONLY ]] || return 0
  [ -f "$D/$name.mp4" ] && return 0
  [ -f "$D/data/$name.parquet" ] || $PY -m blind_insertion.eval "$@" --demo_k 3 --out "$D/data/$name.parquet" 2>&1 | grep success | cut -c1-220
  $PY -m blind_insertion.viz.demos --rollouts "$D/data/$name.rollouts.npz" --out "$D/$name" --title "$title" ${DEMO_CAM:+--camera $DEMO_CAM} 2>&1 | grep wrote
}
B0=runs/B-peg-proprio-base-s0/last.pt; V0=runs/B-peg-rgb-base-s0/last.pt
E1=runs/E-usbc-proprio-currsph-s0/last.pt
case_ B_blind_indist "Phase B: blind policy (Flexiv), study pegs, unseen geometries" --policy $B0 --episodes 2048
case_ B_blind_random_convex "Phase B: blind policy, random convex pegs (OOD)" --policy $B0 --episodes 1024 --ood random_convex
case_ B_blind_extreme_size "Phase B: blind policy, extreme sizes (OOD)" --policy $B0 --episodes 1024 --ood extreme_size
case_ B_visual_indist "Phase B: wrist-RGB policy, study pegs" --policy $V0 --episodes 1024
case_ B_visual_tighter "Phase B: wrist-RGB policy, tighter clearance (OOD; blind 97%)" --policy $V0 --episodes 1024 --ood tighter
case_ B_spiral "Phase B: scripted spiral search (privileged hole pose)" --policy 'spiral{"gain":10,"speed":0.04}' --env_json '{"episode_s": 15.0}' --episodes 512
case_ G_gear_indist "Gear meshing: blind policy, unseen gear sets" --task gear --policy runs/G-gear-proprio-base-s0/last.pt --episodes 2048 --n_variants 256
case_ G_gear_random_tooth "Gear meshing: blind policy, random tooth profiles (OOD)" --task gear --policy runs/G-gear-proprio-base-s0/last.pt --episodes 1024 --n_variants 256 --ood random_tooth
case_ C_yam_latency1 "YAM (old mount): base policy with 1-step latency, not in training" --policy runs/C-peg-proprio-base-s0/last.pt --env_file blind_insertion/configs/phase_c_yam.json --env_json '{"latency": 1, "table_top": null, "fix_z": [-0.3, -0.1]}' --episodes 1024
case_ C_yam_backlash1 "YAM (old mount): base policy with 1 arcmin backlash, not in training" --policy runs/C-peg-proprio-base-s0/last.pt --env_file blind_insertion/configs/phase_c_yam.json --env_json '{"backlash_arcmin": 1, "table_top": null, "fix_z": [-0.3, -0.1]}' --episodes 1024
case_ E_usbc_final "USB-C, final stage (20-100 um, median start 3 mm), lab-table mount" --policy $E1 --env_file $SPH --family usbc --n_variants 320 --episodes 2048 --peg_offset $H3
case_ E_usbc_far "USB-C, far starts (up to 9 mm per axis), 50 um" --policy $E1 --env_file $SPH --family usbc --n_variants 320 --episodes 2048 --clearance 5e-5 --peg_offset 0.009
case_ E_usbc_chamfer "USB-C E4: chamfered port, policy never trained on it" --policy $E1 --env_file $SPH --family usbc --n_variants 320 --episodes 2048 --peg_offset $H3 --variant_kw '{"chamfer": [2e-4, 5e-4]}'
case_ E_usbc_latch "USB-C E5: 5-15 N latch, policy never trained on it" --policy $E1 --env_file blind_insertion/configs/phase_e_usbc_latch.json --family usbc --n_variants 320 --episodes 2048 --peg_offset $H3
case_ E_usbc_side "USB-C E6: port facing the robot, port-up policy" --policy $E1 --env_file blind_insertion/configs/phase_e_usbc_side.json --family usbc --n_variants 320 --episodes 1024 --peg_offset $H3
case_ E_usbc_latency1 "USB-C with 1-step latency, policy trained without" --policy $E1 --env_file $SPH --family usbc --n_variants 320 --episodes 2048 --peg_offset $H3 --env_json '{"latency": 1}'
# fine-tunes (exist once the overnight queue has trained them)
[ -f runs/E-usbc-proprio-noiserandft-s0/DONE ] && case_ E_usbc_noiserandft_latency1 "USB-C fine-tuned with randomized latency/backlash/noise, tested at latency 1" --policy runs/E-usbc-proprio-noiserandft-s0/last.pt --env_file $SPH --family usbc --n_variants 320 --episodes 2048 --peg_offset $H3 --env_json '{"latency": 1}'
[ -f runs/E-usbc-proprio-wide-s0/DONE ] && case_ E_usbc_wide_far "USB-C fine-tuned on wider starts, far starts (up to 9 mm per axis)" --policy runs/E-usbc-proprio-wide-s0/last.pt --env_file $SPH --family usbc --n_variants 320 --episodes 2048 --clearance 5e-5 --peg_offset 0.009
[ -f runs/E-usbc-proprio-latchft-s0/DONE ] && case_ E_usbc_latchft "USB-C fine-tuned with the latch (5-15 N)" --policy runs/E-usbc-proprio-latchft-s0/last.pt --env_file blind_insertion/configs/phase_e_usbc_latch.json --family usbc --n_variants 320 --episodes 2048 --peg_offset $H3
# Phase D (scale study), random robot conditions; args as in scripts/server/jobs/D_eval.txt
DNR='"noise_rand":{"q_noise":[0,0.0001,0.0005],"q_quant_bits":[0,16,14],"backlash_arcmin":[0,1,3,5],"latency":[0,1,2]}'
DSPH='"pad_spheres":4,"nconmax":128,"njmax":1024'
YAM=blind_insertion/configs/phase_c_yam.json
case_ D_o60_c1 "Scale study: 60 mm peg, 1.0 mm clearance, random robot errors" --policy runs/D-peg-proprio-o60_c1_scaled-s0/last.pt --env_file $YAM --family scale:0.06:0.001 --env_json "{$DNR,\"peg_offset\":0.013072}" --episodes 1024
DEMO_CAM=0.07,150,-22 case_ D_o8_c1 "Scale study: 8 mm peg, 1.0 mm clearance, random robot errors" --policy runs/D-peg-proprio-o8_c1_scaled_sph-s0/last.pt --env_file $YAM --family scale:0.008:0.001 --env_json "{$DNR,\"peg_offset\":0.001743,$DSPH}" --episodes 1024
[ -f runs/D-peg-proprio-o8_c0.1_scaled_sph-s0/DONE ] && DEMO_CAM=0.07,150,-22 case_ D_o8_c0.1 "Scale study: 8 mm peg, 0.1 mm clearance, random robot errors" --policy runs/D-peg-proprio-o8_c0.1_scaled_sph-s0/last.pt --env_file $YAM --family scale:0.008:0.0001 --env_json "{$DNR,\"peg_offset\":0.001743,$DSPH}" --episodes 1024
true
