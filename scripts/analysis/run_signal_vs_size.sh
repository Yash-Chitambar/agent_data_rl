#!/usr/bin/env bash
# Plan D §10.3 (3): contact signal against peg size, scripted, no policy (CPU).
# A-S5 minimum detectable force and A-S6 drop-in peak-to-noise for each opening (2 N press, 1 mm clearance).
# usage: scripts/analysis/run_signal_vs_size.sh -> results/D/signal/A-S{5,6}_*.json
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
S=".venv/bin/python -m blind_insertion.checks.signal_checks --env_file blind_insertion/configs/phase_c_yam.json --out results/D/signal"
for o in 0.06 0.03 0.015 0.008; do
  ej=""; case $o in 0.015|0.008) ej='{"pad_spheres":4}';; esac
  $S --only 5 --family scale:$o:5e-05 ${ej:+--env_json "$ej"} > results/D/signal/s5_$o.log 2>&1 &
  $S --only 6 --press 2 --family scale:$o:0.001 ${ej:+--env_json "$ej"} > results/D/signal/s6_$o.log 2>&1 &
done
wait
