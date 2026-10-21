#!/usr/bin/env bash
# Logged rollouts for the YAM presentation examples: matched starts (256 episodes, worlds 0-63
# logged) for the base policy and the policy trained with each factor, under that factor.
# usage: CUDA_VISIBLE_DEVICES=<gpu> scripts/yam_example_rollouts.sh
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
PY=.venv/bin/python; CF=blind_insertion/configs/phase_c_yam.json; O=results/C/examples
LW=$(seq -s, 0 63)
mkdir -p $O
ev() {  # run test_tag env_json
  [ -f $O/$1_on_$2.parquet ] && return
  $PY -m blind_insertion.eval --policy runs/C-peg-proprio-$1-s0/last.pt --env_file $CF --episodes 256 \
      --log_worlds $LW ${3:+--env_json "$3"} --out $O/$1_on_$2.parquet
}
ev base base
ev base latency1 '{"latency": 1}'
ev latency1 latency1 '{"latency": 1}'
ev base backlash3 '{"backlash_arcmin": 3}'
ev backlash3 backlash3 '{"backlash_arcmin": 3}'
G4='{"gains": [[320,320,320,120,120,120],[24,24,24,8,8,8]]}'
ev base gain4 "$G4"
ev gain4x gain4 "$G4"
