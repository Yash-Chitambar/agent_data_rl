#!/bin/bash
# Part 2 closed-loop evaluations (results/usbc/policy/approach/). n = 2 x 512 per configuration.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU:-4}
run() { name=$1; shift; for s in 1 2; do .venv/bin/python scripts/task/eval_approach.py --nworld 512 --seed $s --name ${name}_s$s "$@" > /dev/null 2>&1; done; echo "$name done"; }
case "${1:-base}" in
base)
  run open_loop_zed --servo none --fric-comp
  run open_loop_zed_nofc --servo none
  run servo_oracle_fc --servo oracle --fric-comp
  run servo_oracle_nofc --servo oracle ;;
wrist)
  run servo_wrist_fc --servo wrist --fric-comp
  run servo_wrist_fine_fc --servo wrist --fric-comp --fine-below-mm 12
  run servo_wrist_fine_fc_stop3 --servo wrist --fric-comp --fine-below-mm 12 --stop 3
  run servo_wrist_fine_fc_stop10 --servo wrist --fric-comp --fine-below-mm 12 --stop 10
  run servo_wrist_fine_fc_frames4 --servo wrist --fric-comp --fine-below-mm 12 --frames 4 ;;
v2)   # wrist estimator trained with in-hand rotation +/-20 deg (covers the grasp sag seen in the loop)
  run servo_wristv2_fc --servo wrist --fric-comp --coarse-net wrist_v2
  run servo_wristv2_fc_frames4 --servo wrist --fric-comp --coarse-net wrist_v2 --frames 4
  run servo_wristv2_fc_stop3 --servo wrist --fric-comp --coarse-net wrist_v2 --stop 3
  run servo_wristv2_fc_stop10 --servo wrist --fric-comp --coarse-net wrist_v2 --stop 10 ;;
esac
# two-view fine estimator (right + left wrist RGB): run with `sweep_approach.sh both`
if [ "${1}" = both ]; then
  run servo_both_fine_fc --servo wrist --fric-comp --fine-below-mm 12 --fine-net wrist2_fine
  run servo_both_fine_fc_frames4 --servo wrist --fric-comp --fine-below-mm 12 --fine-net wrist2_fine --frames 4
fi
