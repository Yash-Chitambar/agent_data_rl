#!/bin/bash
# Round 6: why do arm-robust ZED nets (v4) fail on real frames while v1 transfers? One factor at a time
# from the v1 recipe; each net is trained (n = 16384, 25 epochs) and checked on the 123 real stationary
# frames (manual landmark) and on sim arm conditions. Usage: zed_ablation.sh <gpu> <name> [train flags...]
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$1
name=$2; shift 2
F="grep -v linesearch"
.venv/bin/python scripts/perception/train_approach_estimators.py --which zed --tag _$name --n 16384 --n-test 1024 --epochs 25 "$@" > runs/perception/abl_$name.log 2>&1
.venv/bin/python scripts/perception/check_real_zed.py --net zed_$name --out results/usbc/policy/approach/ablation/real_$name.json 2>&1 | grep bimanual_stationary
.venv/bin/python scripts/perception/eval_zed_conditions.py --nets zed_$name --n 256 --out results/usbc/policy/approach/ablation/cond_$name.json 2>&1 | grep -E "^(rest|parked|pressing) " | cut -c1-200
