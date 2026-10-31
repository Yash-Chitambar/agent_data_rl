#!/bin/bash
# Part 3 sweep (results/usbc/policy/hold/). GPU 4 by default.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU:-4}
O=results/usbc/policy/hold
run() { name=$1; shift; echo "== $name"; .venv/bin/python scripts/task/eval_left_hold.py "$@" --out $O/$name.json 2>&1 | grep -E '"pass_rate"|"n"' ; }
run press_f15_push0-20 --mode press --force 15 --push 0 20 --nworld 2048
run press_f25_push0-20 --mode press --force 25 --push 0 20 --nworld 2048
run press_f35_push0-20 --mode press --force 35 --push 0 20 --nworld 2048
run press_f25_openloop_push0-20 --mode press --force 25 --push 0 20 --nworld 2048 --open-loop
run brace_f5_push0-20 --mode brace --force 5 --push 0 20 --nworld 1024
run press_f25_idle7s --mode press --force 25 --push 0 0 --push-hold-s 7 --nworld 1024
run press_f25_insert --mode press --force 25 --load insert --nworld 1024
run press_f35_insert --mode press --force 35 --load insert --nworld 1024
