#!/bin/bash
# Part 3 sweep (results/usbc/policy/hold/). GPU 4 by default.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=${GPU:-4}
O=results/usbc/policy/hold
run() { name=$1; shift; echo "== $name"; .venv/bin/python scripts/task/eval_left_hold.py "$@" --out $O/$name.json 2>&1 | grep -E '"pass_rate"|"n"' ; }
[ -z "$1" ] && run press_f15_push0-20 --mode press --force 15 --push 0 20 --nworld 2048
[ -z "$1" ] && run press_f25_push0-20 --mode press --force 25 --push 0 20 --nworld 2048
[ -z "$1" ] && run press_f35_push0-20 --mode press --force 35 --push 0 20 --nworld 2048
[ -z "$1" ] && run press_f25_openloop_push0-20 --mode press --force 25 --push 0 20 --nworld 2048 --open-loop
[ -z "$1" ] && run brace_f5_push0-20 --mode brace --force 5 --push 0 20 --nworld 1024
[ -z "$1" ] && run press_f25_idle7s --mode press --force 25 --push 0 0 --push-hold-s 7 --nworld 1024
[ -z "$1" ] && run press_f25_insert --mode press --force 25 --load insert --nworld 1024
[ -z "$1" ] && run press_f35_insert --mode press --force 35 --load insert --nworld 1024
# Recommended hold (front half of the top, in line with port 2, 35 N, guarded engage): pass vs fixed load
for F in 0 1 2 3 5 10; do
[ -z "$1" ] && run rec_press_f35_front_push$F --mode press --force 35 --along -0.02 --push $F $F --nworld 1024
done
[ -z "$1" ] && run rec_press_f35_front_insert --mode press --force 35 --along -0.02 --load insert --nworld 1024
[ -z "$1" ] && run rec_press_f35_front_push0-10_preload4 --mode press --force 35 --along -0.02 --preload 4 --push 0 10 --nworld 1024
# Round 2 (PC pose DR +/-1.5 cm/+/-5 deg, feet mu U(0.8, 1.0)): `sweep_left_hold.sh r2`
if [ "$1" = r2 ]; then
  for F in 35 25 15; do for P in 1 3 5 10; do
    run r2_press_f${F}_front_push$P --mode press --force $F --along -0.02 --push $P $P --nworld 1024
  done; done
fi
