#!/bin/bash
# Faster servo settings on AlignTask (chain Coulomb FF, vision-done), n = 256 each; reference = align_ctrl/servo_ff (gain 1.5, vmax 8).
# bash scripts/eval/align_speed_sweep.sh results/real_align/align_speed
OUT=${1:-results/real_align/align_speed}; mkdir -p $OUT runs/real4
PY=/home/yashchitambar/peg_test_bimanual/.venv/bin/python
FF='"right_fric_comp":{"tau":[0.0275,0.546,0.819,0.0463,0.0433,0.0467],"ramp":0.005}'
ENVX="{\"align_done_on\":\"vision\",$FF}"
run() { [ -f $OUT/$2.json ] && return; CUDA_VISIBLE_DEVICES=$3 PYTHONPATH=. MUJOCO_GL=egl $PY scripts/eval/align_compare.py --method servo --kw "$1" --env_extra "$ENVX" --nworld 256 --out $OUT/$2.json > runs/real4/align_speed_$2.log 2>&1; }
B='"vmin_mm_s":1.5,"deadband_mm":0.4,"deadband_deg":0.4'
run "{$B,\"gain\":1.5,\"vmax_mm_s\":8}"  g15_v8 1 &
run "{$B,\"gain\":2.5,\"vmax_mm_s\":8}"  g25_v8 2 &
run "{$B,\"gain\":1.5,\"vmax_mm_s\":12}" g15_v12 3 &
run "{$B,\"gain\":2.5,\"vmax_mm_s\":12}" g25_v12 4 &
wait
run "{$B,\"gain\":3.5,\"vmax_mm_s\":12}" g35_v12 1 &
run "{$B,\"gain\":2.5,\"vmax_mm_s\":12,\"wmax_deg_s\":8}" g25_v12_w8 2 &
run "{$B,\"gain\":2.5,\"vmax_mm_s\":12,\"wmax_deg_s\":8,\"ang_gain\":2.0}" g25_v12_w8_a2 3 &
run "{$B,\"gain\":2.5,\"vmax_mm_s\":16}" g25_v16 4 &
wait
