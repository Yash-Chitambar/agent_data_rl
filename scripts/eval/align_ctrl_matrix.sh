#!/bin/bash
# Step and servo on AlignTask x {plain PD, chain Coulomb FF, precision controller}, vision-done, n = 256 each.
# CUDA_VISIBLE_DEVICES=1 bash scripts/eval/align_ctrl_matrix.sh results/real_align/align_ctrl
OUT=${1:-results/real_align/align_ctrl}; mkdir -p $OUT runs/real4
PY=/home/yashchitambar/peg_test_bimanual/.venv/bin/python
FF='"right_fric_comp":{"tau":[0.0275,0.546,0.819,0.0463,0.0433,0.0467],"ramp":0.005}'
PC='"ctrl_precision":{"integral":true,"settle_lock":true,"friction_ff":true,"friction":[0.138,0.447,1.087,0.069,0.091,0.087],"deadband":0.0004,"latch_band":0.001,"unlatch_band":0.002,"freeze":0.012}'
SERVO='{"vmin_mm_s":1.5,"deadband_mm":0.4,"deadband_deg":0.4}'
run() { [ -f $OUT/$4.json ] && return; PYTHONPATH=. MUJOCO_GL=egl $PY scripts/eval/align_compare.py --method $1 --kw "$2" --env_extra "$3" --nworld 256 --out $OUT/$4.json > runs/real4/align_ctrl_$4.log 2>&1; }
for m in step servo; do
  kw='{}'; [ $m = servo ] && kw=$SERVO
  run $m "$kw" '{"align_done_on":"vision"}' ${m}_pd &
  run $m "$kw" "{\"align_done_on\":\"vision\",$FF}" ${m}_ff
  wait
  run $m "$kw" "{\"align_done_on\":\"vision\",$PC}" ${m}_prec
done
