#!/usr/bin/env bash
# Job lists of docs/real/CONTACT_SENS_20261115.md (sim only). Usage: scripts/eval/contact_sens_all.sh <lane> <gpu>
#   lanes: t1_k4, t1_k7, t1_k7box, t1_k4box, t2_k4, t2_k7 (each runs its jobs one after the other on <gpu>)
# Then: python scripts/eval/contact_sens_table.py  -> results/real_align/contact_sens/{test1,test2}.{md,json}
set -u
LANE=$1; GPU=$2
cd "$(dirname "$0")/../.."
PY=${PY:-/home/yashchitambar/peg_test_bimanual/.venv/bin/python}
LOG=runs/contact_sens/logs; mkdir -p $LOG
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONPATH=.
K4=runs/insert_K4.pt; K7=runs/insert_K7.pt
VA=base,pmu_lo,pmu_hi,pmu_vhi; VB=ptc_2ms,ptc_8ms,pdamp_05,psolimp_soft; VC=padmu_08,padmu_20,padtc_6ms,grip_06,grip_14
t1() {  # policy tag place
  local P=$1 T=$2 PL=$3
  for job in "4x1.2 nom $VA" "4x1.2 nom $VB" "4x1.2 nom $VC" "2x1.2 nom base,pmu_lo,pmu_hi" "6x0.8 nom base,pmu_lo,pmu_hi" \
             "4x1.2 tight base,pmu_hi" "4x1.2 loose base,pmu_hi"; do
    set -- $job
    $PY scripts/eval/contact_sens.py --policy $P --tag $T --place $PL --pad $1 --plug $2 --variants $3 2>&1 | grep -v linesearch > $LOG/${T}_${PL}_$1_$2_${3//,/-}.log
    echo "$(date +%H:%M) $T $PL $job exit ${PIPESTATUS[0]}"
  done
}
t2() {  # policy tag
  for grp in truth:0.5,truth:1,truth:1.5,truth:2 truth:2.5,truth:3,real:0.5,real:1 real:1.5,real:2,real:2.5,real:3; do
    $PY scripts/eval/funnel_rings.py --policy $1 --tag $2 --groups $grp 2>&1 | grep -v linesearch > $LOG/funnel_$2_${grp//[:,]/_}.log
    echo "$(date +%H:%M) funnel $2 $grp exit ${PIPESTATUS[0]}"
  done
}
case $LANE in
  t1_k4) t1 $K4 k4 real ;;
  t1_k7) t1 $K7 k7 real ;;
  t1_k7box) t1 $K7 k7 box ;;
  t1_k4box) P=$K4; $PY scripts/eval/contact_sens.py --policy $K4 --tag k4 --place box --variants $VA 2>&1 | grep -v linesearch > $LOG/k4_box_A.log ;;
  t2_k4) t2 $K4 k4 ;;
  t2_k7) t2 $K7 k7 ;;
  *) echo "unknown lane $LANE" ;;
esac
