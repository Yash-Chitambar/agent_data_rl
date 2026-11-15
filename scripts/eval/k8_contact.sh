#!/usr/bin/env bash
# Contact grid of docs/real/CONTACT_SENS_20261115.md (test 1, rig spot) for a K8 candidate (docs/real/K8_20261115.md). Sim only.
# Same probe, seed and n = 1024 per cell as K4 / K7. Usage: scripts/eval/k8_contact.sh <policy.pt> <tag> <gpu>
# Output: results/real_align/k8/contact/<tag>_real_<pad>_<plug>.json; logs runs/k8_eval/logs/.
set -u
P=$1; T=$2; GPU=$3
cd "$(dirname "$0")/../.."
PY=${PY:-/home/yashchitambar/peg_test_bimanual/.venv/bin/python}
LOG=runs/k8_eval/logs; mkdir -p $LOG results/real_align/k8/contact
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYTHONPATH=.
VA=base,pmu_lo,pmu_hi,pmu_vhi; VB=ptc_2ms,ptc_8ms,pdamp_05,psolimp_soft; VC=padmu_08,padmu_20,padtc_6ms,grip_06,grip_14
for job in "4x1.2 nom $VA" "4x1.2 tight base,pmu_hi" "4x1.2 loose base,pmu_hi" "4x1.2 nom $VC" "4x1.2 nom $VB" \
           "2x1.2 nom base,pmu_lo,pmu_hi" "6x0.8 nom base,pmu_lo,pmu_hi"; do
  set -- $job
  $PY scripts/eval/contact_sens.py --policy $P --tag $T --place real --pad $1 --plug $2 --variants $3 --out_dir results/real_align/k8/contact \
    2>&1 | grep -av linesearch > $LOG/cs_${T}_$1_$2_${3//,/-}.log
  echo "$(date +%H:%M) $T $job exit ${PIPESTATUS[0]}"
done
