#!/usr/bin/env bash
# Phase C evals for a YAM checkpoint: in-dist + robustness factors (§9.3) + gain sweep (§9.4).
# usage: scripts/eval_phase_c.sh <run_name> <tag>
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2; CK=runs/$RUN/last.pt; PY=.venv/bin/python
CF=blind_insertion/configs/phase_c_yam.json
E="$PY -m blind_insertion.eval --policy $CK --env_file $CF --episodes 2048"
mkdir -p results/C
$E --log_worlds 0,1,2,3 --out results/C/eval_${TAG}_base.parquet
for s in 1e-4 5e-4; do $E --env_json "{\"q_noise\": $s}" --out results/C/eval_${TAG}_qnoise$s.parquet; done
for b in 14 16; do $E --env_json "{\"q_quant_bits\": $b}" --out results/C/eval_${TAG}_quant$b.parquet; done
for b in 1 3 5; do $E --env_json "{\"backlash_arcmin\": $b}" --out results/C/eval_${TAG}_backlash$b.parquet; done
for k in 1 2; do $E --env_json "{\"latency\": $k}" --out results/C/eval_${TAG}_latency$k.parquet; done
for g in 0.5 2 4; do
  $E --env_json "{\"gains\": [[$(python3 -c "print(','.join(str(80*$g) for _ in range(3))+','+','.join(str(30*$g) for _ in range(3)))")],[$(python3 -c "print(','.join(str(6*$g) for _ in range(3))+','+','.join(str(2*$g) for _ in range(3)))")]]}" --out results/C/eval_${TAG}_gain$g.parquet
done
