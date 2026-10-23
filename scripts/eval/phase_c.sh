#!/usr/bin/env bash
# Phase C evals for a YAM checkpoint: in-dist + robustness factors (§9.3) + gain sweep (§9.4).
# usage: scripts/eval/phase_c.sh <run_name> <tag> [base_env_json]
# base_env_json (optional) is merged under every test condition, e.g. '{"action_clip": 1.0}' for a
# policy trained with a non-default env setting that is not one of the tested factors.
# ONLY=<factor> (base|qnoise|quant|backlash|latency|gain) runs one factor only, e.g. ONLY=backlash to
# redo the backlash columns after the 2026-10-21 backlash-model fix.
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
RUN=$1; TAG=$2; BASE=${3:-}; CK=runs/$RUN/last.pt; PY=.venv/bin/python
CF=blind_insertion/configs/phase_c_yam.json
E="$PY -m blind_insertion.eval --policy $CK --env_file $CF --episodes 2048"
J() {  # merge BASE under a condition's JSON
  if [ -z "$BASE" ]; then echo "$1"; else python3 -c 'import json,sys; b=json.loads(sys.argv[1]); b.update(json.loads(sys.argv[2])); print(json.dumps(b))' "$BASE" "$1"; fi
}
mkdir -p results/C
want() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }
want base && $E --log_worlds 0,1,2,3 ${BASE:+--env_json "$BASE"} --out results/C/eval_${TAG}_base.parquet
want qnoise && for s in 1e-4 5e-4; do $E --env_json "$(J "{\"q_noise\": $s}")" --out results/C/eval_${TAG}_qnoise$s.parquet; done
want quant && for b in 14 16; do $E --env_json "$(J "{\"q_quant_bits\": $b}")" --out results/C/eval_${TAG}_quant$b.parquet; done
want backlash && for b in 1 3 5; do $E --env_json "$(J "{\"backlash_arcmin\": $b}")" --out results/C/eval_${TAG}_backlash$b.parquet; done
want latency && for k in 1 2; do $E --env_json "$(J "{\"latency\": $k}")" --out results/C/eval_${TAG}_latency$k.parquet; done
want gain && for g in 0.5 2 4; do
  $E --env_json "$(J "{\"gains\": [[$(python3 -c "print(','.join(str(80*$g) for _ in range(3))+','+','.join(str(30*$g) for _ in range(3)))")],[$(python3 -c "print(','.join(str(6*$g) for _ in range(3))+','+','.join(str(2*$g) for _ in range(3)))")]]}")" --out results/C/eval_${TAG}_gain$g.parquet
done
true   # a skipped last factor (ONLY=...) must not fail the job
