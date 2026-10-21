#!/usr/bin/env bash
# One manifest line per SLURM array task. Submit with scripts/server/slurm/submit.sh, which
# freezes the manifest and fills in partition/account/GPU/CPU/mem/time; the values below are
# only defaults.
#SBATCH --job-name=bi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=runs/slurm/%x-%A_%a.out
#
# Task i runs the i-th command line (blank and '#' lines do not count) of the frozen manifest
# given as $1. It
#   - skips (exit 0) if the line trains run X and runs/X/.complete exists;
#   - fails (exit 3) if an `after=` dependency has no runs/<dep>/.complete (submit eval arrays
#     with `submit.sh ... --after <train array job id>`);
#   - resumes a partial runs/X (no .complete, e.g. a preempted attempt) by appending --resume when
#     runs/X/last.pt exists, else moves it to runs/X.failed-<time>;
#   - writes runs/X/.complete (JSON) when a training command exits 0.
# Preemption: with --requeue, SLURM requeues the task if the cluster preempts it (PreemptMode=
# REQUEUE); it then resumes from runs/X/last.pt (checkpoints every 25 iterations, ~10M steps).
# The USR1 sent 300 s before the time limit only logs a warning unless BI_REQUEUE_ON_TIMEOUT=1
# (set it on clusters with short time limits: each requeue continues from the last checkpoint).
set -uo pipefail
MANIFEST=${1:?usage: sbatch job_array.sbatch FROZEN_MANIFEST}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
# shellcheck source=../env.sh
source scripts/server/env.sh
cd "$REPO"
TID=${SLURM_ARRAY_TASK_ID:?not an array task}
line=$(grep -v '^[[:space:]]*#' "$MANIFEST" | grep -v '^[[:space:]]*$' | sed -n "${TID}p")
[ -n "$line" ] || { echo "no line $TID in $MANIFEST"; exit 2; }
cmd="${line%% # *}"
ann=""; [[ "$line" == *" # "* ]] && ann="${line#* # }"
name=$(grep -oP -- '--name[ =]\K\S+' <<< "$cmd" || true)
is_train=0
[[ -n "$name" && "$cmd" =~ blind_insertion\.train(_visual)?[[:space:]] ]] && is_train=1
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] task ${SLURM_ARRAY_JOB_ID:-?}_$TID on $(hostname) restart=${SLURM_RESTART_COUNT:-0}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "  cmd: $cmd"
[ -n "$ann" ] && echo "  annotations: $ann"

if [ "$is_train" = 1 ] && [ -f "runs/$name/.complete" ]; then
  echo "[$(ts)] runs/$name/.complete exists: skip"; exit 0
fi
for tok in $ann; do
  if [[ "$tok" == after=* ]]; then
    IFS=',' read -ra deps <<< "${tok#after=}"
    for d in "${deps[@]}"; do
      [ -f "runs/$d/.complete" ] || { echo "[$(ts)] dependency runs/$d/.complete missing: not running"; exit 3; }
    done
  fi
done
if [ "$is_train" = 1 ] && [ -f "runs/$name/last.pt" ]; then
  echo "[$(ts)] partial runs/$name has last.pt: resuming"
  cmd=$("$PY" -c 'import sys; sys.path.insert(0, "scripts/server"); from run_jobs import with_resume; print(with_resume(sys.argv[1]))' "$cmd")
elif [ "$is_train" = 1 ] && [ -d "runs/$name" ]; then
  bak="runs/$name.failed-$(date +%Y%m%d-%H%M%S)"
  echo "[$(ts)] partial runs/$name (no .complete, no last.pt) -> $bak"; mv "runs/$name" "$bak"
fi

child=""
on_usr1() {
  echo "[$(ts)] USR1: time limit in <= 300 s"
  if [ "${BI_REQUEUE_ON_TIMEOUT:-0}" = 1 ]; then
    echo "[$(ts)] requeueing ${SLURM_ARRAY_JOB_ID}_$TID"
    [ -n "$child" ] && kill -TERM "$child" 2>/dev/null
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_$TID"
  fi
}
trap on_usr1 USR1
trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM

t0=$SECONDS
bash -c "$cmd" &
child=$!
# `wait` returns early when a trapped signal arrives; keep waiting while the job runs
while :; do
  wait "$child"; rc=$?
  kill -0 "$child" 2>/dev/null || break
done
mins=$(( (SECONDS - t0) / 60 ))
echo "[$(ts)] exit rc=$rc after $mins min"
if [ "$rc" = 0 ] && [ "$is_train" = 1 ]; then
  printf '{"cmd": %s, "rc": 0, "host": "%s", "slurm": "%s", "minutes": %d, "end": "%s", "by": "slurm"}\n' \
    "$("$PY" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$cmd")" "$(hostname)" \
    "${SLURM_ARRAY_JOB_ID:-}_$TID" "$mins" "$(ts)" > "runs/$name/.complete.tmp" \
    && mv "runs/$name/.complete.tmp" "runs/$name/.complete"
fi
exit "$rc"
