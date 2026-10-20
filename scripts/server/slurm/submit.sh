#!/usr/bin/env bash
# Turn a job manifest into a SLURM job array (one command line per array task).
#
# usage: scripts/server/slurm/submit.sh MANIFEST [options]
#   --partition P      sbatch -p
#   --account A        sbatch -A
#   --qos Q            sbatch --qos
#   --gres G           GPU request (default gpu:1; e.g. gpu:a100:1)
#   --cpus N           cpus per task (default 8)
#   --mem M            memory per task (default 48G)
#   --time T           time limit per task (default 06:00:00; use ~12:00:00 for visual jobs
#                      and leave >= 2x the 3090 Ti estimate on slower GPUs)
#   --max-parallel K   at most K tasks at once (array %K; default 16)
#   --after JOBID      start only after JOBID ends (afterany), e.g. the training array
#   --only REGEX       submit only lines matching REGEX (e.g. '-s0 ' or 'visual')
#   --name NAME        job name (default: manifest file stem)
#   --extra "ARGS"     extra sbatch arguments, e.g. "--constraint=a100 --nice=10"
#   --dry-run          write the frozen manifest and print the sbatch command, do not submit
# Lines whose training run already has runs/<name>/.complete are left out, so re-running
# submit.sh on the same manifest only submits what is missing. The frozen copy is
# runs/slurm/manifests/<name>-<time>.txt; task logs are runs/slurm/<name>-<jobid>_<task>.out.
set -euo pipefail
cd "$(dirname "$0")/../../.."
REPO=$PWD

MAN=${1:?usage: $0 MANIFEST [options]}; shift
PART="" ACCT="" QOS="" GRES="gpu:1" CPUS=8 MEM=48G TIME=06:00:00 MAXP=16 AFTER="" ONLY="" NAME="" EXTRA="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --partition) PART=$2; shift ;;
    --account) ACCT=$2; shift ;;
    --qos) QOS=$2; shift ;;
    --gres) GRES=$2; shift ;;
    --cpus) CPUS=$2; shift ;;
    --mem) MEM=$2; shift ;;
    --time) TIME=$2; shift ;;
    --max-parallel) MAXP=$2; shift ;;
    --after) AFTER=$2; shift ;;
    --only) ONLY=$2; shift ;;
    --name) NAME=$2; shift ;;
    --extra) EXTRA=$2; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$MAN" ] || { echo "no manifest $MAN" >&2; exit 2; }
NAME=${NAME:-$(basename "${MAN%.*}")}
mkdir -p runs/slurm/manifests
FROZEN="runs/slurm/manifests/$NAME-$(date +%Y%m%d-%H%M%S).txt"

n_skip=0
: > "$FROZEN"
while IFS= read -r line || [ -n "$line" ]; do
  [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
  [ -n "$ONLY" ] && ! grep -qE -- "$ONLY" <<< "$line" && continue
  cmd="${line%% # *}"
  name=$(grep -oP -- '--name[ =]\K\S+' <<< "$cmd" || true)
  if [ -n "$name" ] && [[ "$cmd" =~ blind_insertion\.train ]] && [ -f "runs/$name/.complete" ]; then
    n_skip=$((n_skip + 1)); continue
  fi
  printf '%s\n' "$line" >> "$FROZEN"
done < "$MAN"
N=$(wc -l < "$FROZEN")
echo "frozen $N job(s) -> $FROZEN ($n_skip already complete)"
[ "$N" -gt 0 ] || { rm -f "$FROZEN"; echo "nothing to submit"; exit 0; }

args=(--job-name "$NAME" --array "1-$N%$MAXP" --gres "$GRES" --cpus-per-task "$CPUS" --mem "$MEM"
      --time "$TIME" --output "runs/slurm/%x-%A_%a.out" --chdir "$REPO")
[ -n "$PART" ] && args+=(--partition "$PART")
[ -n "$ACCT" ] && args+=(--account "$ACCT")
[ -n "$QOS" ] && args+=(--qos "$QOS")
[ -n "$AFTER" ] && args+=(--dependency "afterany:$AFTER")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && args+=($EXTRA)
echo "sbatch ${args[*]} scripts/server/slurm/job_array.sbatch $FROZEN"
if [ "$DRY" = 1 ]; then echo "(dry run: not submitted)"; exit 0; fi
command -v sbatch >/dev/null || { echo "sbatch not found" >&2; exit 1; }
jid=$(sbatch --parsable "${args[@]}" scripts/server/slurm/job_array.sbatch "$FROZEN")
echo "submitted array job $jid ($N tasks). Watch: squeue -j $jid; logs runs/slurm/$NAME-${jid%%;*}_*.out"
