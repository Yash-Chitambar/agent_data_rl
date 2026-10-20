#!/usr/bin/env bash
# Pull training outputs and results back from a server with rsync.
#
# usage: scripts/server/sync_down.sh [-n] [--runs GLOB[,GLOB...]] [--ckpts] [--no-results] [--no-logs] SRC
#   SRC          user@host:/path/to/peg_test   (or a local path; or set SYNC_SRC)
#   -n           dry run
#   --runs GLOB  which run dirs to pull (default '*'; e.g. 'C-*,D-peg-*-o8_*')
#   --ckpts      also pull intermediate ckpt_*.pt (skipped by default)
#   --no-results do not pull results/
#   --no-logs    do not pull runs/*.log, runs/queue_server/, runs/slurm/
# Pulled per run: config.json, train_log.csv, last.pt, best.pt, curriculum.json, .complete
# (+ ckpt_*.pt with --ckpts). Run dirs starting with '_' (smoke tests) and '*.failed-*'
# (partial attempts) are skipped. Files that are newer locally are never overwritten
# (rsync --update) and nothing local is deleted.
# Env: SYNC_SSH (default "ssh").
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=() RUNS="*" CKPTS=0 RESULTS=1 LOGS=1 SRC="${SYNC_SRC:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=(-n) ;;
    --runs) RUNS="$2"; shift ;;
    --ckpts) CKPTS=1 ;;
    --no-results) RESULTS=0 ;;
    --no-logs) LOGS=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) SRC="$1" ;;
  esac
  shift
done
[ -n "$SRC" ] || { echo "usage: $0 [-n] [--runs GLOB] [--ckpts] [--no-results] user@host:/path" >&2; exit 2; }
SRC="${SRC%/}"
RS=(rsync -az --update --info=stats1,name1 --human-readable -e "${SYNC_SSH:-ssh}" "${DRY[@]}")
mkdir -p runs results

inc=(--exclude='/_*' --exclude='/*.failed-*/')
IFS=',' read -ra globs <<< "$RUNS"
for g in "${globs[@]}"; do
  inc+=(--include="/$g/")
  for f in config.json train_log.csv last.pt best.pt curriculum.json .complete stdout.log; do
    inc+=(--include="/$g/$f")
  done
  [ "$CKPTS" = 1 ] && inc+=(--include="/$g/ckpt_*.pt")
done
if [ "$LOGS" = 1 ]; then
  inc+=(--include='/*.log' --include='/queue_server/' --include='/queue_server/*.txt'
        --include='/queue_server/*.log' --include='/queue_server/jobs/' --include='/queue_server/jobs/*.log'
        --include='/slurm/' --include='/slurm/***')
fi
echo "== $SRC/runs/ -> runs/ (runs: $RUNS, ckpts: $CKPTS, logs: $LOGS)"
"${RS[@]}" --prune-empty-dirs "${inc[@]}" --exclude='*' "$SRC/runs/" ./runs/

if [ "$RESULTS" = 1 ]; then
  echo "== $SRC/results/ -> results/"
  "${RS[@]}" "$SRC/results/" ./results/
fi
