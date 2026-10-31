#!/usr/bin/env bash
# Push code (and optionally finished runs) from this machine to a server with rsync.
#
# The primary way to get code onto a server is `git clone --recurse-submodules` (docs/operations/SERVER.md).
# Use this script for (a) uncommitted work-in-progress code, (b) a machine without GitHub
# access, or (c) copying finished local runs (checkpoints) that server jobs need.
#
# usage: scripts/server/sync_up.sh [-n] [--code|--no-code] [--runs NAME[,NAME...]] [--ckpts] DEST
#   DEST       user@host:/path/to/peg_test   (or a local path; or set SYNC_DEST)
#   -n         dry run (rsync -n, prints what would be sent)
#   --no-code  skip the code push (e.g. with --runs only)
#   --runs     also push runs/<NAME>/{config.json,train_log.csv,last.pt,best.pt,curriculum.json}
#              (and runs/<NAME>.log); NAME may be a glob such as 'B-peg-*'
#   --ckpts    with --runs, also push intermediate ckpt_*.pt
# Env: SYNC_SSH (default "ssh"), e.g. SYNC_SSH="ssh -p 2222 -i ~/.ssh/id_server".
# Never deletes anything on the destination. Excludes .git, .venv, .cache, .python, runs/,
# third_party/ and files over 50 MB.
set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=() CODE=1 RUNS="" CKPTS=0 DEST="${SYNC_DEST:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=(-n) ;;
    --code) CODE=1 ;;
    --no-code) CODE=0 ;;
    --runs) RUNS="$2"; shift ;;
    --ckpts) CKPTS=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) DEST="$1" ;;
  esac
  shift
done
[ -n "$DEST" ] || { echo "usage: $0 [-n] [--no-code] [--runs NAMES] [--ckpts] user@host:/path" >&2; exit 2; }
DEST="${DEST%/}"
RS=(rsync -az --info=stats1,name1 --human-readable -e "${SYNC_SSH:-ssh}" "${DRY[@]}")

if [ "$CODE" = 1 ]; then
  echo "== code -> $DEST/"
  "${RS[@]}" --max-size=50m \
    --exclude='.git' --exclude='.git/' \
    --exclude='/.venv/' --exclude='/.cache/' --exclude='/.python/' \
    --exclude='/runs/' --exclude='/third_party/' --exclude='/outputs/' \
    --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' --exclude='*.egg-info/' \
    --exclude='/.agents/' --exclude='/.aws/' --exclude='/.codex/' --exclude='/.claude/' \
    --exclude='MUJOCO_LOG.TXT' \
    ./ "$DEST/"
fi

if [ -n "$RUNS" ]; then
  inc=()
  IFS=',' read -ra names <<< "$RUNS"
  for n in "${names[@]}"; do
    inc+=(--include="/$n/" --include="/$n.log")
    for f in config.json train_log.csv last.pt best.pt curriculum.json .complete; do
      inc+=(--include="/$n/$f")
    done
    [ "$CKPTS" = 1 ] && inc+=(--include="/$n/ckpt_*.pt")
  done
  echo "== runs ($RUNS) -> $DEST/runs/"
  "${RS[@]}" --prune-empty-dirs "${inc[@]}" --exclude='*' ./runs/ "$DEST/runs/"
fi
