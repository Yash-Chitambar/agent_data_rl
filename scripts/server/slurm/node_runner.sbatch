#!/usr/bin/env bash
# Alternative to job arrays: take whole multi-GPU node(s) for a long slot and run
# scripts/server/run_jobs.py inside, which packs jobs onto the allocated GPUs (and can run
# several per GPU). Useful where the cluster prefers few long allocations, or to pack
# small eval jobs.
#   sbatch --gres=gpu:8 --cpus-per-task=64 --mem=400G --time=48:00:00 -p PART \
#          scripts/server/slurm/node_runner.sbatch scripts/server/jobs/D_train_tier1.txt [--per-gpu 2]
# Everything after the sbatch script name is passed to run_jobs.py. State and the done list
# live in runs/queue_server/ (shared with plain-box use), so a later allocation continues
# where this one stopped (unfinished jobs restart from scratch: no resume yet).
#SBATCH --job-name=bi-node
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
source scripts/server/env.sh
cd "$REPO"
echo "[$(date)] $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# SLURM sends SIGTERM at the time limit: run_jobs.py then stops its jobs cleanly
exec "$PY" scripts/server/run_jobs.py --gpus auto "$@"
