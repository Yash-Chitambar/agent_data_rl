#!/usr/bin/env bash
# Train LeRobot ACT on the task H pick demos (LeRobot venv, one GPU).
# usage: CUDA_VISIBLE_DEVICES=3 scripts/act/train_act.sh <dataset_root> <run_dir> [steps] [extra lerobot-train args]
cd "$(dirname "$0")/../.."
ROOT=${1:-data/act/reader_pick_v1}; OUT=${2:-runs/H-act-pick-v1}; STEPS=${3:-60000}; shift 3 2>/dev/null
HF_HUB_OFFLINE=1 .venv-act/bin/lerobot-train \
  --dataset.repo_id=local/$(basename "$ROOT") --dataset.root="$ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=10 \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=32 --num_workers=8 --steps="$STEPS" --save_freq=10000 --log_freq=200 \
  --output_dir="$OUT" --wandb.enable=false "$@"
