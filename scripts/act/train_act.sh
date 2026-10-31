#!/usr/bin/env bash
# Train LeRobot ACT on the task H pick demos (LeRobot venv).
# usage: CUDA_VISIBLE_DEVICES=3 scripts/act/train_act.sh <dataset_root> <run_dir> [steps] [extra lerobot-train args]
# NPROC=2 CUDA_VISIBLE_DEVICES=1,4: data parallel over 2 GPUs (accelerate; batch_size is per GPU, so halve
# the steps for the same samples). AMP=1: fp16 mixed precision (V100 tensor cores).
cd "$(dirname "$0")/../.."
ROOT=${1:-data/act/reader_pick_v1}; OUT=${2:-runs/H-act-pick-v1}; STEPS=${3:-60000}; shift 3 2>/dev/null
NPROC=${NPROC:-1}
if [ "$NPROC" -gt 1 ]; then
  RUN=(.venv-act/bin/accelerate-launch --multi_gpu --num_processes "$NPROC" --mixed_precision "$([ "${AMP:-0}" = 1 ] && echo fp16 || echo no)" .venv-act/bin/lerobot-train)
else
  RUN=(.venv-act/bin/lerobot-train)
fi
HF_HUB_OFFLINE=1 "${RUN[@]}" \
  --dataset.repo_id=local/$(basename "$ROOT") --dataset.root="$ROOT" \
  --dataset.image_transforms.enable=true \
  --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=10 \
  --policy.device=cuda --policy.push_to_hub=false --policy.use_amp="$([ "${AMP:-0}" = 1 ] && echo true || echo false)" \
  --batch_size=32 --num_workers=8 --steps="$STEPS" --save_freq=10000 --log_freq=200 \
  --output_dir="$OUT" --wandb.enable=false "$@"
