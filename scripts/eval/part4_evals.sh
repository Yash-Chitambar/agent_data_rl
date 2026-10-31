#!/usr/bin/env bash
# Part 4 held-out evaluations (seed 1000, n = 1024 per row): Part 2 start errors at scale 1.0 with the
# guarded left hold, the same with the PC pinned (mass x1000), and robustness at scale 1.5.
#   CUDA_VISIBLE_DEVICES=1 bash scripts/eval/part4_evals.sh <label> <checkpoint> <nworld> [tags...]
# tags: hold (default), fixed, s15
set -euo pipefail
label=$1; ckpt=$2; nw=$3; shift 3
tags=${*:-hold fixed s15}
out=results/usbc/policy/insertion
npz=results/usbc/policy/approach/preinsert_error_samples.npz
per=$(( (1024 + nw - 1) / nw ))
export PYTHONPATH=$PWD MUJOCO_GL=egl
for t in $tags; do
  case $t in
    hold)  ej="{\"start_source\": \"$npz\"}"; sc=1.0 ;;
    fixed) ej="{\"start_source\": \"$npz\", \"pc_mass_scale\": 1000.0}"; sc=1.0 ;;
    s15)   ej="{\"start_source\": \"$npz\"}"; sc=1.5 ;;
  esac
  # merge the checkpoint's own env_json (privileged teacher, cameras) with the eval settings
  ej=$(.venv/bin/python - "$ckpt" "$ej" <<'EOF'
import json, sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
base = ck.get('env_json') or {}
base = json.loads(base) if isinstance(base, str) else dict(base)
base.update(json.loads(sys.argv[2]))
print(json.dumps(base))
EOF
)
  .venv/bin/python -m blind_insertion.eval_insertion --policy "$ckpt" --nworld "$nw" --per_world "$per" \
      --seed 1000 --scale "$sc" --env_json "$ej" --out "$out/eval_${label}_${t}" > "runs/logs/eval_${label}_${t}.log" 2>&1
  grep -A3 '"rate"' "runs/logs/eval_${label}_${t}.log" | head -4 | tr -d '\n'; echo "  <- $label $t"
done
