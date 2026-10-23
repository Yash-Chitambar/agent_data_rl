#!/usr/bin/env bash
# USB-C stage-1 plateau diagnosis: evaluate one checkpoint on 2,048 matched starts at the
# curriculum's stage-1 conditions (clearance 0.3 mm, median start offset 0.5 mm), changing ONE
# factor per condition. Output: results/E/fail/stage1_<cond>.parquet
# usage: scripts/eval/usbc_stage1_diag.sh [checkpoint] [cond ...]
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
CK=${1:-runs/E-usbc-proprio-curr-s0/last.pt}; shift || true
PY=.venv/bin/python
OUT=results/E/fail
mkdir -p $OUT
base() {  # name, extra args...
  local name=$1; shift
  [ -s "$OUT/stage1_$name.parquet" ] && { echo "skip $name"; return; }
  echo "[$(date +%H:%M:%S)] $name: $*"
  $PY -m blind_insertion.eval --policy "$CK" --env_file blind_insertion/configs/phase_e_usbc_up.json \
    --family usbc --n_variants 320 --episodes 2048 "$@" --out "$OUT/stage1_$name.parquet" 2>&1 \
    | grep -v "^Module" | tail -1
}
S1=(--clearance 3e-4 --peg_offset 6.54e-4)   # median radial 0.5 mm = 0.765 x half-width
conds=${*:-"notilt_hand notilt_all nooffset clear1mm softpads long16s"}
for c in $conds; do
  case $c in
    notilt_hand) base $c "${S1[@]}" --env_json '{"inhand_tilt_deg": 0.0}' ;;
    notilt_all)  base $c "${S1[@]}" --env_json '{"inhand_tilt_deg": 0.0, "grip_rot_deg": 0.0, "fix_tilt_deg": 0.0}' ;;
    nooffset)    base $c --clearance 3e-4 --peg_offset 0.0 ;;
    clear1mm)    base $c --clearance 1e-3 --peg_offset 6.54e-4 ;;
    softpads)    base $c "${S1[@]}" --env_json '{"pad_timeconst": 0.004, "pad_solimp": null}' ;;
    long16s)     base $c "${S1[@]}" --env_json '{"episode_s": 16.0}' ;;
    *) echo "unknown condition $c" >&2 ;;
  esac
done
