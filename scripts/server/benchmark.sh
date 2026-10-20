#!/usr/bin/env bash
# Measure training speed and peak GPU memory on ONE named GPU, before you queue long runs.
#
#   bash scripts/server/benchmark.sh --gpu 3 [--nworld 12288] [--iters 12] [--visual]
#
# Runs real training (blind: blind_insertion.train; --visual: blind_insertion.train_visual) for
# a few PPO iterations into runs/_bench-*, samples this job's GPU memory every 0.5 s with
# nvidia-smi, and writes runs/_bench/<host>_<gpu-name>_<kind>_<nworld>.json:
#   env steps/s (median of the logged iterations after the first), peak GPU memory (MiB),
#   GPU total memory, time to the first iteration (includes Warp kernel compile on a new
#   GPU type and the model build), exit status.
# --gpu is required: on a shared server, choose a FREE GPU with the capacity check first.
# The run directory is deleted afterwards (keep it with --keep).
set -uo pipefail
GPU="" NWORLD="" ITERS=12 VISUAL=0 KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=$2; shift ;;
    --nworld) NWORLD=$2; shift ;;
    --iters) ITERS=$2; shift ;;
    --visual) VISUAL=1 ;;
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$GPU" ] || { echo "--gpu N is required (pick a FREE GPU first)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
# shellcheck source=scripts/server/env.sh
source scripts/server/env.sh
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

if [ "$VISUAL" = 1 ]; then
  KIND=visual; NWORLD=${NWORLD:-4096}; MOD=blind_insertion.train_visual
else
  KIND=blind; NWORLD=${NWORLD:-12288}; MOD=blind_insertion.train
fi
NAME="_bench-$KIND-$NWORLD"
STEPS=$((ITERS * 32 * NWORLD))
mkdir -p runs/_bench
rm -rf "runs/$NAME"
LOG="runs/_bench/$NAME.log"
MEMLOG="runs/_bench/$NAME.mem"
: > "$MEMLOG"

t0=$(date +%s.%N)
"$PY" -m "$MOD" --name "$NAME" --nworld "$NWORLD" --steps "$STEPS" --print_every 1 \
  --ckpt_every 100000 > "$LOG" 2>&1 &
PID=$!
# sample this process's GPU memory (nvidia-smi lists our own processes)
( while kill -0 "$PID" 2>/dev/null; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' -v p="$PID" '$1==p {print $2}' >> "$MEMLOG"
    sleep 0.5
  done ) &
SAMPLER=$!
wait "$PID"; RC=$?
wait "$SAMPLER" 2>/dev/null
t1=$(date +%s.%N)

"$PY" - "$LOG" "$MEMLOG" "$KIND" "$NWORLD" "$GPU" "$RC" "$t0" "$t1" <<'EOF'
import json, re, socket, statistics, subprocess, sys
log, memlog, kind, nworld, gpu, rc, t0, t1 = sys.argv[1:]
text = open(log, errors="replace").read()
sps = [float(m) for m in re.findall(r"\bsps (\d+(?:\.\d+)?)", text)]
mem = [int(x) for x in open(memlog).read().split() if x.isdigit()]
q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
name, total, drv, cc = [s.strip() for s in q.split(",")] if q else ("?", "0", "?", "?")
import importlib.metadata as md
res = dict(host=socket.gethostname(), gpu_index=gpu, gpu=name, gpu_mem_total_MiB=int(total), driver=drv,
           compute_cap=cc, torch=md.version("torch"), warp=md.version("warp-lang"), kind=kind,
           nworld=int(nworld), iterations_logged=len(sps),
           sps_median=statistics.median(sps[1:]) if len(sps) > 1 else (sps[0] if sps else None),
           sps_all=sps, peak_mem_MiB=max(mem) if mem else None, wall_s=round(float(t1) - float(t0), 1),
           exit_code=int(rc), oom=("out of memory" in text.lower()))
out = f"runs/_bench/{res['host']}_{name.replace(' ', '-')}_{kind}_{nworld}.json"
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("host", "gpu", "gpu_mem_total_MiB", "kind", "nworld", "sps_median",
                                      "peak_mem_MiB", "wall_s", "exit_code", "oom")}))
if res["sps_median"]:
    print(f"300M steps would take {300e6 / res['sps_median'] / 3600:.2f} h at this speed "
          f"(local RTX 3090 Ti: blind ~37.5k steps/s at 12,288 worlds, visual ~8.5k at 4,096)")
print("wrote", out)
EOF
[ "$KEEP" = 1 ] || rm -rf "runs/$NAME"
[ "$RC" = 0 ] || { echo "benchmark run exited $RC; see $LOG" >&2; exit "$RC"; }
