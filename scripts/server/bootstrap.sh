#!/usr/bin/env bash
# Idempotent setup + verification of this repo on a fresh Linux GPU machine.
#
#   git clone --recurse-submodules https://github.com/Yash-Chitambar/peg_test.git
#   cd peg_test && bash scripts/server/bootstrap.sh
#
# Steps (each is skipped when already done):
#   1. uv (pinned version) into ~/.local/bin if missing
#   2. git submodules (mjfit, robosimstudio) at the commits pinned by the repo
#   3. Python 3.12 + `uv sync --locked --inexact --group train` into .venv
#   4. MuJoCo Menagerie sparse clone (flexiv_rizon4s, robotiq_2f85, i2rt_yam) at a pinned commit
#   5. verify.py: versions, torch + Warp see the GPU, EGL render, MuJoCo Warp step
#   6. smoke runs (~1-2 min on a warm kernel cache; the first run on a new GPU type also
#      compiles Warp kernels, which can take several minutes): train.py (256 worlds, 2 PPO
#      iterations), eval.py on that checkpoint (64 episodes), train_visual.py (128 worlds,
#      2 iterations). They write to runs/_smoke-*; delete them afterwards if you like.
#
# Options:
#   --skip-install   skip steps 1-4 (only verify + smoke)
#   --skip-smoke     skip step 6
#   --no-egl         EGL failure is a warning (EGL is only needed for viz/replay, not training)
#   --no-visual      skip the train_visual smoke run
# Env overrides: UV_VERSION, MENAGERIE_COMMIT, SMOKE_TIMEOUT (s per smoke step, default 900).
# Exit status is non-zero if any step fails.
set -euo pipefail

UV_VERSION="${UV_VERSION:-0.12.18}"
MENAGERIE_URL="${MENAGERIE_URL:-https://github.com/google-deepmind/mujoco_menagerie.git}"
MENAGERIE_COMMIT="${MENAGERIE_COMMIT:-cda5236f98383f6fbaa4820c6b18dba892cd087f}"
MENAGERIE_ROBOTS=(flexiv_rizon4s robotiq_2f85 i2rt_yam)
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-900}"

SKIP_INSTALL=0 SKIP_SMOKE=0 NO_EGL=0 NO_VISUAL=0
for a in "$@"; do
  case "$a" in
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    --no-egl) NO_EGL=1 ;;
    --no-visual) NO_VISUAL=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/../.."
# shellcheck source=scripts/server/env.sh
source scripts/server/env.sh
cd "$REPO"
log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }
die() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------ 0. system sanity
command -v git >/dev/null || die "git not found"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no NVIDIA driver?)"
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
log "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr '\n' ';') driver $drv"
if [ "${drv%%.*}" -lt 580 ]; then
  die "driver $drv is too old: torch cu130 / Warp 1.18 (CUDA 13) need driver >= 580"
fi
free_gb=$(df -Pk . | awk 'NR==2{print int($4/1024/1024)}')
[ "$free_gb" -ge 20 ] || log "WARNING: only ${free_gb} GB free here; .venv + uv cache need ~15 GB"

if [ "$SKIP_INSTALL" = 0 ]; then
  # ---------------------------------------------------------------- 1. uv
  if ! command -v uv >/dev/null; then
    command -v curl >/dev/null || die "curl not found (needed to install uv)"
    log "installing uv $UV_VERSION into ~/.local/bin"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
  fi
  log "uv: $(uv --version)"

  # ---------------------------------------------------------------- 2. submodules
  if [ -e .git ]; then
    log "submodules"
    git submodule sync --recursive >/dev/null
    git submodule update --init --recursive
  else
    [ -f mjfit/pyproject.toml ] && [ -f robosimstudio/pyproject.toml ] \
      || die "not a git checkout and mjfit/ or robosimstudio/ missing (use git clone --recurse-submodules)"
    log "no .git: using mjfit/ and robosimstudio/ as present"
  fi

  # ---------------------------------------------------------------- 3. Python env
  log "uv sync (Python 3.12, locked, train group)"
  uv python install 3.12
  # --inexact: never remove extra packages (e.g. the dev group) from an existing .venv
  uv sync --locked --inexact --group train

  # ---------------------------------------------------------------- 4. Menagerie
  M=third_party/mujoco_menagerie
  if [ ! -d "$M/.git" ]; then
    log "Menagerie sparse clone -> $M"
    mkdir -p third_party
    git clone --filter=blob:none --no-checkout "$MENAGERIE_URL" "$M"
    git -C "$M" sparse-checkout init --cone
    git -C "$M" sparse-checkout set "${MENAGERIE_ROBOTS[@]}"
    git -C "$M" checkout --quiet "$MENAGERIE_COMMIT"
  else
    have=$(git -C "$M" sparse-checkout list 2>/dev/null | sort | tr '\n' ' ')
    want=$(printf '%s\n' "${MENAGERIE_ROBOTS[@]}" | sort | tr '\n' ' ')
    if [ "$have" != "$want" ]; then
      log "Menagerie sparse set '$have' -> '$want'"
      git -C "$M" sparse-checkout set "${MENAGERIE_ROBOTS[@]}"
    fi
    head=$(git -C "$M" rev-parse HEAD)
    if [ "$head" != "$MENAGERIE_COMMIT" ]; then
      log "Menagerie at $head, checking out pinned $MENAGERIE_COMMIT"
      git -C "$M" fetch --quiet --filter=blob:none origin "$MENAGERIE_COMMIT" || true
      git -C "$M" checkout --quiet "$MENAGERIE_COMMIT"
    fi
    log "Menagerie ok ($(git -C "$M" rev-parse --short HEAD): $want)"
  fi
  for r in "${MENAGERIE_ROBOTS[@]}"; do [ -d "$M/$r" ] || die "Menagerie robot $r missing"; done
fi

[ -x "$PY" ] || die ".venv/bin/python missing (run without --skip-install)"

# ------------------------------------------------------------------ 5. verification
log "verify.py"
vargs=()
[ "$NO_EGL" = 1 ] && vargs+=(--no-egl)
mkdir -p runs
"$PY" scripts/server/verify.py "${vargs[@]}" --json runs/_verify.json 2>&1 \
  | grep -v "^Module .* load on device" || die "verification failed (see above)"
# (grep -v masks verify.py's exit code; re-check from the JSON)
"$PY" - <<'EOF' || die "verification failed (runs/_verify.json)"
import json, sys
r = json.load(open("runs/_verify.json"))
sys.exit(any(v["required"] and not v["ok"] for v in r.values()))
EOF

# ------------------------------------------------------------------ 6. smoke runs
if [ "$SKIP_SMOKE" = 0 ]; then
  smoke() {  # name, timeout, command...
    local name=$1 to=$2; shift 2
    local t0=$SECONDS
    log "smoke: $name"
    if ! timeout "$to" "$@" > "runs/$name.log" 2>&1; then
      tail -30 "runs/$name.log" >&2
      die "smoke $name failed or timed out after $((SECONDS - t0)) s (log runs/$name.log)"
    fi
    log "smoke: $name ok in $((SECONDS - t0)) s"
  }
  # 2 PPO iterations = 2 * 32 steps * nworld env steps
  rm -rf runs/_smoke-train runs/_smoke-visual
  smoke _smoke-train "$SMOKE_TIMEOUT" "$PY" -m blind_insertion.train --name _smoke-train \
    --nworld 256 --n_variants 16 --steps $((2 * 32 * 256)) --ckpt_every 1 --print_every 1
  for f in config.json train_log.csv last.pt; do
    [ -s "runs/_smoke-train/$f" ] || die "smoke train did not write runs/_smoke-train/$f"
  done
  [ "$(wc -l < runs/_smoke-train/train_log.csv)" -ge 3 ] || die "smoke train_log.csv has < 2 rows"
  smoke _smoke-eval "$SMOKE_TIMEOUT" "$PY" -m blind_insertion.eval --policy runs/_smoke-train/last.pt \
    --episodes 64 --n_variants 16 --out runs/_smoke-train/eval.parquet
  [ -s runs/_smoke-train/eval.parquet ] || die "smoke eval wrote no parquet"
  if [ "$NO_VISUAL" = 0 ]; then
    smoke _smoke-visual "$SMOKE_TIMEOUT" "$PY" -m blind_insertion.train_visual --name _smoke-visual \
      --nworld 128 --n_variants 16 --steps $((2 * 32 * 128)) --ckpt_every 1 --print_every 1
    [ -s runs/_smoke-visual/last.pt ] || die "smoke visual did not write last.pt"
  fi
  grep -h "sps" runs/_smoke-train.log runs/_smoke-visual.log 2>/dev/null | tail -4 || true
fi

log "ALL OK. Next: docs/SERVER.md (run_jobs.py on a multi-GPU box, slurm/submit.sh on SLURM)"
