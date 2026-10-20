# Shared environment for server jobs. Source it from the repo root or any script:
#   source scripts/server/env.sh
# Every variable can be overridden by exporting it before sourcing.
_bi_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$_bi_here/../.." && pwd)}"
unset _bi_here

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# CPU MuJoCo renderer (viz/replay, render.py). Training (also visual) uses the MuJoCo Warp
# CUDA ray tracer and does not need EGL.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# uv: keep the interpreter and the wheel cache inside the repo (as on the local machine)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO/.python}"
# Warp kernel cache (Warp appends its version, e.g. .../warp/1.18.0). $HOME/.cache/warp is
# Warp's own default. Kernels are compiled per GPU architecture, so the first run on a new GPU
# type compiles. On SLURM, fill it once (bootstrap smoke run) before submitting an array, or
# point it at a shared or node-local path.
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$HOME/.cache/warp}"
# Each job is one Python process driving one GPU; avoid oversubscribing CPU threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PATH="$HOME/.local/bin:$PATH"
PY="$REPO/.venv/bin/python"
export PY
