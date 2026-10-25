#!/usr/bin/env bash
# Install the real-arm stack (yam-control: i2rt, python-can, RealSense, ZED) into .venv with uv,
# without touching the locked sim stack. Run after scripts/server/bootstrap.sh.
#
#   bash scripts/real/setup_real.sh
#
# i2rt pins numpy==2.2.6 (the sim stack is locked to numpy 2.5.x) and tyro==0.9.26, so i2rt goes
# in with --no-deps and its other deps are installed with every package already in .venv held at
# its current version. ruckig 0.15.3 has no wheel (sdist build fails): use a binary ruckig<0.15.
# These packages are not in uv.lock: `uv sync` without --inexact removes them; re-run this script.
# The ZED wheel needs the ZED SDK in /usr/local/zed (its major.minor picks the wheel).
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/server/env.sh
I2RT_COMMIT="${I2RT_COMMIT:-4c43932cdb8e74dcb576419a81a76b47b90f2705}"
[ -x "$PY" ] || { echo "no .venv: run scripts/server/bootstrap.sh first" >&2; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$PY" -c "import importlib.metadata as m; [print(f'{d.metadata[\"Name\"]}=={d.version}') for d in m.distributions()]" \
  | grep -viE "^(i2rt|pyzed)==" > "$tmp/constraints.txt"

uv pip install --python "$PY" -c "$tmp/constraints.txt" --only-binary ruckig \
  "click<8.2.0" "dm-env==1.6" mink "portal==3.7.3" pydantic "pygame==2.6.1" "python-can==4.5.0" \
  pyyaml "qpsolvers[quadprog]==4.7.1" "ruckig<0.15" "threadpoolctl==3.6.0" tyro viser \
  opencv-python pyrealsense2
uv pip install --python "$PY" --no-deps "i2rt @ git+https://github.com/i2rt-robotics/i2rt@$I2RT_COMMIT"

if [ -d /usr/local/zed/include ]; then
  zv=$(sed -nE 's/.*#define ZED_SDK_(MAJOR|MINOR)_VERSION ([0-9]+).*/\2/p' /usr/local/zed/include/sl/Camera.hpp | head -2 | paste -sd.)
  pv=$("$PY" -c "import sys; print(f'{sys.version_info[0]}{sys.version_info[1]}')")
  whl="pyzed-$zv-cp$pv-cp$pv-linux_$(uname -m).whl"
  curl -fsSL -o "$tmp/$whl" "https://download.stereolabs.com/zedsdk/$zv/whl/linux_$(uname -m)/$whl"
  uv pip install --python "$PY" -c "$tmp/constraints.txt" "$tmp/$whl"
else
  echo "no ZED SDK in /usr/local/zed: skipping pyzed"
fi

"$PY" -c "import i2rt, can, ruckig, cv2, pyrealsense2, numpy, torch, mujoco; print('real stack ok: numpy', numpy.__version__, 'torch', torch.__version__)"
"$PY" -c "import pyzed.sl" 2>/dev/null && echo "pyzed ok" || true
