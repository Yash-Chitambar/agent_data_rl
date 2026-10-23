"""Phase F: what the YAM wrist camera (Menagerie D405, `wrist_cam`) sees at the USB-C start states.
Top row: RGB with the render randomization of phase_f_usbc.json; bottom row: depth after the
D405-like sensor model (black = invalid: closer than 7 cm, dropout, edges). 64 x 64 as in training.

python scripts/report/f_camera_views.py -> results/F/camera_views.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.envs.peg_env import EnvCfg, PegEnv  # noqa: E402

cfg = json.load(open(ROOT / "blind_insertion/configs/phase_f_usbc.json"))
env = PegEnv(EnvCfg(**dict(cfg, nworld=16, variants=B.sample_usbc(16, 1000), camera=True, cam_modality="rgbd",
                           peg_offset=3e-3 / 0.765)))
env.render()
fr = env.frame.cpu().numpy()
d = env.depth_img.cpu().numpy()
n = 6
fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 5.0))
for i in range(n):
    ax[0, i].imshow(fr[i, :3].transpose(1, 2, 0), interpolation="nearest")
    m = np.ma.masked_equal(d[i], 0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    im = ax[1, i].imshow(m * 100, cmap=cmap, vmin=6, vmax=20, interpolation="nearest")
    for a in ax[:, i]:
        a.set_xticks([]), a.set_yticks([])
ax[0, 0].set_ylabel("RGB", fontsize=11)
ax[1, 0].set_ylabel("depth (cm)", fontsize=11)
fig.colorbar(im, ax=ax[1, :].tolist(), shrink=0.8, label="depth (cm); black = invalid")
valid = (d > 0).mean()
fig.suptitle(f"YAM wrist D405 at USB-C start states, 64 x 64 (valid depth {100 * valid:.0f}%, "
             f"range {100 * d[d > 0].min():.1f}-{100 * d.max():.1f} cm)", fontsize=11)
out = ROOT / "results/F/camera_views.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out, "valid", valid)
