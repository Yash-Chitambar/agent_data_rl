"""Visual check of the Phase E assemblies (E4 chamfer, E5 latch, E6 side port): one start state
per config from the GPU env, copied to CPU MuJoCo and rendered from a fixed side camera.

python scripts/report/usbc_variant_views.py --out results/E/variants/views.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.envs.peg_env import EnvCfg, PegEnv  # noqa: E402

CF = ROOT / "blind_insertion/configs"


def start_image(cfg: dict, variants, w=0, size=(480, 360), dist=0.12, az=150, el=-20):
    env = PegEnv(EnvCfg(**dict(cfg, nworld=4, variants=variants, auto_reset=False)))
    m, lay = env.mjm, env.lay
    B.apply_variant_cpu(m, lay, int(env.variant[w]))
    d = mujoco.MjData(m)
    d.qpos[:] = env.t_qpos[w].cpu().numpy()
    d.mocap_pos[:] = env.t_mocap_pos[w].cpu().numpy()
    d.mocap_quat[:] = env.t_mocap_quat[w].cpu().numpy()
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.mocap_pos[0]
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    r = mujoco.Renderer(m, size[1], size[0])
    r.update_scene(d, cam)
    img = r.render()
    r.close()
    return img


def chamfer_closeup(chamfer=4e-4, size=(480, 360)):
    v = B.usbc_variant(chamfer=chamfer)
    m, lay = B.build("yam", [v])
    B.apply_variant_cpu(m, lay, 0)
    d = mujoco.MjData(m)
    d.mocap_pos[0] = (0.35, 0.0, 0.0)
    d.qpos[lay.peg_qadr:lay.peg_qadr + 3] = (0.35, 0.0, 0.5)   # drive out of view
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.mocap_pos[0]
    cam.distance, cam.azimuth, cam.elevation = 0.03, 120, -40
    r = mujoco.Renderer(m, size[1], size[0])
    r.update_scene(d, cam)
    img = r.render()
    r.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E/variants/views.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vs = B.sample_usbc(4, 1000, chamfer=(2e-4, 5e-4))
    sph = json.load(open(CF / "phase_e_usbc_spheres.json"))
    side = json.load(open(CF / "phase_e_usbc_side.json"))
    panels = [("E4: chamfered port (0.4 mm lead-in)", chamfer_closeup()),
              ("E1/E4: start state, port up", start_image(sph, vs)),
              ("E6: start state, port facing the robot (side view)", start_image(side, B.sample_usbc(4, 1000), az=90, el=-10, dist=0.15))]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for x, (t, im) in zip(ax, panels):
        x.imshow(im)
        x.set_title(t)
        x.axis("off")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
