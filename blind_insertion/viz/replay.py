"""Render logged rollouts (from eval --log_worlds) with CPU MuJoCo: video + contact sheet + plots.

Qualitative only. python -m blind_insertion.viz.replay --rollouts X.rollouts.npz --world 3 \
    --variant_seed 1000 --n_variants 1024 --out results/A/viz/spiral_w3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.viz.render import save_grid, save_video  # noqa: E402


def render_rollout(m, lay, variant_k, roll, out: Path, *, title="", every=1, size=(480, 360),
                   azimuth=150, elevation=-20, distance=None):
    out.parent.mkdir(parents=True, exist_ok=True)
    B.apply_variant_cpu(m, lay, variant_k)
    vis = m.vis
    vis.scale.contactwidth, vis.scale.contactheight = 0.004, 0.001
    vis.scale.forcewidth, vis.map.force = 0.002, 0.004
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    v = lay.variants[variant_k]
    frames = []
    T = len(roll["qpos"])
    for t in range(0, T, every):
        d.qpos[:] = roll["qpos"][t]
        d.mocap_pos[0] = roll["mocap_pos"][t]
        d.mocap_quat[0] = roll["mocap_quat"][t]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = roll["mocap_pos"][t]
        cam.distance = distance or max(0.25, 2.2 * v.plate_size)
        cam.azimuth, cam.elevation = azimuth, elevation
        r.update_scene(d, cam, opt)
        frames.append(r.render().copy())
    r.close()
    save_video(frames, str(out) + ".mp4", fps=max(1, int(50 / every)))
    pick = np.linspace(0, len(frames) - 1, 8).astype(int)
    save_grid([frames[i] for i in pick], str(out) + "_sheet.png", ncol=4)
    # top view path + descent
    p = roll["peg_p"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hole = lay.hole_polys[variant_k]
    axes[0].fill(1000 * hole[:, 0], 1000 * hole[:, 1], color="#dddddd", label="opening")
    axes[0].plot(1000 * p[:, 0], 1000 * p[:, 1], color="#4477AA", lw=1.5, label="peg centre")
    axes[0].plot(1000 * p[0, 0], 1000 * p[0, 1], "o", color="#EE6677", label="start")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].legend(fontsize=7)
    t = np.arange(len(p)) * 0.02
    axes[1].plot(t, 1000 * p[:, 2], color="#4477AA", label="peg bottom centre z")
    axes[1].axhline(-1000 * v.depth, color="k", ls="--", lw=1, label="hole floor")
    lab = roll["labels"]
    axes[1].plot(t, 1000 * (-v.depth) * (lab == 2).mean(1), color="#228833", lw=1, alpha=0.7,
                 label="fraction inside (scaled)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("mm")
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(out) + "_path.png", dpi=110)
    plt.close(fig)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--robot", default="flexiv")
    ap.add_argument("--n_variants", type=int, default=1024)
    ap.add_argument("--variant_seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    vs = B.sample_study_pegs(args.n_variants, args.variant_seed)
    m, lay = B.build(args.robot, vs)
    z = np.load(args.rollouts)
    for w in [int(x) for x in args.worlds.split(",")]:
        roll = {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(f"w{w}_")}
        render_rollout(m, lay, w % args.n_variants, roll, Path(f"{args.out}_w{w}"), every=args.every,
                       title=f"world {w}")
        print("rendered world", w)


if __name__ == "__main__":
    main()
