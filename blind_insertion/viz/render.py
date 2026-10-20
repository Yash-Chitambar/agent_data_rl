"""CPU MuJoCo rendering of single worlds of the batched env (qualitative checks only)."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from blind_insertion.assets import builder as B  # noqa: E402


class WorldRenderer:
    def __init__(self, env, width=640, height=480):
        self.env = env
        self.m = env.mjm
        self.d = mujoco.MjData(self.m)
        vis = self.m.vis
        vis.scale.contactwidth, vis.scale.contactheight = 0.004, 0.001
        vis.scale.forcewidth, vis.map.force = 0.002, 0.004
        self.r = mujoco.Renderer(self.m, height, width)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def load(self, w: int, qpos: np.ndarray | None = None):
        env, m = self.env, self.m
        k = int(env.variant[w])
        B.apply_variant_cpu(m, env.lay, k)
        self.d.mocap_pos[0] = env.t_mocap_pos[w, 0].cpu().numpy()
        self.d.mocap_quat[0] = env.t_mocap_quat[w, 0].cpu().numpy()
        self.d.qpos[:] = env.t_qpos[w].cpu().numpy() if qpos is None else qpos
        mujoco.mj_forward(m, self.d)

    def look_at(self, target, distance=0.35, azimuth=135, elevation=-25):
        self.cam.lookat[:] = target
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation

    def render(self, *, contacts=False):
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contacts
        self.r.update_scene(self.d, self.cam, self.opt)
        return self.r.render()

    def close(self):
        self.r.close()


def save_grid(images, path, ncol=4):
    h, w = images[0].shape[:2]
    nrow = int(np.ceil(len(images) / ncol))
    grid = np.full((nrow * h, ncol * w, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, ncol)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    imageio.imwrite(path, grid)


def save_video(frames, path, fps=25):
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
