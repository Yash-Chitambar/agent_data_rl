"""Video of the task H insertion policy: run a checkpoint in PegEnv (MuJoCo Warp) for a few
worlds and render each world's state in the full CPU scene (PC case, cameras) from an
overview, a close-up of the port and the wrist camera. The status bar under each column turns
green on success and red on failure.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD MUJOCO_GL=egl .venv/bin/python scripts/report/h_insert_video.py \
        --policy runs/H-reader-proprio-top-s0/last.pt --out results/H/insert_demo.mp4 [--stage final|1..5]
"""
from __future__ import annotations

import argparse
import json

import imageio
import mujoco
import numpy as np
import torch

from blind_insertion.assets import builder as B
from blind_insertion.envs.peg_env import EnvCfg, PegEnv
from blind_insertion.eval import load_policy
from blind_insertion.scenes import pc_bench as S
from blind_insertion.train import USBC_STAGES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--stage", default="final", help="curriculum stage of the parts and start offsets")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--env_json", default='{"history": 10}')
    a = ap.parse_args()
    st = len(USBC_STAGES) if a.stage == "final" else int(a.stage)
    c, med = USBC_STAGES[st - 1]
    variants = B.sample_reader(a.n, 1000 + a.seed, clearance="spec" if isinstance(c, tuple) else c)
    kw = json.loads(S.CONFIG.read_text())
    kw.update(json.loads(a.env_json))
    kw.update(nworld=a.n, device="cuda:0", variants=variants, peg_offset=med / 0.765, seed=a.seed, auto_reset=False)
    env = PegEnv(EnvCfg(**kw))
    pol = load_policy(a.policy, env, env.device)
    scenes = [S.build_scene(v) for v in variants]
    datas = [mujoco.MjData(m) for m, _ in scenes]
    r_big = [mujoco.Renderer(m, 360, 480) for m, _ in scenes]
    r_cam = [mujoco.Renderer(m, 240, 320) for m, _ in scenes]
    el = env.lay
    w = imageio.get_writer(a.out, fps=25, codec="libx264", quality=8)
    hist, _ = env.observe()
    status = np.zeros(a.n)            # 0 running, 1 success, -1 failed
    for t in range(env.max_steps):
        if t % 2 == 0:
            cols = []
            for i, ((m, lay), d) in enumerate(zip(scenes, datas)):
                q = env.t_qpos[i].cpu().numpy()
                d.qpos[lay.arm_qadr] = q[el.arm_qadr]
                d.qpos[lay.finger_qadr] = q[el.finger_qadr]
                d.qpos[lay.peg_qadr:lay.peg_qadr + 7] = q[el.peg_qadr:el.peg_qadr + 7]
                d.mocap_pos[0] = env.t_mocap_pos[i, 0].cpu().numpy()
                d.mocap_quat[0] = env.t_mocap_quat[i, 0].cpu().numpy()
                mujoco.mj_forward(m, d)
                port = d.mocap_pos[0]
                cam = mujoco.MjvCamera()
                cam.lookat[:] = port + [-0.05, 0.0, 0.01]          # from the arm side, at the panel
                cam.distance, cam.azimuth, cam.elevation = 0.38, 25, -25
                r_big[i].update_scene(d, cam)
                ov = r_big[i].render()
                cam.lookat[:] = port + [-0.006, 0.0, 0.0]         # side view along the panel
                cam.distance, cam.azimuth, cam.elevation = 0.06, -90, -6
                r_cam[i].update_scene(d, cam)
                close = r_cam[i].render()
                r_cam[i].update_scene(d, "wrist_cam")
                wr = r_cam[i].render()
                bar = np.zeros((10, 480, 3), np.uint8)
                bar[:] = (60, 60, 60) if status[i] == 0 else ((40, 170, 60) if status[i] > 0 else (200, 50, 40))
                bottom = np.concatenate([close[:, 40:280], wr[:, 40:280]], 1)   # close-up | wrist camera
                cols.append(np.concatenate([ov, bar, bottom], 0))
            w.append_data(np.concatenate(cols, 1))
        with torch.no_grad():
            act, _ = pol.act_mean(hist)
        act[torch.as_tensor(status != 0, device=env.device)] = 0
        hist, _, _, done, info = env.step(act)
        for i in range(a.n):
            if status[i] == 0 and bool(done[i]):
                status[i] = 1 if bool(info["success"][i]) else -1
        if (status != 0).all() and t > 25:
            for _ in range(20):
                w.append_data(np.concatenate(cols, 1))
            break
    w.close()
    print("outcome", status.tolist())


if __name__ == "__main__":
    main()
