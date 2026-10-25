"""Generate ACT pick demonstrations (task H) with the scripted expert in MuJoCo Warp.

Each episode: top and wrist camera videos (H.264, 50 fps) and an .npz with the robot state
before each action (6 joints + finger opening), the expert action (6 joint targets, gripper
command, done flag), and the episode outcome. A fraction of the episodes get DART noise on the
executed joint targets (the recorded action stays the clean expert action): a smooth random walk
(Ornstein-Uhlenbeck, time constant `dart_tau`) during approach and carry (not near the grasp), and a smaller one for the
first `align_noise_s` of the final alignment at the port, so the data shows recoveries near the
port. Every batch draws a new reader within the USB-C tolerances (`builder.sample_reader`); the
scene randomizes physics, sensors and rendering per episode (`PickCfg.phys`, `PickCfg.rand`).
Convert with scripts/act/to_lerobot.py (LeRobot venv).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD MUJOCO_GL=egl .venv/bin/python scripts/act/gen_demos.py \
        --episodes 512 --nworld 64 --out data/act/raw_v1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio
import numpy as np
import torch

from blind_insertion.assets import builder as B
from blind_insertion.envs.pick_env import Expert, PickCfg, PickEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--out", default="data/act/raw_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dart_frac", type=float, default=0.5)
    ap.add_argument("--dart_sigma", type=float, default=0.01, help="rad, std of executed-target noise (reach, carry)")
    ap.add_argument("--align_sigma", type=float, default=0.004, help="rad, std during the start of alignment")
    ap.add_argument("--align_noise_s", type=float, default=0.8)
    ap.add_argument("--dart_tau", type=float, default=0.25, help="s, noise correlation time")
    ap.add_argument("--fixed_part", action="store_true", help="nominal reader in every batch")
    ap.add_argument("--episode_s", type=float, default=12.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(16)
    summary = []
    ep0 = 0
    t_start = time.time()
    for b in range((a.episodes + a.nworld - 1) // a.nworld):
        v = None if a.fixed_part else B.sample_reader(1, a.seed * 1000 + b)[0]
        env = PickEnv(PickCfg(nworld=a.nworld, seed=a.seed * 1000 + b, episode_s=a.episode_s), variant=v)
        dev, N, T = env.device, env.N, env.T
        env.gen.manual_seed(a.seed * 1000 + b)
        env.reset_all()
        ex = Expert(env, seed=a.seed * 1000 + b)
        ex.reset()
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed * 1000 + b + 7)
        dart = torch.rand(N, generator=g, device=dev) < a.dart_frac
        rho = math.exp(-env.control_dt / a.dart_tau)
        ou = torch.zeros(N, 6, device=dev)
        n_align = int(a.align_noise_s / env.control_dt)
        n_ep = min(N, a.episodes - ep0)
        writers = {(i, c): imageio.get_writer(out / f"ep{ep0 + i:05d}_{c}.mp4", fps=50, codec="libx264",
                                              quality=None, ffmpeg_params=["-crf", "18", "-preset", "fast"],
                                              macro_block_size=8)
                   for i in range(n_ep) for c in ("top", "wrist")}
        states, actions = [], []
        for t in range(T):
            fr = env.render()
            imgs = {c: fr[k].cpu().numpy() for c, k in (("top", "top"), ("wrist", "wrist_cam"))}
            list(pool.map(lambda ic: writers[ic].append_data(imgs[ic[1]][ic[0]]), writers.keys()))
            states.append(env.state().cpu().numpy())
            act, done = ex.act()
            actions.append(torch.cat([act, done[:, None].float()], 1).cpu().numpy())
            ou = rho * ou + math.sqrt(1 - rho ** 2) * torch.randn(N, 6, generator=g, device=dev)
            ph = ex.phase                       # no noise while descending, closing, lifting (knocks the reader)
            sig = torch.where((ph == 0) | (ph == 4), a.dart_sigma,
                              torch.where((ph == 5) & (ex.timer < n_align), a.align_sigma, 0.0))
            env.step(act, noise=ou * (sig * dart)[:, None])
        for w in writers.values():
            w.close()
        r = {k: v.cpu().numpy() for k, v in ex.result().items()}
        S, A = np.stack(states, 1), np.stack(actions, 1)
        for i in range(n_ep):
            meta = dict(episode=ep0 + i, ok=bool(r["ok"][i]), dart=bool(dart[i]), lat_mm=float(r["lat_mm"][i]),
                        gap_mm=float(r["gap_mm"][i]), tilt_deg=float(r["tilt_deg"][i]))
            np.savez_compressed(out / f"ep{ep0 + i:05d}.npz", state=S[i].astype(np.float32),
                                action=A[i].astype(np.float32), meta=json.dumps(meta))
            summary.append(meta)
        ep0 += n_ep
        ok = np.mean([m["ok"] for m in summary])
        print(f"batch {b}: {ep0} episodes, expert ok {ok:.3f}, {time.time() - t_start:.0f} s", flush=True)
        del env
    (out / "summary.json").write_text(json.dumps(dict(
        episodes=len(summary), ok=float(np.mean([m["ok"] for m in summary])), fps=50, T=T,
        state=["joint1..6 (rad)", "finger opening (m)"], action=["joint1..6 target (rad)", "gripper (+1 open, -1 close)", "done"],
        args=vars(a), episodes_meta=summary), indent=1))


if __name__ == "__main__":
    main()
