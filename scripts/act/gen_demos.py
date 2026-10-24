"""Generate ACT pick demonstrations (task H) with the scripted expert in MuJoCo Warp.

Each episode: top and wrist camera videos (H.264, 50 fps) and an .npz with the robot state
before each action (6 joints + finger opening), the expert action (6 joint targets, gripper
command, done flag), and the episode outcome. Half the episodes get DART noise on the executed
joint targets during reach and carry (the recorded action stays the clean expert action). Convert with
scripts/act/to_lerobot.py (LeRobot venv).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD MUJOCO_GL=egl .venv/bin/python scripts/act/gen_demos.py \
        --episodes 512 --nworld 64 --out data/act/raw_v1
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio
import numpy as np
import torch

from blind_insertion.envs.pick_env import Expert, PickCfg, PickEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--out", default="data/act/raw_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dart_frac", type=float, default=0.5)
    ap.add_argument("--dart_sigma", type=float, default=0.01, help="rad, std of executed-target noise")
    ap.add_argument("--episode_s", type=float, default=10.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PickEnv(PickCfg(nworld=a.nworld, seed=a.seed, episode_s=a.episode_s))
    dev, N, T = env.device, env.N, env.T
    pool = ThreadPoolExecutor(16)
    summary = []
    ep0 = 0
    t_start = time.time()
    for b in range((a.episodes + N - 1) // N):
        env.gen.manual_seed(a.seed * 1000 + b)
        env.reset_all()
        ex = Expert(env, seed=a.seed * 1000 + b)
        ex.reset()
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed * 1000 + b + 7)
        dart = torch.rand(N, generator=g, device=dev) < a.dart_frac
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
            on = dart & (ex.phase < 5)          # perturb reach and carry, not the final alignment
            noise = torch.randn(N, 6, generator=g, device=dev) * a.dart_sigma * on[:, None]
            env.step(act, noise=noise)
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
    (out / "summary.json").write_text(json.dumps(dict(
        episodes=len(summary), ok=float(np.mean([m["ok"] for m in summary])), fps=50, T=T,
        state=["joint1..6 (rad)", "finger opening (m)"], action=["joint1..6 target (rad)", "gripper (+1 open, -1 close)", "done"],
        args=vars(a), episodes_meta=summary), indent=1))


if __name__ == "__main__":
    main()
