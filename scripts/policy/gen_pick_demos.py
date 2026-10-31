"""Part 1 ACT demos: the privileged pick expert (blind_insertion/envs/bimanual_pick.py) on the GPU
bimanual runtime with all DR on (arm dynamics, layout on the Part 1 reader region, latency,
encoder offsets, contact DR) and visual DR (BimanualCfg.visual_dr: appearance, lighting, image
noise, camera pose/intrinsics error; `--no_visual_dr` for the fitted nominal look). RGB only.

Per episode: ZED ('top') and right wrist videos (H.264, 50 fps, frames through the measured camera
cadence) and an .npz with
  state  (T, 7): right-arm encoder angles as the policy sees them (model frame, with the episode's
                 encoder-offset error) + finger opening (m, sum of both fingers);
  action (T, 8): absolute right-arm joint targets in the same (believed) frame after the step's
                 accumulate update, gripper command (+1 open / -1 close), done flag.
DART: a fraction of episodes get an OU random walk on the executed targets away from the grasp
(phases 0 and 4); the recorded action stays the clean expert target.
Outcome per episode from the same success test as scripts/policy/pick_expert_eval.py.
Convert with scripts/policy/pick_to_lerobot.py (LeRobot venv).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/policy/gen_pick_demos.py \
        --episodes 1024 --nworld 256 --out data/act/pick_raw_v1
"""
from __future__ import annotations

import argparse, json, math, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio
import numpy as np
import torch
import warp as wp

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg

wp.config.quiet = True
CAMS = (('top', 'top'), ('wrist', 'wrist_cam'))


def finger_opening(task):
    rt = task.runtime
    return rt.q[:, task.finger_qadr].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=1024)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--out', default='data/act/pick_raw_v1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--res', type=int, default=224)
    ap.add_argument('--layout', default='results/usbc/policy/pick/task_layout.json')
    ap.add_argument('--no_visual_dr', action='store_true')
    ap.add_argument('--dart_frac', type=float, default=0.5)
    ap.add_argument('--dart_sigma', type=float, default=0.01)
    ap.add_argument('--dart_tau', type=float, default=0.25)
    ap.add_argument('--episode_s', type=float, default=9.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout=a.layout, render_rgb=True,
                                          camera_resolution=a.res, visual_dr=None if a.no_visual_dr else True),
                      episode_s=a.episode_s)
    task = PickTask(cfg)
    task.finger_qadr = [int(task.runtime.mjm.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
    pool = ThreadPoolExecutor(16)
    T = task.max_steps - 1
    summary, ep0, t0 = [], 0, time.time()
    for b in range(math.ceil(a.episodes / a.nworld)):
        seed = a.seed * 100000 + b
        torch.manual_seed(seed)
        obs = task.reset(); ex = PickExpert(task, seed=seed); ex.reset()
        dev, N = task.device, a.nworld
        g = torch.Generator(device=dev); g.manual_seed(seed + 7)
        dart = torch.rand(N, generator=g, device=dev) < a.dart_frac
        rho = math.exp(-task.dt / a.dart_tau)
        ou = torch.zeros(N, 6, device=dev)
        n_ep = min(N, a.episodes - ep0)
        writers = {(i, c): imageio.get_writer(out / f'ep{ep0 + i:05d}_{c}.mp4', fps=50, codec='libx264', quality=None,
                                              ffmpeg_params=['-crf', '18', '-preset', 'fast'], macro_block_size=8)
                   for i in range(n_ep) for c, _ in CAMS}
        states, actions, phases = [], [], []
        start_hold_p = torch.zeros(N, 3, device=dev); start_hold_R = torch.eye(3, device=dev).repeat(N, 1, 1)
        start_tilt = torch.zeros(N, device=dev); final = None
        for t in range(T):
            imgs = {c: obs['rgb'][k][:n_ep].cpu().numpy() for c, k in CAMS}
            list(pool.map(lambda ic: writers[ic].append_data(imgs[ic[1]][ic[0]]), writers.keys()))
            enc = obs['encoder_rad'][:, :6] + task.runtime.encoder_offsets[0]          # believed model frame
            states.append(torch.cat([enc, finger_opening(task)[:, None]], 1).cpu().numpy())
            prev = ex.phase.clone()
            act = ex.act()
            # recorded action: the absolute believed-frame target this step sends (clean)
            tgt = (task.joint_target[:, 0] + act[:, :6] * task.cfg.action_scale_rad)
            actions.append(torch.cat([tgt, act[:, 6:7], ex.done[:, None].float()], 1).cpu().numpy())
            phases.append(prev.cpu().numpy())
            ou = rho * ou + math.sqrt(1 - rho ** 2) * torch.randn(N, 6, generator=g, device=dev)
            sig = torch.where((prev == 0) | (prev == 4), a.dart_sigma, 0.) * dart
            act = act.clone(); act[:, :6] = (act[:, :6] + ou * sig[:, None] / task.cfg.action_scale_rad).clamp(-1, 1)
            obs = task.step(act)[0]
            m = task.pick_metrics()
            ent = (ex.phase == 5) & (prev == 4)
            start_hold_p[ent], start_hold_R[ent], start_tilt[ent] = m['inhand_p'][ent], m['inhand_R'][ent], m['tilt_deg'][ent]
            fin = (ex.phase == 6) & (prev == 5)
            if final is None: final = {k: v.clone() for k, v in m.items()}
            for k, v in m.items(): final[k][fin] = v[fin]
            if ex.done.all() and t > 20: break
        for w in writers.values(): w.close()
        dp = (final['inhand_p'] - start_hold_p).norm(dim=1)
        dR = torch.rad2deg(K.rot_log(final['inhand_R'] @ start_hold_R.transpose(1, 2)).norm(dim=1))
        ok = (ex.done & ~task.failed & (final['lift_m'] > .06) & (final['tilt_deg'] < 3) & (start_tilt < 3) & (dp < 1e-3) & (dR < 1.)).cpu().numpy()
        S, A, P = np.stack(states, 1), np.stack(actions, 1), np.stack(phases, 1)
        for i in range(n_ep):
            done_t = int(np.argmax(A[i, :, 7] > 0)) if A[i, :, 7].any() else A.shape[1] - 1
            L = min(A.shape[1], done_t + 10)                       # keep a few done frames
            meta = dict(episode=ep0 + i, ok=bool(ok[i]), dart=bool(dart[i]), length=L, tilt_deg=float(final['tilt_deg'][i]),
                        lift_mm=float(final['lift_m'][i] * 1e3), grasp_tilt_cmd_deg=float(ex.tilt[i]))
            np.savez_compressed(out / f'ep{ep0 + i:05d}.npz', state=S[i, :L].astype(np.float32), action=A[i, :L].astype(np.float32),
                                phase=P[i, :L].astype(np.int8), meta=json.dumps(meta))
            summary.append(meta)
        ep0 += n_ep
        print(f'batch {b}: {ep0} episodes, expert ok {np.mean([s["ok"] for s in summary]):.4f}, {time.time() - t0:.0f} s', flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(
        episodes=len(summary), ok=float(np.mean([m['ok'] for m in summary])), fps=50, res=a.res,
        state=['joint1..6 encoder, model frame as believed (rad)', 'finger opening (m)'],
        action=['joint1..6 absolute target, believed frame (rad)', 'gripper (+1 open, -1 close)', 'done'],
        visual_dr=not a.no_visual_dr, args=vars(a), episodes_meta=summary), indent=1))


if __name__ == '__main__':
    main()
