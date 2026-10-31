"""Closed-loop sim eval of a Part 1 ACT policy (TorchScript from scripts/act/export_act.py) on the GPU
bimanual runtime, same DR as the demos (dynamics, layout on the Part 1 region, latency, encoder
offsets, contact DR, visual DR), held-out seeds.

Policy inputs per control step: state (7: right encoders in the believed model frame + finger
opening), ZED 'top' and right wrist RGB in [0, 1] (frames through the measured camera cadence).
Output chunk (chunk, 8): absolute joint targets, gripper, done. Executed with temporal ensembling
(exp weights, `--ens_k`) or open-loop chunks of `--n_action_steps`.

Success (same test as the expert): after `--episode_s`, reader lifted >= 60 mm, tilt < 3 deg now
and 1 s earlier, in-hand pose change over that last 1 s < 1 mm and < 1 deg.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/policy/pick_act_eval.py \
        --policy runs/P1-act-pick-v1/act_traced.pt --episodes 1024 --seed 5000
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg

wp.config.quiet = True


def wilson_lo(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--episodes', type=int, default=1024)
ap.add_argument('--nworld', type=int, default=256)
ap.add_argument('--seed', type=int, default=5000)
ap.add_argument('--res', type=int, default=224)
ap.add_argument('--episode_s', type=float, default=10.0)
ap.add_argument('--ens_k', type=float, default=0.01, help='temporal ensembling weight exp(-k i); <0: chunked execution')
ap.add_argument('--n_action_steps', type=int, default=10)
ap.add_argument('--layout', default='results/usbc/policy/pick/task_layout.json')
ap.add_argument('--no_visual_dr', action='store_true')
ap.add_argument('--out', default='results/usbc/policy/pick/act_eval')
a = ap.parse_args()

pol = torch.jit.load(a.policy).eval().cuda()
cfg = PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout=a.layout, render_rgb=True,
                                      camera_resolution=a.res, visual_dr=None if a.no_visual_dr else True),
                  episode_s=a.episode_s + 0.5)
task = PickTask(cfg)
dev, N = task.device, a.nworld
T = round(a.episode_s / task.dt); hold = round(1.0 / task.dt)
rows, t0 = [], time.time()
for b in range(math.ceil(a.episodes / N)):
    torch.manual_seed(a.seed + b)
    obs = task.reset()
    chunk = None
    for t in range(T):
        enc = obs['encoder_rad'][:, :6] + task.runtime.encoder_offsets[0]
        state = torch.cat([enc, task.runtime.q[:, task.finger_qadr].sum(1, keepdim=True)], 1).float()
        imgs = [obs['rgb'][k].permute(0, 3, 1, 2).float() / 255 for k in ('top', 'wrist_cam')]
        if a.ens_k >= 0 or t % a.n_action_steps == 0:
            with torch.no_grad():
                chunk = torch.cat([pol(state[i:i + 128], imgs[0][i:i + 128], imgs[1][i:i + 128]) for i in range(0, N, 128)])
        if a.ens_k >= 0:   # temporal ensembling as LeRobot ACTTemporalEnsembler (oldest prediction weight 1)
            H = chunk.shape[1]
            if t == 0:
                wts = torch.exp(-a.ens_k * torch.arange(H, device=dev, dtype=torch.float32)); cum = torch.cumsum(wts, 0)
                ens, cnt = chunk.clone(), torch.ones(H, dtype=torch.long, device=dev)
            else:
                L = ens.shape[1]
                ens = (ens * cum[cnt - 1][None, :, None] + wts[cnt][None, :, None] * chunk[:, :L]) / cum[cnt][None, :, None]
                cnt = (cnt + 1).clamp(max=H - 1)
                ens = torch.cat([ens, chunk[:, L:]], 1); cnt = torch.cat([cnt, torch.ones(H - L, dtype=torch.long, device=dev)])
            act8 = ens[:, 0]; ens, cnt = ens[:, 1:], cnt[1:]
        else:
            act8 = chunk[:, t % a.n_action_steps]
        act = torch.zeros(N, 14, device=dev)
        act[:, :6] = ((act8[:, :6] - task.joint_target[:, 0]) / task.cfg.action_scale_rad).clamp(-1, 1)
        act[:, 6] = torch.where(act8[:, 6] > 0, 1., -1.); act[:, 13] = 1.
        obs = task.step(act)[0]
        if t == T - hold - 1:
            m0 = task.pick_metrics(); p0, R0, tilt0 = m0['inhand_p'].clone(), m0['inhand_R'].clone(), m0['tilt_deg'].clone()
    m = task.pick_metrics()
    dp = (m['inhand_p'] - p0).norm(dim=1); dR = torch.rad2deg(K.rot_log(m['inhand_R'] @ R0.transpose(1, 2)).norm(dim=1))
    ok = ~task.failed & (m['lift_m'] > .06) & (m['tilt_deg'] < 3) & (tilt0 < 3) & (dp < 1e-3) & (dR < 1.)
    for i in range(min(N, a.episodes - b * N)):
        rows.append(dict(success=bool(ok[i]), lift_mm=float(m['lift_m'][i] * 1e3), tilt_deg=float(m['tilt_deg'][i]),
                         hold_drift_mm=float(dp[i] * 1e3), hold_rot_drift_deg=float(dR[i]), latency=int(task.latency[i])))
    print(f'batch {b}: {np.mean([r["success"] for r in rows]):.4f} ({len(rows)}) {time.time() - t0:.0f}s', flush=True)
import pandas as pd
df = pd.DataFrame(rows); n, k = len(df), int(df.success.sum())
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
tag = f'{out.name}_{Path(a.policy).parent.name}_s{a.seed}' + ('_ens' if a.ens_k >= 0 else f'_chunk{a.n_action_steps}')
df.to_parquet(out.parent / f'{tag}.parquet')
summ = {'policy': a.policy, 'n': n, 'success': k, 'rate': k / n, 'wilson95_lo': wilson_lo(k, n), 'seed': a.seed,
        'execution': f'temporal ensembling k={a.ens_k}' if a.ens_k >= 0 else f'chunks of {a.n_action_steps}',
        'failures': {'lift<60mm': int((df.lift_mm <= 60).sum()), 'tilt>=3': int((df.tilt_deg >= 3).sum()), 'drift': int((df.hold_drift_mm >= 1).sum())}}
(out.parent / f'{tag}.json').write_text(json.dumps(summ, indent=1) + '\n')
print(json.dumps(summ, indent=1))
