"""Insertion depth and seat time of an insertion policy (K7 evaluation set, docs/real/K7_20261114.md). Sim only.

Cell 'real_ho' (gap 5.5 mm, ring 0.5 mm), real-like plant, real right-arm lead, rig-spot layout (v2, PC +/-5 mm, 4 +/- 1.5 deg).
The sim success end is switched off (hold_steps huge), so each world runs its first episode on, and the script records:
  * t_seat: first time the sim seat test held 0.2 s (geometric seat within 0.25 mm of the 5.8 mm seat depth) = the usual success;
  * a deploy-like seat detector on the TRUE plug depth (insert.seat2: depth >= seat - 1 mm with the axial speed under 6 mm/s
    for 0.3 s): its time and the true depth when it fires (under the policy's load);
  * after the detector: 0.2 s measured-pose hold (right target = measured joints, action 0, as deploy after a seat), then the
    true depth 'at rest'. With a mating force (spring model) the rest depth is not valid (the spring pushes the plug back out),
    so it is reported only for mate_force 0.
  * the maximum true depth of the episode.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/eval/k7_depth.py --policy runs/insert_K4.pt --mate 0,6,10 --out results/real_align/k7/k4_depth.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.k3_robust import build  # noqa: E402
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402

RIG = {'task_layout': 'results/placement/task_layout_real_v2.json', 'pc_pose_range': [0.005, 0.005, 1.5], 'vision_prior_sigma_scale': None}
DET_MARGIN_MM, DET_SPEED_MM_S, DET_HOLD_S, REST_S = 1.0, 6.0, 0.3, 0.2


def pct(x, q):
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else None


def one(policy, mate, nworld, seed):
    extra = dict(RIG, mate_force_n=None if mate == 0 else [mate, mate])
    env, pol = build(policy, nworld, seed, extra)
    hold0 = env.hold_steps
    env.hold_steps = 10 ** 9           # no success end: the episode runs on after the seat
    N, dev, dt = env.N, env.device, env.dt
    seat_mm = env.seat_depth * 1e3
    n_det, n_rest = int(round(DET_HOLD_S / dt)), int(round(REST_S / dt))
    S = {}

    def hook(kind, e, act=None):
        if kind == 'reset':
            q0 = e._measured_model_angles()[:, 0]
            e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
            e.hist[:] = e._actor_obs()[:, None]
            nan = torch.full((N,), float('nan'), device=dev)
            S.update(live=torch.ones(N, dtype=torch.bool, device=dev), prev_steps=e.steps.clone(), prev_d=None,
                     t_seat=nan.clone(), t_det=nan.clone(), d_det=nan.clone(), d_rest=nan.clone(), d_max=torch.zeros(N, device=dev),
                     run=torch.zeros(N, dtype=torch.long, device=dev), hold=torch.full((N,), -1, dtype=torch.long, device=dev))
            return None
        live = S['live'] & (e.steps >= S['prev_steps'])          # a world whose step count fell has reset: first episode over
        S['live'] = live
        S['prev_steps'] = e.steps.clone()
        p, _ = e.runtime.plug_in_port()
        d = -p[:, 2] * 1e3
        t = e.steps.float() * dt
        S['d_max'] = torch.where(live, torch.maximum(S['d_max'], d), S['d_max'])
        new_seat = live & torch.isnan(S['t_seat']) & (e.seated_steps >= hold0)
        S['t_seat'] = torch.where(new_seat, t, S['t_seat'])
        v = torch.zeros_like(d) if S['prev_d'] is None else (d - S['prev_d']).abs() / dt
        S['prev_d'] = d.clone()
        ok = (d >= seat_mm - DET_MARGIN_MM) & (v < DET_SPEED_MM_S)
        S['run'] = torch.where(ok, S['run'] + 1, torch.zeros_like(S['run']))
        fire = live & torch.isnan(S['t_det']) & (S['run'] >= n_det)
        S['t_det'] = torch.where(fire, t, S['t_det'])
        S['d_det'] = torch.where(fire, d, S['d_det'])
        S['hold'] = torch.where(fire, torch.zeros_like(S['hold']), S['hold'])
        holding = live & (S['hold'] >= 0) & (S['hold'] < n_rest)
        done_hold = live & (S['hold'] == n_rest)
        S['d_rest'] = torch.where(done_hold & torch.isnan(S['d_rest']), d, S['d_rest'])
        if holding.any():
            meas = e._measured_model_angles()
            e.joint_target[holding, 0] = meas[holding, 0]
            act = torch.where(holding[:, None], torch.zeros_like(act), act)
        S['hold'] = torch.where(S['hold'] >= 0, S['hold'] + 1, S['hold'])
        return act
    t0 = time.time()
    r = run_cond(env, pol, dict(gap=5.5, ring=.5), rows_base(), seed, on_step=hook)
    g = {k: S[k].cpu().numpy() for k in ('t_seat', 't_det', 'd_det', 'd_rest', 'd_max')}
    n = N
    seat = np.isfinite(g['t_seat'])
    det = np.isfinite(g['t_det'])
    rest = np.isfinite(g['d_rest'])
    out = dict(policy=policy, mate_force_n=mate, n=n, seed=seed, seat_depth_mm=seat_mm, env_extra=extra,
               sim_seat_rate=float(seat.mean()), t_seat_p50_p99=pct(g['t_seat'], [50, 99]),
               det_rate=float(det.mean()), t_det_p50_p99=pct(g['t_det'], [50, 99]),
               det_depth_mm_p5_p50_p95=pct(g['d_det'], [5, 50, 95]),
               det_short_gt_0p5mm=float(((seat_mm - g['d_det']) > 0.5)[det].mean()) if det.any() else None,
               max_depth_mm_p5_p50=pct(g['d_max'], [5, 50]),
               end_classes_after_rest_not_a_rate=r['classes'], wall_s=round(time.time() - t0))
    if mate == 0:
        out.update(rest_n=int(rest.sum()), rest_depth_mm_p5_p50_p95=pct(g['d_rest'], [5, 50, 95]),
                   rest_seated_frac=float((g['d_rest'][rest] >= seat_mm - 0.25).mean()) if rest.any() else None)
    del env
    torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--mate', default='0,6,10')
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=6500)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['mate_force_n'] for d in done}
    torch.manual_seed(A.seed)
    for m in [float(x) for x in A.mate.split(',')]:
        if m in seen:
            continue
        r = one(A.policy, m, A.nworld, A.seed)
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != 'env_extra'}), flush=True)
