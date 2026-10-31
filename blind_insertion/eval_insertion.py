"""Evaluate a Part 4 insertion policy on held-out randomized starts (full DR), with failure classes.

Each world runs its first ``--per_world`` episodes (no bias toward short episodes); deterministic
(mean) actions; held-out seed for the start pool and all per-world randomization.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m blind_insertion.eval_insertion \
        --policy runs/P4-.../last.pt --nworld 1024 --per_world 2 --seed 1000 --scale 1.0 \
        --out results/usbc/policy/insertion/P4-...
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from blind_insertion.envs.insertion_task import failure_class
from blind_insertion.policies.tcn_actor_critic import ActorCritic
from blind_insertion.train_bimanual import make_env, set_scale


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load_policy(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    ac = ActorCritic(ck['obs_dim'], ck['act_dim'], ck['critic_dim'], history=ck['history'], n_points=ck['n_points']).to(dev)
    ac.load_state_dict(ck['model'])
    ac.eval()
    ac.obs_norm.freeze(); ac.crit_norm.freeze()
    return ac, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, help="checkpoint path or 'scripted' (privileged controller)")
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--per_world', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--action_mode', default='')
    ap.add_argument('--env_json', default=None, help='default: the checkpoint\'s env_json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--start_source', default='', help="override the start errors: 'uniform' or an npz path")
    ap.add_argument('--pc_mass_scale', type=float, default=1.0, help='documentation only; set it in --env_json')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device('cuda:0')
    student = None
    if a.policy == 'scripted':
        ac, ck = None, {}
    else:
        ck = torch.load(a.policy, map_location=dev, weights_only=False)
        if 'variant' in ck:          # DAgger student (distill_insertion.py)
            from blind_insertion.distill_insertion import Student
            student = Student(ck['cams'], ck['res'], history=ck['history'], n_points=ck['n_points']).to(dev)
            student.load_state_dict(ck['model']); student.eval(); student.obs_norm.freeze()
            ac = None
            ck = dict(ck, env_json=json.dumps(ck['env_json']))
        else:
            ac, ck = load_policy(a.policy, dev)
    args = argparse.Namespace(nworld=a.nworld, seed=a.seed,
                              action_mode=a.action_mode or ck.get('action_mode', 'accumulate'),
                              env_json=a.env_json if a.env_json is not None else ck.get('env_json', ''))
    env = make_env(args)
    if a.start_source:
        env.icfg.start_source = a.start_source
    if a.pc_mass_scale != 1.0 and env.runtime.cfg.pc_mass_scale != a.pc_mass_scale:
        raise ValueError('pass pc_mass_scale through --env_json (the model is built at construction)')
    set_scale(env, a.scale)
    env.reset_all()
    hist, _ = env.observe()
    N = env.N
    count = torch.zeros(N, dtype=torch.long, device=dev)
    rows = []
    starts = {k: v.clone() for k, v in env.start.items()}
    params = env.runtime.contact_params.clone()
    lat, enc = env.latency.clone(), env.enc_error.clone()
    pc_max = torch.zeros(N, device=dev)
    for t in range(env.max_steps * a.per_world + 10):
        with torch.no_grad():
            if student is not None:
                rt = env.runtime
                frames, ages = rt.camera_sensor.observation(rt.sensor_time) if student.cams else ({}, {})
                mask = (env.steps.float() * env.dt < ck['vis_s']).float() if ck['variant'] == 'wrist_approach' \
                    else torch.ones(N, device=dev)
                act, _ = student(hist[..., :62], frames, ages, mask)
            elif ac is None:
                act = env.scripted_action(gain=0.1)
            else:
                mu, _ = ac.act_mean(hist)
                act = mu if not a.stochastic else ac.dist(hist)[0].sample()
        hist, _, _, done, info = env.step(act)
        pc_max = torch.maximum(pc_max, info['pc_d'])
        if 'final' in info:
            ids = info['final_ids']
            f = info['final']
            cls = failure_class(f)
            for j, w in enumerate(ids.tolist()):
                if count[w] < a.per_world:
                    rows.append(dict(world=w, episode=int(count[w]), outcome=cls[j], success=bool(f['success'][j]),
                                     strict_seated_hold=bool(f['strict_hold'][j]),
                                     t_s=float(info['steps'][w]) * env.dt,
                                     tip_x_mm=float(f['p'][j, 0] * 1e3), tip_y_mm=float(f['p'][j, 1] * 1e3),
                                     tip_z_mm=float(f['p'][j, 2] * 1e3), tilt_deg=float(f['tilt'][j]),
                                     pc_max_mm=float(pc_max[w] * 1e3),
                                     start_lat_x_mm=float(starts['err_p'][w, 0] * 1e3),
                                     start_lat_y_mm=float(starts['err_p'][w, 1] * 1e3), start_rx_deg=float(starts['err_ang'][w, 0] * 180 / math.pi), start_ry_deg=float(starts['err_ang'][w, 1] * 180 / math.pi), start_roll_deg=float(starts['err_ang'][w, 2] * 180 / math.pi),
                                     start_gap_mm=float(starts['err_p'][w, 2] * 1e3),
                                     start_ang_deg=float(starts['err_ang'][w].norm() * 180 / math.pi),
                                     inhand_x_mm=float(starts['inhand_p'][w, 0] * 1e3),
                                     inhand_z_mm=float(starts['inhand_p'][w, 2] * 1e3),
                                     latency=int(lat[w]), enc_err_max_deg=float(enc[w].abs().max() * 180 / math.pi),
                                     **{k: float(params[w, i]) for i, k in enumerate(env.runtime.CONTACT_DR_KEYS)}))
                count[w] += 1
            # parameters of the new episodes
            for k in starts:
                starts[k][ids] = env.start[k][ids]
            params[ids] = env.runtime.contact_params[ids]
            lat[ids], enc[ids] = env.latency[ids], env.enc_error[ids]
            pc_max[ids] = 0
        if (count >= a.per_world).all():
            break
    import pandas as pd
    df = pd.DataFrame(rows)
    n, k = len(df), int(df.success.sum())
    lo, hi = wilson(k, n)
    summary = dict(policy=a.policy, n=n, seated=k, rate=k / n, wilson95=[lo, hi], scale=a.scale, seed=a.seed,
                   outcomes=df.outcome.value_counts().to_dict(),
                   strict_seated_rate=float(df.strict_seated_hold.mean()),
                   aperture_tol_m=env.icfg.aperture_tol_m,
                   t_seat_s_median=float(df[df.success].t_s.median()) if k else None,
                   t_seat_s_p95=float(df[df.success].t_s.quantile(.95)) if k else None,
                   pc_max_mm_p99=float(df.pc_max_mm.quantile(.99)),
                   gate_pass=bool(k / n >= 0.995 and lo >= 0.99 and n >= 1000))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out) + '.csv', index=False)
    Path(str(out) + '.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    main()
