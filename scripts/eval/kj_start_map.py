"""K-J success against the hand-over state: gap, lateral offset, up offset, target windup, joint friction. Same env as the export gate (full DR), n per
condition. Starts come from the chain's error rows (angles and base lateral) with the condition's gap or lateral put on top.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/kj_start_map.py --group gap --nworld 1024 --out results/real_align/start_map_gap.json
Resumable: a condition already in --out is skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.envs.insertion_task import failure_class  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy, wilson  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402

POL = ROOT / 'deploy/bundle/models/insert_KJ.pt'
BASE = ROOT / 'results/usbc/policy/insertion/aux/starts_round3_plus_chain.npz'
TH_RING = 8


def rows_base():
    z = np.load(BASE)
    r = np.stack([z[k] for k in ('x_mm', 'y_mm', 'gap_mm', 'rx_deg', 'ry_deg', 'roll_deg')], 1)
    keep = np.isfinite(r).all(1) & (np.hypot(r[:, 0], r[:, 1]) <= 8) & (r[:, 2] >= .5) & (r[:, 2] <= 15)
    return r[keep]


def cond_rows(c, base, g):
    n = 4096
    r = base[g.integers(0, len(base), n)].copy()
    if 'gap' in c:
        r[:, 2] = c['gap']
    if 'gap_sigma' in c:
        r[:, 2] = r[:, 2] + 0
    if 'ring' in c:
        th = g.uniform(0, 2 * math.pi, n)
        r[:, 0], r[:, 1] = c['ring'] * np.cos(th), c['ring'] * np.sin(th)
    if 'x' in c:
        r[:, 0] = c['x']
    if 'y' in c:
        r[:, 1] = c['y']
    return r


GROUPS = {
    'gap': [dict(gap=x) for x in (1., 2., 3., 4., 5., 5.5, 6., 7., 8., 9., 10., 11., 12., 14.)],
    'ring': [dict(gap=6., ring=x) for x in (0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6.)],
    'ring5': [dict(gap=5., ring=x) for x in (0., 1., 2., 3.)],
    'up': [dict(gap=6., x=0., y=x) for x in (-6., -4., -3., -2., -1., 1., 2., 3., 4., 6.)],
    'high': [dict(gap=11.5, x=0., y=x) for x in (0., 3., 5., 7., 10., 14.)],
    'base': [dict()],
}


def run_cond(env, pol, c, base, seed, per_world=1, on_step=None):
    g = np.random.default_rng(seed)
    env._npz = torch.tensor(cond_rows(c, base, g), device=env.device, dtype=torch.float32)
    env.refresh_pool()
    env.reset_all()
    if on_step is not None:
        on_step('reset', env)
    hist, _ = env.observe()
    N = env.N
    count = torch.zeros(N, dtype=torch.long, device=env.device)
    res = []
    for t in range(env.max_steps * per_world + 10):
        act = pol(hist)
        if on_step is not None:
            act = on_step('act', env, act)
        hist, _, _, done, info = env.step(act)
        if 'final' in info:
            f = info['final']
            cls = failure_class(f)
            for j, w in enumerate(info['final_ids'].tolist()):
                if count[w] < per_world:
                    res.append(dict(ok=bool(f['success'][j]), cls=cls[j], force=float(f['peak_force'][j]), t=float(info['steps'][w]) * env.dt))
                count[w] += 1
        if (count >= per_world).all():
            break
    k = sum(r['ok'] for r in res)
    n = len(res)
    lo, hi = wilson(k, n)
    cl = {}
    for r in res:
        cl[r['cls']] = cl.get(r['cls'], 0) + 1
    fr = np.array([r['force'] for r in res])
    ts = np.array([r['t'] for r in res if r['ok']]) if k else np.array([np.nan])
    return dict(cond=c, k=k, n=n, rate=k / max(n, 1), wilson=[lo, hi], classes=cl, force_p50=float(np.median(fr)), force_p99=float(np.percentile(fr, 99)), t_ok_p50=float(np.median(ts)), t_ok_p99=float(np.percentile(ts, 99)))


def build(nworld, seed, friction_scale=1., extra=None):
    pol = InsertionPolicy(str(POL), 'cuda:0')
    ej = dict(pol.meta['env_json'], hard_frac=0., scale_range_frac=0., live_frac=0.)
    ej.update(extra or {})
    args = argparse.Namespace(nworld=nworld, seed=seed, action_mode='accumulate', env_json=json.dumps(ej))
    env = make_env(args)
    env.set_scale = None
    return env, pol


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=sorted(GROUPS))
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=2000)
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {json.dumps(d['cond'], sort_keys=True) for d in done}
    torch.manual_seed(A.seed)
    env, pol = build(A.nworld, A.seed, extra=json.loads(A.env_extra))
    base = rows_base()
    for i, c in enumerate(GROUPS[A.group]):
        if json.dumps(c, sort_keys=True) in seen:
            continue
        t0 = time.time()
        r = run_cond(env, pol, c, base, A.seed + i)
        done.append(r)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(done, indent=1))
        print(json.dumps(r), f'{time.time() - t0:.0f} s', flush=True)
