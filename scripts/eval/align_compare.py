"""Final alignment methods on the AlignTask sim (blind_insertion/envs/align_task.py), same starts and plant:
step = today's stage (1 mm steps, settle, 5 frames), servo = continuous vision servo, rl:<ckpt> = a trained policy.
Per method, first episode of each world: success (Wilson 95 %), time to success, vision-consistent and true lateral error
at the end, peak tip speed, path length / start error (1 = straight line), overshoot past the 6 mm goal, failures.

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/align_compare.py --method servo --nworld 512 --out results/real_align/align_cmp/servo.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.train_bimanual import make_env  # noqa: E402
from blind_insertion.envs.align_task import ServoPolicy, StepPolicy  # noqa: E402
from blind_insertion.eval_insertion import wilson  # noqa: E402

ENV = dict(align_task=True, start_source='uniform', lateral_m=[0.005, 0.008], gap_m=[0.012, 0.024], angle_deg=3.0, episode_s=30.0,
           mask_left_obs='j6', hold_along_m=[-0.04, -0.015], hold_yaw_deg=30.0, target_cap_rad=0.05,
           vel_obs_noise=[0.006, 0.003, 0.003, 0.018, 0.016, 0.028], friction_scale_range=[0.5, 2.0], gravity_scale_jitter=0.12)


def pct(x, q):
    return np.percentile(x, q).round(2).tolist() if len(x) else None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seed', type=int, default=9100)
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--kw', default='{}', help='scripted policy keyword arguments')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    ej = dict(ENV); ej.update(json.loads(A.env_extra))
    torch.manual_seed(A.seed)
    if A.method.startswith('rl:'):
        from blind_insertion.eval_insertion import InsertionPolicy
        net = InsertionPolicy(A.method[3:], 'cuda:0')
        ej = dict(net.meta['env_json'], **{k: v for k, v in ej.items() if k in ('episode_s', 'friction_scale_range', 'gravity_scale_jitter', 'lateral_m', 'gap_m', 'angle_deg')})
        ej.update(json.loads(A.env_extra))
    env = make_env(argparse.Namespace(nworld=A.nworld, seed=A.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    kw = json.loads(A.kw)
    pol = ServoPolicy(env, **kw) if A.method == 'servo' else StepPolicy(env, **kw) if A.method == 'step' else None
    res, seen = [], torch.zeros(env.N, dtype=torch.bool, device=env.device)
    hist = env.hist
    t0 = time.time()
    for t in range(env.max_steps + 5):
        with torch.no_grad():
            a = pol() if pol is not None else net(hist).clamp(-1, 1)
        hist, crit, r, done, info = env.step(a)
        if 'final' in info:
            f, ids = info['final'], info['final_ids']
            for j, w in enumerate(ids.tolist()):
                if not seen[w]:
                    res.append({k: float(v[j]) for k, v in f.items()})
                    seen[w] = True
            if pol is not None:
                pol.reset(ids)
        if seen.all():
            break
    n, ok = len(res), [x for x in res if x['success']]
    k = len(ok)
    g = lambda key, rows=ok: np.array([x[key] for x in rows])  # noqa: E731
    out = dict(method=A.method, kw=kw, n=n, k=k, rate=k / max(n, 1), wilson=list(wilson(k, n)),
               fail={c: int(sum(x[c] for x in res)) for c in ('pushed', 'far', 'slip', 'bad')}, timeout=n - k - int(sum(x['pushed'] + x['far'] + x['slip'] + x['bad'] > 0 for x in res)),
               t_s_p50_p90=pct(g('steps') * env.dt, [50, 90]), consistent_lat_p50_p95=pct(g('lat'), [50, 95]), true_lat_p50_p95=pct(g('true_lat'), [50, 95]),
               max_speed_p50_p99=pct(g('max_speed', res), [50, 99]), max_true_speed_p50_p99=pct(g('max_true_speed', res), [50, 99]),
               true_path_ratio_p50_p90=pct(g('true_path_mm') / np.maximum(g('start_err'), 1e-3), [50, 90]), path_ratio_p50_p90=pct(g('path_mm') / np.maximum(g('start_err'), 1e-3), [50, 90]),
               overshoot_p95_max=pct(g('overshoot_mm', res), [95, 100]), wall_s=round(time.time() - t0))
    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    Path(A.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
