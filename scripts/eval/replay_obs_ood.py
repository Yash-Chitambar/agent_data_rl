"""Which observation channels of the real insertions are outside the sim's distribution? (docs/real/K2_PROGRESS.md,
"Real-to-sim replay and roll", task 4)

Sim: the policy's own env (real-like plant, real hand-over lead, gap 5.5 mm, ring 0.5 mm), n worlds, actor obs of the first
T steps of each world's first episode. Real: policy.npz obs of the plain-PD align runs (the K-J runs; obs layout 0:68).
Per channel group and time bin: Mahalanobis distance of the real obs to the sim obs of the same time bin (Ledoit-Wolf
shrinkage covariance), and the sim's own distance distribution (p50/p99) for scale. A K2/K3 policy is checked on its own
observation: for K3 the real obs are converted (pose features relative to the hand-over, as on the robot).

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/replay_obs_ood.py --policy deploy/bundle/models/insert_KJ.pt --tag kj
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, cond_rows  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402

DATA = Path('/home/yashchitambar/peg_test/data/real_runs')
RUNS = ['2026-11-03/full_155211_align_full2', '2026-11-03/full_164912_align_full5', '2026-11-06/full_173941_align_full9', '2026-11-06/full_175030_align_full10']
GROUPS = {'q_right': (0, 6), 'qd_right': (12, 18), 'prev_act': (24, 30), 'lead_right': (38, 44), 'q_left': (6, 11), 'qd_left': (18, 23),
          'lead_left': (44, 49), 'tcp_pos': (50, 53), 'tcp_eul': (53, 56), 'twist': (56, 62), 'inhand': (62, 68)}
BINS = [(0, 10), (10, 30), (30, 60)]          # warm-up, first contact, push (steps of 20 ms)


def maha(Xs, x):
    mu = Xs.mean(0)
    C = np.cov(Xs, rowvar=False).reshape(Xs.shape[1], Xs.shape[1])
    tr = np.trace(C) / len(C)
    C = 0.9 * C + 0.1 * tr * np.eye(len(C))          # fixed shrinkage toward a scaled identity
    P = np.linalg.inv(C)
    d = lambda v: float(np.sqrt(np.einsum('...i,ij,...j->...', v - mu, P, v - mu)))  # noqa: E731
    own = np.sqrt(np.einsum('ni,ij,nj->n', Xs - mu, P, Xs - mu))
    return d(x), np.percentile(own, [50, 99])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--out', default='')
    A = ap.parse_args()
    pol = InsertionPolicy(A.policy, 'cuda:0')
    ej0 = pol.meta['env_json']
    rel = bool(ej0.get('rel_pose_obs'))
    ej = dict(ej0, hard_frac=0., scale_range_frac=0., live_frac=0., extra_ang_frac=0., handover_lead_frac=0.,
              friction_scale_range=[1.3, 2.0], gravity_scale_jitter=0.12, vel_obs_noise=[0.006, 0.003, 0.003, 0.018, 0.016, 0.028])
    env = make_env(argparse.Namespace(nworld=A.nworld, seed=7, action_mode='accumulate', env_json=json.dumps(ej)))
    g = np.random.default_rng(7)
    env._npz = torch.tensor(cond_rows({'gap': 5.5, 'ring': .5}, rows_base(), g), device=env.device, dtype=torch.float32)
    env.refresh_pool()
    env.reset_all()
    q0 = env._measured_model_angles()[:, 0]
    env.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
    o = env._actor_obs()
    env.hist[:] = o[:, None]
    hist, _ = env.observe()
    T = BINS[-1][1]
    S = np.zeros((T, env.N, 68))
    alive = np.ones(env.N, bool)
    for t in range(T):
        with torch.no_grad():
            a = pol(hist)
        hist, _, _, done, _ = env.step(a)
        S[t] = hist[:, -1, :68].cpu().numpy()
        S[t, ~alive] = np.nan
        alive &= ~done.cpu().numpy()
    res = {}
    for r in RUNS:
        P = np.load(DATA / r / 'policy.npz')['obs'][:T].copy()
        if rel:      # K2/K3 see the pose relative to the hand-over: convert the logged absolute features (first row = hand-over)
            from blind_insertion.envs import kinematics as K
            P[:, 50:53] = P[:, 50:53] - P[0, 50:53]
            P[:, 53:56] = P[:, 53:56] - P[0, 53:56]          # small-angle approximation of R0^T R
        rr = {}
        for gname, (i0, i1) in GROUPS.items():
            for b0, b1 in BINS:
                if b1 > len(P):
                    continue
                Xs = S[b0:b1, :, i0:i1].reshape(-1, i1 - i0)
                Xs = Xs[np.isfinite(Xs).all(1)]
                if Xs.std(0).max() < 1e-9:
                    continue
                x = P[b0:b1, i0:i1].mean(0)
                Xm = S[b0:b1, :, i0:i1].mean(0)
                Xm = Xm[np.isfinite(Xm).all(1)]
                dist, own = maha(Xm, x)
                rr[f'{gname}@{b0 * 0.02:.1f}-{b1 * 0.02:.1f}s'] = dict(d=round(dist, 2), sim_p50=round(float(own[0]), 2), sim_p99=round(float(own[1]), 2))
        res[r.split('/')[-1]] = rr
    # summary: groups whose real distance exceeds the sim's own p99 in every run
    keys = sorted({k for v in res.values() for k in v})
    summ = {k: [res[r][k]['d'] for r in res if k in res[r]] + [res[next(iter(res))][k]['sim_p99']] for k in keys}
    for k in keys:
        ds = [res[r][k]['d'] for r in res if k in res[r]]
        p99 = res[next(iter(res))][k]['sim_p99']
        print(f"{A.tag} {k:28s} real d {np.round(ds, 1)}  sim p50 {res[next(iter(res))][k]['sim_p50']:.1f} p99 {p99:.1f}  {'OOD' if min(ds) > p99 else ''}")
    Path(A.out or ROOT / f'results/real_align/replay/obs_ood_{A.tag}.json').write_text(json.dumps(res, indent=1))
