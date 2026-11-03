"""Why does K-J climb on the real arm? Offline on logged policy.npz (no motion, no sim physics).

1. Target decomposition, port axes (mm): target tip against the hand-over tip = offset already there at the hand-over (t = 0, before any action) + sum of the policy's actions + the tracking-cap clamp.
2. Feature ablation: the policy re-run on the logged observation history with one group of inputs replaced; mean action in the port frame (mm/s of target).

    python scripts/real/climb_analysis.py --json results/real_align/climb_real.json
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
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402

D = ROOT / 'data/real_runs'
RUNS = ['2026-11-02/full_174850_full_A', '2026-11-02/full_175742_full_A2', '2026-11-01/approach_insert_171123_ai4', '2026-11-01/approach_insert_173731_ai5', '2026-11-01/approach_insert_164528_ai2', '2026-11-01/approach_insert_172549_dry_s7000_cmp_bimanual', '2026-11-01/approach_insert_172630_dry_s7001_cmp_bimanual', '2026-11-01/approach_insert_172712_dry_s7002_cmp_bimanual']
SCALE = .04


def jac(q):
    q_ = torch.as_tensor(q, dtype=torch.float32).reshape(-1, 6)
    return torch.cat([L.FK.fk(q_[i:i + 16], with_jac=True)[2] for i in range(0, len(q_), 16)]).numpy().astype(float)


def decompose(r):
    tr = L.trace(r)
    eR, q, jt, a, t = r['eR'], tr['q_m'], tr['jt'], tr['action'], tr['t']
    J = jac(q)
    pol = (J[:, :3] @ (a * SCALE)[..., None])[..., 0] @ eR * 1e3      # (n, 3) port axes, one tick of action
    dj = np.diff(jt, axis=0, prepend=jt[:1])
    tot = (J[:, :3] @ dj[..., None])[..., 0] @ eR * 1e3
    cum_pol, cum_tot = np.cumsum(pol, 0), np.cumsum(tot, 0)
    init = tr['rel_t'][0]
    pick = lambda s: int(min(np.searchsorted(t, s), len(t) - 1))  # noqa: E731
    rows = []
    for s in (.5, 1, 2, 3, 5, 7, 9.5):
        if s > t[-1] + .1:
            break
        i = pick(s)
        rows.append(dict(t=s, measured_up=float(tr['rel_m'][i, 1]), target_up=float(tr['rel_t'][i, 1]), handover_offset_up=float(init[1]), actions_up=float(cum_pol[i, 1]), clamp_up=float(cum_tot[i, 1] - cum_pol[i, 1]),
                         measured_across=float(tr['rel_m'][i, 0]), measured_in=float(-tr['rel_m'][i, 2]), target_in=float(-tr['rel_t'][i, 2])))
    return dict(run=r['name'], t0_joint_err_mrad=((jt[0] - q[0]) * 1e3).round(1).tolist(), target_offset_at_handover_mm=init.round(1).tolist(), rows=rows, action_rate_up_mm_s=dict(first_1s=float(pol[:50, 1].mean() / .02), rest=float(pol[50:, 1].mean() / .02) if len(pol) > 50 else None), j5_action_mean=float(a[:, 4].mean()), action_mean=a.mean(0).round(3).tolist(), seat_t=None)


def ablate(r, pol):
    tr = L.trace(r)
    o0 = torch.as_tensor(np.asarray(r['pol']['obs'], np.float32), device='cuda:0')
    H = int(pol.meta['history'])
    q0 = o0[0]

    def variant(fn):
        o = fn(o0.clone())
        idx = torch.arange(len(o)).to(o.device)
        hs = torch.stack([o[(idx - k).clamp(min=0)] for k in range(H - 1, -1, -1)], 1)
        with torch.no_grad():
            return torch.cat([pol(hs[i:i + 128]) for i in range(0, len(hs), 128)]).cpu().numpy().astype(float)
    sim_err = torch.tensor([-.0001, .0042, -.0036, -.0003, .0205, -.0029], device=o0.device)

    def no_inhand(o):
        o[:, 62:68] = 0
        return o

    def no_windup(o):
        o[:, 38:44] = o[:, 38:44] - o[0, 38:44] + sim_err
        return o

    def pose_nominal(o):
        o[:, 50:56] = o[:, 50:56] - o[0, 50:56]
        return o

    def no_vel(o):
        o[:, 12:18] = 0
        o[:, 56:62] = 0
        return o

    def enc_plus(o):
        o[:, :6] = o[:, :6] + torch.tensor([0, .026, .026, .026, .026, 0], device=o.device)
        return o

    def enc_minus(o):
        o[:, :6] = o[:, :6] - torch.tensor([0, .026, .026, .026, .026, 0], device=o.device)
        return o
    out = {}
    J = jac(tr['q_m'])
    for name, fn in dict(base=lambda o: o, inhand_zero=no_inhand, target_err_sim_mean=no_windup, tcp_pose_nominal=pose_nominal, velocity_zero=no_vel, enc_plus_1p5deg=enc_plus, enc_minus_1p5deg=enc_minus).items():
        a = variant(fn)
        v = (J[:, :3] @ (a * SCALE)[..., None])[..., 0] @ r['eR'] * 1e3 / .02
        out[name] = dict(up_mm_s_first1s=float(v[:50, 1].mean()), up_mm_s_rest=float(v[50:, 1].mean()) if len(v) > 50 else None, in_mm_s_first1s=float(-v[:50, 2].mean()), in_mm_s_rest=float(-v[50:, 2].mean()) if len(v) > 50 else None, j5=float(a[:, 4].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    A = ap.parse_args()
    pol = InsertionPolicy(str(ROOT / 'deploy/bundle/models/insert_KJ.pt'), 'cuda:0')
    res = []
    for name in RUNS:
        run = str(D / name)
        r = L.load(run)
        d = decompose(r)
        d['ablation'] = ablate(r, pol)
        res.append(d)
        print(json.dumps(d), flush=True)
    if A.json:
        Path(A.json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
