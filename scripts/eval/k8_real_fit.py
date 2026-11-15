"""Which port friction do the real K4 insertions fit? (docs/real/K8_20261115.md, part 1). Sim only + real logs (read-only).

sim:  contact_sens.py probe (cell real_ho, rig spot, real-like plant, deploy-like seat detector) with the port friction in bins,
      n worlds per bin. Per world: detector time t_det (s), sim seat, and the lateral path of the commanded right TCP (FK of the
      right joint target, mm, path length perpendicular to the start -> detection line) up to the detection.
real: the same two numbers from the logged real insertions (policy.npz joint_target, insert_result steps).
compare: per bin, median / p10-p90 of the sim numbers, and the two-sample KS p-value of the real values against the bin.

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/k8_real_fit.py sim --policy runs/insert_K4.pt --tag k4 --n 512
    PYTHONPATH=. python scripts/eval/k8_real_fit.py compare --tags k4
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
OUT = ROOT / 'results/real_align/k8'
RAW = ROOT / 'runs/k8_eval'
REAL = Path('/home/yashchitambar/peg_test/data/real_runs')
BINS = {'mu0.05-0.10': (0.05, 0.10), 'mu0.10-0.20': (0.10, 0.20), 'mu0.20-0.30': (0.20, 0.30), 'mu0.30-0.40': (0.30, 0.40),
        'mu0.40-0.50': (0.40, 0.50), 'mu0.50-0.60': (0.50, 0.60), 'mu0.60-0.80': (0.60, 0.80)}
# Real insertions with K4 (not dry runs; kjrc runs used another policy). Seat = seated_detected on the rig.
REAL_K4 = ['2026-11-10/full_152717_k4_1', '2026-11-10/full_154422_k4_2', '2026-11-10/full_155541_k4_3',
           '2026-11-10/full_161011_k4_4', '2026-11-10/full_163515_k4_5', '2026-11-10/full_170521_k4_6',
           '2026-11-11/full_155351_demo_1', '2026-11-11/full_160310_demo_2', '2026-11-11/full_161223_demo_3',
           '2026-11-11/full_161705_demo_4', '2026-11-11/full_163144_speed075_1', '2026-11-11/full_171905_fastmove_1',
           '2026-11-13/full_165400_all9_full_1', '2026-11-13/full_171043_all9b_full_1']
DT = 0.02


def lateral_path(p):
    """p (T, 3) TCP positions (m) up to the detection -> path length (mm) perpendicular to the start -> end line."""
    if len(p) < 2:
        return float('nan')
    ax = p[-1] - p[0]
    ax = ax / max(np.linalg.norm(ax), 1e-9)
    d = np.diff(p, axis=0)
    lat = d - (d @ ax)[:, None] * ax[None]
    return float(np.linalg.norm(lat, axis=1).sum() * 1e3)


def sim(A):
    from scripts.eval import contact_sens as CS
    from scripts.eval.kj_start_map import rows_base
    names = list(BINS)
    for k, (lo, hi) in BINS.items():
        CS.VARIANTS[k] = dict(port_mu=(lo, hi))
    CS.patch_scene('4x1.2', A.plug)
    torch.manual_seed(A.seed)
    N = A.n * len(names)
    env, pol = CS.build(A.policy, N, A.seed, 'real')
    g = np.random.default_rng(A.seed)
    rows, _ = CS.ring_rows(rows_base(), g, N * env.icfg.pool_per_world, 0.5)
    jt = []

    class Rec:                                     # records the right joint target at each policy call
        meta = pol.meta

        def __call__(self, hist, *a, **k):
            jt.append(env.joint_target[:, 0].clone())
            return pol(hist, *a, **k)
    res, cls, ok_row = CS.run_probe(env, Rec(), [(x, CS.VARIANTS[x]) for x in names], rows, A.seed)
    Q = torch.stack(jt)                            # (T, N, 6)
    T = Q.shape[0]
    P = env.right_chain.fk(Q.reshape(-1, 6))[0].reshape(T, N, 3).cpu().numpy()
    td = res['t_det']
    lat = np.full(N, np.nan)
    for w in range(N):
        if np.isfinite(td[w]):
            s = int(round(td[w] / env.dt))
            lat[w] = lateral_path(P[:s + 1, w])
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / f'realfit_{A.tag}_{A.plug}.npz', bins=np.array(names), n=A.n, lat_mm=lat, cls=cls.astype(str),
                        ok_row=ok_row, **res)
    for k, name in enumerate(names):
        sl = slice(k * A.n, (k + 1) * A.n)
        t = td[sl]; t = t[np.isfinite(t)]
        print(name, f'det {len(t)}/{A.n} seat {np.isfinite(res["t_seat"][sl]).sum()} t_det p50 {np.median(t):.2f} '
              f'lat p50 {np.nanmedian(lat[sl]):.1f} mm', flush=True)


def real_rows():
    out = []
    for r in REAL_K4:
        d = REAL / r
        ev = [json.loads(x) for x in open(d / 'events.jsonl')]
        ir = [e for e in ev if e['kind'] == 'insert_result'][-1]
        z = np.load(d / 'policy.npz')
        steps = int(ir['steps'])
        out.append(dict(run=r, outcome=ir['outcome'], steps=steps, t_det=ir.get('t_seat_s'), jt=z['joint_target'][:, 0].tolist()))
    return out


def compare(A):
    from scipy.stats import ks_2samp
    rr = real_rows()
    # real lateral path: FK of the logged right joint target with the sim chain (same model as the sim numbers)
    from scripts.eval import contact_sens as CS
    import os
    os.environ.setdefault('MUJOCO_GL', 'egl')
    env, _ = CS.build(A.fk_policy, 8, 1, 'real')
    for r in rr:
        q = torch.tensor(r['jt'], dtype=torch.float32, device=env.device)
        p = env.right_chain.fk(q)[0].cpu().numpy()
        n = r['steps'] if r['outcome'] == 'seated_detected' else len(p)
        r['lat_mm'] = lateral_path(p[:n])
        r['n_logged'] = len(p)
        del r['jt']
    seated = [r for r in rr if r['outcome'] == 'seated_detected']
    rt = np.array([r['t_det'] for r in seated]); rl = np.array([r['lat_mm'] for r in seated])
    res = dict(real=rr, real_summary=dict(n=len(rr), seated=len(seated), t_det_p10_p50_p90=np.percentile(rt, [10, 50, 90]).round(3).tolist(),
                                         lat_mm_p10_p50_p90=np.percentile(rl, [10, 50, 90]).round(2).tolist()), sim={})
    lines = ['# Real K4 insertions against sim friction bins', '',
             f'Real: {len(seated)} seated of {len(rr)} K4 insertions (rig, 2026-11-10/11/13). '
             f'Seat time p10/p50/p90 {res["real_summary"]["t_det_p10_p50_p90"]} s; lateral TCP path p10/p50/p90 '
             f'{res["real_summary"]["lat_mm_p10_p50_p90"]} mm.', '']
    for tag in A.tags.split(','):
        z = np.load(RAW / f'realfit_{tag}_{A.plug}.npz')
        n = int(z['n'])
        lines += [f'## {tag} (plug {A.plug}, n = {n} per bin)', '',
                  '| Bin | detected | sim seat | t_det p10/p50/p90 s | KS p (t) | lateral p10/p50/p90 mm | KS p (lat) |', '|---|---|---|---|---|---|---|']
        res['sim'][tag] = {}
        for k, name in enumerate(z['bins']):
            sl = slice(k * n, (k + 1) * n)
            t = z['t_det'][sl]; m = np.isfinite(t)
            la = z['lat_mm'][sl][m]
            kt, kl = ks_2samp(rt, t[m]).pvalue, ks_2samp(rl, la).pvalue
            row = dict(n=n, detected=int(m.sum()), seated=int(np.isfinite(z['t_seat'][sl]).sum()),
                       t_det_p10_p50_p90=np.percentile(t[m], [10, 50, 90]).round(3).tolist(), ks_t=float(kt),
                       lat_mm_p10_p50_p90=np.percentile(la, [10, 50, 90]).round(2).tolist(), ks_lat=float(kl))
            res['sim'][tag][str(name)] = row
            lines.append(f"| {name} | {row['detected']}/{n} | {row['seated']}/{n} | {row['t_det_p10_p50_p90']} | {kt:.3f} | "
                         f"{row['lat_mm_p10_p50_p90']} | {kl:.3f} |")
        lines.append('')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'real_fit.json').write_text(json.dumps(res, indent=1))
    (OUT / 'real_fit.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['sim', 'compare'])
    ap.add_argument('--policy', default='runs/insert_K4.pt')
    ap.add_argument('--fk_policy', default='runs/insert_K4.pt')
    ap.add_argument('--tag', default='k4')
    ap.add_argument('--tags', default='k4')
    ap.add_argument('--plug', default='nom')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--seed', type=int, default=8500)
    A = ap.parse_args()
    sim(A) if A.mode == 'sim' else compare(A)
