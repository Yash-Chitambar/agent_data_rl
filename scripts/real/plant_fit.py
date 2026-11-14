"""Small parametric push plant fitted to the logged real pushes (docs/real/FORCE_BUDGET_20261115.md, Part 2).

    PYTHONPATH=. python scripts/real/plant_fit.py            # needs results/real_align/force_budget/contact_traces.npz (force_budget.py contact)

1-D quasi-static plant along the port axis, input = the estimated force at the plug F(t) (force_budget.py, tip map), output = the
hand advance by encoder FK adv(t), from the push start to the release start (load, drop, rest with the jaws closed):
  series element (arm + grasp + PC side): deflection d with compliance c_s (mm/N) and a Coulomb play f_h (N):
      d_t = clip(d_{t-1}, c_s (F_t - f_h), c_s (F_t + f_h))
  plug: moves in only (the port holds it; by hand a plug 1-3 mm out stays put) when F exceeds the mating resistance
      R(x) = R0 + s_R x, up to x_max of travel:  x_t = max(x_{t-1}, min(x_max, (F_t - R0) / s_R))
  adv_t = x_t + d_t - d_0
Train: the 10 logged position pushes (2026-11-10/11). Held out: the 2 force pushes of 2026-11-13 (10 N, 14 N).
Fit: exhaustive grid (5 parameters), then a bootstrap over the training pushes (resample with replacement) for the spread.
"Before" = the present sim plant: c_s 0.03 mm/N (dry runs 0.02-0.04 mm/N), no play, no mating resistance, 0.1 mm of travel left.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/real_align/force_budget'
TRAIN = ['kjrc_3_push', 'k4_1', 'k4_3', 'k4_4', 'k4_5', 'k4_6', 'demo_2', 'demo_3', 'demo_4', 'fastmove_1']
TEST = ['all9_full_1', 'all9b_full_1']
GRID = dict(c_s=np.round(np.arange(0.10, 0.401, 0.01), 3), f_h=np.array([0., .25, .5, 1., 1.5, 2., 3., 4.]),
            R0=np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 14., 16., 20., 99.]),
            s_R=np.array([0.5, 1., 2., 4., 8., 16., 50.]), x_max=np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))
BEFORE = dict(c_s=0.03, f_h=0., R0=0., s_R=50., x_max=0.1)


def traces():
    z = np.load(OUT / 'contact_traces.npz')
    out = {}
    for n in TRAIN + TEST:
        j0, jr = int(z[n + '__j0']), int(z[n + '__jr'])
        F = z[n + '__F_tip'][j0:jr].astype(float)
        a = z[n + '__adv'][j0:jr].astype(float)
        out[n] = (F, a - a[0], int(z[n + '__jd']) - j0)
    return out


def simulate(F, P):
    """F (T,), P dict of arrays (k,) -> adv (T, k)."""
    c, fh, R0, sR, xm = (np.asarray(P[k], float) for k in ('c_s', 'f_h', 'R0', 's_R', 'x_max'))
    d = np.clip(0., c * (F[0] - fh), c * (F[0] + fh))
    d0 = d.copy()
    x = np.zeros_like(c)
    out = np.empty((len(F), len(c)))
    for t, f in enumerate(F):
        d = np.clip(d, c * (f - fh), c * (f + fh))
        x = np.maximum(x, np.minimum(xm, np.maximum(0., (f - R0) / sR)))
        out[t] = x + d - d0
    return out


def err(tr, names, P):
    """Per-push RMS (mm) of adv over the push window, arrays (n_push, k)."""
    return np.stack([np.sqrt(((simulate(tr[n][0], P) - tr[n][1][:, None]) ** 2).mean(0)) for n in names])


def main():
    tr = traces()
    keys = list(GRID)
    combos = np.array(list(itertools.product(*[GRID[k] for k in keys])))
    P = {k: combos[:, i] for i, k in enumerate(keys)}
    E_tr = err(tr, TRAIN, P)                     # (10, k)
    E_te = err(tr, TEST, P)
    loss = E_tr.mean(0)
    b = int(np.argmin(loss))
    best = {k: float(P[k][b]) for k in keys}
    Pb = {k: np.array([v]) for k, v in BEFORE.items()}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(200):
        idx = rng.integers(0, len(TRAIN), len(TRAIN))
        boot.append(int(np.argmin(E_tr[idx].mean(0))))
    bp = {k: P[k][boot] for k in keys}
    # mating force reached at full travel and its spread; profile of the loss over R0 (best over the other parameters)
    F_mate_full = P['R0'] + P['s_R'] * P['x_max']
    prof = {float(r): round(float(loss[P['R0'] == r].min()), 3) for r in GRID['R0']}
    near = loss <= loss[b] * 1.10
    res = dict(model='series compliance + Coulomb play + in-only plug with linear mating resistance',
               n_train=len(TRAIN), n_test=len(TEST), grid={k: v.tolist() for k, v in GRID.items()}, n_combos=int(len(combos)),
               best=best, best_train_rms_mm=round(float(loss[b]), 3),
               best_test_rms_mm={n: round(float(E_te[i, b]), 3) for i, n in enumerate(TEST)},
               before=BEFORE, before_train_rms_mm=round(float(err(tr, TRAIN, Pb).mean()), 3),
               before_test_rms_mm={n: round(float(v), 3) for n, v in zip(TEST, err(tr, TEST, Pb)[:, 0])},
               bootstrap_p5_p50_p95={k: np.percentile(bp[k], [5, 50, 95]).round(3).tolist() for k in keys},
               bootstrap_F_mate_full_p5_p50_p95=np.percentile(F_mate_full[boot], [5, 50, 95]).round(2).tolist(),
               within_10pct_of_best=dict(n=int(near.sum()), R0_range=[float(P['R0'][near].min()), float(P['R0'][near].max())],
                                         travel_at_14N_mm=[round(float(v), 2) for v in np.percentile(
                                             np.minimum(P['x_max'], np.maximum(0, (14 - P['R0']) / P['s_R']))[near], [0, 50, 100])],
                                         c_s_range=[float(P['c_s'][near].min()), float(P['c_s'][near].max())],
                                         f_h_range=[float(P['f_h'][near].min()), float(P['f_h'][near].max())]),
               loss_profile_R0=prof,
               per_push_best_rms_mm={n: round(float(v), 3) for n, v in zip(TRAIN, E_tr[:, b])})
    # held-out traces of the best plant for the figure
    pred = {n: simulate(tr[n][0], {k: np.array([best[k]]) for k in keys})[:, 0] for n in TRAIN + TEST}
    np.savez_compressed(OUT / 'plant_fit_traces.npz', **{f'{n}__pred': v for n, v in pred.items()},
                        **{f'{n}__F': tr[n][0] for n in pred}, **{f'{n}__adv': tr[n][1] for n in pred}, **{f'{n}__jd': tr[n][2] for n in pred})
    (OUT / 'plant_fit.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('grid',)}, indent=1))


if __name__ == '__main__':
    main()
