"""Real arm: does a small commanded step move the hand? Target against measured in the collect run (no contact in the sweep).

    python scripts/real/tracking_analysis.py data/real_runs/2026-11-02/collect_182100_collect1 --json results/real_align/tracking.json

Segments ticks where the right target stays constant for at least --hold_s and compares consecutive segments: the commanded tip step
(FK of the target) and the measured tip step (FK of the encoders), port axes. Joint level: PD error at rest, per joint, by direction of the
last move (half the difference = friction band, mean = constant load such as a gravity deficit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402


def segments(qt, stage_ok, n_min):
    """(start, end) index pairs where qt is constant for >= n_min ticks and stage_ok holds."""
    same = np.all(np.abs(np.diff(qt, axis=0)) < 1e-7, axis=1)
    out, i, n = [], 0, len(same)
    while i < n:
        if same[i] and stage_ok[i]:
            j = i
            while j < n and same[j] and stage_ok[j]:
                j += 1
            if j - i >= n_min:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyse(run, hold_s=.25, t_from=None):
    r = L.load(run)
    t = r['t']
    k = t['k'].astype(int)
    qm = t['q'][:, 0] + L.OFF
    qt = np.asarray(t['target_b'][:, 0], float)
    st = np.asarray(t['stage']).astype(int)
    # stage 'collect' logs as -1
    ok = st == -1
    anchors = [e['tick'] for e in r['ev'] if e['kind'] in ('collect_anchor',)]
    samples = [e['tick'] for e in r['ev'] if e['kind'] == 'collect_sample']
    k0, k1 = min(samples) - 300, max(samples)         # the sweep only: hands off, no contact
    ok &= (k >= k0) & (k <= k1)
    seg = segments(qt, ok[:-1], int(hold_s / .02))
    idx0 = np.array([a for a, b in seg])
    idx1 = np.array([b for a, b in seg])
    tip_t = L.fk_tip(qt[idx1])[0] @ r['eR'] * 1e3
    tip_m = L.fk_tip(qm[idx1])[0] @ r['eR'] * 1e3
    dt, dm = np.diff(tip_t, axis=0), np.diff(tip_m, axis=0)
    nt = np.linalg.norm(dt, axis=1)
    gain = (dm * dt).sum(1) / np.maximum(nt ** 2, 1e-9)
    bins = [(0, .25), (.25, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, 100)]
    tab = []
    for lo, hi in bins:
        m = (nt >= lo) & (nt < hi)
        if not m.any():
            continue
        tab.append(dict(cmd_step_mm=[lo, hi], n=int(m.sum()), gain_median=float(np.median(gain[m])), gain_p10=float(np.percentile(gain[m], 10)), gain_p90=float(np.percentile(gain[m], 90)),
                        moved_lt_25pct=float((gain[m] < .25).mean()), err_perp_mm_median=float(np.median(np.linalg.norm(dm[m] - gain[m, None] * dt[m], axis=1)))))
    # static error of target against measured
    err_tip = (L.fk_tip(qt[idx1])[0] - L.fk_tip(qm[idx1])[0]) @ r['eR'] * 1e3
    # joint level
    je = (qt - qm)[idx1 - 1]
    dq = np.diff(qm[idx1], axis=0)
    joint = {}
    for j in range(6):
        e = je[1:, j]
        up, dn = dq[:, j] > 2e-3, dq[:, j] < -2e-3
        joint[f'j{j + 1}'] = dict(mean_err_mrad=float(e.mean() * 1e3), err_when_last_up=float(e[up].mean() * 1e3) if up.any() else None, err_when_last_dn=float(e[dn].mean() * 1e3) if dn.any() else None,
                                  n_up=int(up.sum()), n_dn=int(dn.sum()), kp=float(L.SC.kp[0, j]))
        if up.any() and dn.any():
            joint[f'j{j + 1}']['friction_band_nm'] = float(L.SC.kp[0, j] * (e[up].mean() - e[dn].mean()) / 2 * 1e-0)
            joint[f'j{j + 1}']['constant_load_nm'] = float(L.SC.kp[0, j] * (e[up].mean() + e[dn].mean()) / 2)
    return dict(run=r['name'], n_segments=len(seg), step_table=tab, target_minus_measured_tip_mm=dict(mean=err_tip.mean(0).tolist(), std=err_tip.std(0).tolist(), p95_abs=np.percentile(np.abs(err_tip), 95, axis=0).tolist()), joints=joint,
                settled_time_s=float(np.median((idx1 - idx0) * .02)))


def jogs(run):
    """Operator jogs (anchor search, 3 mm/s): asked and moved, from the jog events and the ticks."""
    r = L.load(run)
    t = r['t']
    k = t['k'].astype(int)
    qm = t['q'][:, 0] + L.OFF
    ev = [e for e in r['ev'] if e['kind'] == 'collect_jog']
    tip = L.fk_tip(qm[np.searchsorted(k, [e['tick'] for e in ev])])[0] @ r['eR'] * 1e3
    rows = []
    for a, b, ta, tb in zip(ev[:-1], ev[1:], tip[:-1], tip[1:]):
        asked = np.array(b['offset_mm']) - np.array(a['offset_mm'])
        moved = tb - ta
        if np.linalg.norm(asked) > 1e-6 and a['cmd'].split()[0] in ('r', 'l', 'u', 'd', 'o'):
            rows.append(dict(cmd=a['cmd'], asked=asked.tolist(), moved=moved.round(2).tolist(), gain=float(moved @ asked / (asked @ asked))))
    g = np.array([x['gain'] for x in rows])
    return dict(n=len(rows), gain_median=float(np.median(g)) if len(g) else None, gain_p10_p90=np.percentile(g, [10, 90]).tolist() if len(g) else None, rows=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--json', default=None)
    A = ap.parse_args()
    out = analyse(A.run)
    out['operator_jogs'] = jogs(A.run)
    print(json.dumps({k: v for k, v in out.items() if k != 'operator_jogs'}, indent=1))
    print(json.dumps({k: v for k, v in out['operator_jogs'].items() if k != 'rows'}), json.dumps(out['operator_jogs']['rows'][:30]))
    if A.json:
        Path(A.json).parent.mkdir(parents=True, exist_ok=True)
        Path(A.json).write_text(json.dumps(out, indent=1))
