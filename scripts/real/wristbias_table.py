"""Wrist net (r10f, align secondary) against port vision (align primary), every align cycle of the real full_* runs; seat-2 lateral at the seat.
No motion, logs only. Part B of docs/real/ROBUSTNESS_PLAN_20261113.md.

    python scripts/real/wristbias_table.py [--days 2026-11-03 ...] [--out results/real/wristbias.json]

Per run: last cycle (hand-over) vision v / net v / FK tip, per-axis slope of each source against FK tip motion (all cycles),
net - vision per cycle, seat-2 lateral at the seat (guard.npz), FK motion hand-over -> seat (lat2 - lat0).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
DAYS = ('2026-11-03', '2026-11-06', '2026-11-07', '2026-11-10', '2026-11-11')


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) < 1.0:
        return None, None
    A = np.stack([x, np.ones_like(x)], 1)
    (b, c), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ np.array([b, c])
    se = float(np.sqrt(r @ r / max(len(x) - 2, 1)) / np.sqrt(((x - x.mean()) ** 2).sum()))
    return float(b), se


def run_row(d):
    evp = d / 'events.jsonl'
    if not evp.exists():
        return dict(run=d.name, day=d.parent.name, missing='events.jsonl')
    ev = [json.loads(x) for x in open(evp)]
    am = [e for e in ev if e['kind'] == 'align_measure']
    if not am:
        return None
    st = next((e for e in ev if e['kind'] == 'align_start'), {})
    res = next((e for e in ev if e['kind'] == 'align_result'), {})
    cyc = []
    for e in am:
        if not e.get('ok') or e.get('up') is None:
            continue
        s = e.get('second') or {}
        nv = s.get('v') if s.get('ok') else None
        fv = [f['v'] for f in e.get('frames', []) if f.get('ok') and f.get('v') and None not in f['v']]
        cyc.append(dict(tick=e['tick'], cycle=e['cycle'], v=[e['across'], e['up'], e['gap']], net=nv, tip=e['tip_rel'], spread=e.get('spread'),
                        frames=fv, decision=e.get('decision'), reason=e.get('decision_reason'), rv=e.get('rv_deg')))
    row = dict(run=d.name, day=d.parent.name, n_cycles=len(cyc), align_reason=res.get('reason'), align_ok=res.get('ok'),
               secondary=str(st.get('cfg', ''))[:0] or None)
    if not cyc:
        return row
    V = np.array([c['v'] for c in cyc], float)
    T = np.array([c['tip'] for c in cyc], float)
    N = np.array([c['net'] if c['net'] is not None else [np.nan] * 3 for c in cyc], float)
    last = cyc[-1]
    row.update(ho_vision=last['v'], ho_net=last['net'], ho_tip=last['tip'], ho_decision=last['decision'], ho_reason=last['reason'],
               ho_vision_frames=last['frames'], dnv_last=(np.array(last['net'], float) - np.array(last['v'])).tolist() if last['net'] else None)
    D = N - V
    row['dnv_all_median'] = np.nanmedian(D, 0).tolist() if np.isfinite(D).any() else None
    row['dnv_all'] = D.tolist()
    row['gap_all'] = V[:, 2].tolist()
    # FK tip in the align stage's port axes: tip_rel = tip against the stage start; a source should read v = v0 + slope * tip
    row['slope_vision'] = [slope(T[:, i], V[:, i]) for i in range(3)]
    row['slope_net'] = [slope(T[:, i], N[:, i]) for i in range(3)]
    # offsets: v - tip (constant part); net - tip
    row['vision_minus_tip_sd'] = np.nanstd(V - T, 0).tolist()
    row['net_minus_tip_sd'] = np.nanstd(N - T, 0).tolist()
    # seat 2
    sd = next((e for e in ev if e['kind'] == 'seated_detected'), None)
    ho = next((e for e in ev if e['kind'] == 'align_handover'), None)
    row['handover'] = None if ho is None else dict(ok=ho['ok'], lat=ho.get('lat_mm'), gap=ho.get('gap_mm'))
    fp = next((e for e in ev if e['kind'] in ('insert_final_push', 'insert_final_push_skipped')), None)
    row['push'] = None if fp is None else dict(kind=fp['kind'], advance=fp.get('advance_mm'), reason=fp.get('reason'), lat2=fp.get('lat2_mm'))
    ir = next((e for e in ev if e['kind'] == 'insert_result'), None)
    row['insert_outcome'] = None if ir is None else ir.get('outcome')
    g = d / 'guard.npz'
    if sd is not None and g.exists():
        z = np.load(g, allow_pickle=True)
        if 'lat2_mm' in z.files and len(z['k']):
            i = min(int(sd['step']), len(z['k']) - 1)
            lat0 = np.asarray(z['lat2_mm'][0], float)
            row['seat'] = dict(step=int(sd['step']), depth=sd['depth_mm'], lat2=z['lat2_mm'][i].tolist(), lat2_start=lat0.tolist(),
                               dfk_lat=(z['lat2_mm'][i] - lat0).tolist(), lat2_min_up=float(z['lat2_mm'][:i + 1, 1].min()),
                               lat2_max_up=float(z['lat2_mm'][:i + 1, 1].max()))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='*', default=DAYS)
    ap.add_argument('--out', default='results/real/wristbias.json')
    a = ap.parse_args()
    rows = []
    for day in a.days:
        for d in sorted(glob.glob(str(RUNS / day / 'full_*'))):
            if '_dry_' in d:
                continue
            r = run_row(Path(d))
            if r is not None:
                rows.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    f = lambda v: 'nan' if v is None else ' '.join(f'{x:6.2f}' for x in v)  # noqa: E731
    print(f'{"run":36s} {"vision ho (x,up,gap)":22s} {"net-vision ho":22s} {"med net-vis all":22s} seat lat2 / dFK  push')
    for r in rows:
        if 'missing' in r:
            print(f'{r["day"]}/{r["run"]}: no {r["missing"]}')
            continue
        if not r.get('n_cycles'):
            continue
        s = r.get('seat')
        print(f'{r["day"][5:]}/{r["run"][5:]:30s} {f(r["ho_vision"]):22s} {f(r["dnv_last"]):22s} {f(r["dnv_all_median"]):22s} '
              f'{f(s["lat2"]) if s else "-":14s} {f(s["dfk_lat"]) if s else "":14s} {(r["push"] or {}).get("kind", "")[19:]} | {r["align_reason"]}')


if __name__ == '__main__':
    main()
