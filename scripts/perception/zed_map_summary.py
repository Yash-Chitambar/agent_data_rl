"""Tables of docs/real/ZED_MAP_20261114.md from results/placement/zed_map/{frames,arm_port}.json (zed_map.py, zed_arm_port.py).

Edge-fit truth per frame: geometry 'real' (physical port 2), all four top edges, or without the left edge when the left
gripper hides it (fewer than 30 of 40 left-edge points within 3 px). Uncertainty per frame: the spread of the
edge-subset fits (all, no left, no right) plus the validation against the arm.

    PYTHONPATH=. python scripts/perception/zed_map_summary.py --out results/placement/zed_map/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/perception'))
import fit_pc_edges as F  # noqa: E402

D = ROOT / 'results/placement/zed_map'
MAPS = ['capture_173123_map_real', 'capture_173419_map_x_plus', 'capture_173700_map_x_minus',
        'capture_174030_map_y_plus', 'capture_174453_map_y_minus', 'capture_174911_map_yaw']


def truth(row, cam):
    """(x, y, yaw), half-spread (x, y, yaw), variant used."""
    img = cv2.imread(row['jpg'])
    e = row['edge_real']
    init = (e['xy_mm'][0] / 1e3, e['xy_mm'][1] / 1e3, e['yaw_deg'])
    fits = {}
    for name, use in (('all', F.EDGES), ('noL', ('panel', 'right', 'back')), ('noR', ('panel', 'left', 'back'))):
        r = F.fit(img, init, cam=cam, geom=F.GEOMS['real'], use=use)
        fits[name] = (np.array([r['pose'][0] * 1e3, r['pose'][1] * 1e3, r['pose'][2]]), r['edges'])
    left_ok = fits['all'][1]['left']['inl'] >= 30
    use = 'all' if left_ok else 'noL'
    cand = [fits[k][0] for k in (('all', 'noL', 'noR') if left_ok else ('noL',))]
    spread = (np.max(cand, 0) - np.min(cand, 0)) / 2 if len(cand) > 1 else np.array([np.nan] * 3)
    return fits[use][0], spread, use, {k: v[0].tolist() for k, v in fits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(D / 'summary.json'))
    a = ap.parse_args()
    fr = json.loads((D / 'frames.json').read_text())['frames']
    arm = {(r['date'], r['run']): r for r in json.loads((D / 'arm_port.json').read_text())['runs']} if (D / 'arm_port.json').exists() else {}
    cam = F.camera()
    rows = []
    for r in fr:
        t, s, use, fits = truth(r, cam)
        row = dict(src=r['src'], date=r['date'], run=r['run'], tag=r.get('tag'), truth=t.tolist(), spread=s.tolist(), variant=use, fits=fits,
                   gt=r.get('gt'), edge_sim=r['edge_sim']['xy_mm'] + [r['edge_sim']['yaw_deg']])
        for n in ('v1', 'v5'):
            row[n] = r[n]['xy_mm'] + [r[n]['yaw_deg']]
            row[n + '_ok'] = r[n]['ok']
            row[n + '_maha'] = r[n]['mahalanobis']
            row[n + '_err'] = (np.array(row[n]) - t).tolist()
        if r.get('logged'):
            row['logged'] = r['logged']['xy_mm'] + [r['logged']['yaw_deg']]
            row['logged_ok'] = r['logged']['ok']
            row['logged_maha'] = r['logged'].get('mahalanobis')
        a_ = arm.get((r['date'], r['run']))
        if a_:
            row['arm'] = a_['port_mm'][:2] + [a_['yaw_deg']]
            row['arm_n'] = a_['n_meas']
            row['edge_minus_arm'] = (t[:2] - np.array(a_['port_mm'][:2])).tolist()
        rows.append(row)
    out = dict(rows=rows)
    # 1. zed_gt: against the 2026-11-01 fit
    g = [r for r in rows if r['src'] == 'zed_gt']
    if g:
        d = np.array([np.array(r['truth']) - np.array(r['gt']['xy_mm'] + [r['gt']['yaw_deg']]) for r in g])
        out['zed_gt_vs_20261101_fit'] = dict(n=len(g), median=np.median(d, 0).tolist(), min=d.min(0).tolist(), max=d.max(0).tolist())
    # 2. edge vs arm
    ea = np.array([r['edge_minus_arm'] for r in rows if 'edge_minus_arm' in r])
    if len(ea):
        out['edge_minus_arm'] = dict(n=len(ea), median=np.median(ea, 0).tolist(), p10=np.percentile(ea, 10, 0).tolist(), p90=np.percentile(ea, 90, 0).tolist(),
                                     min=ea.min(0).tolist(), max=ea.max(0).tolist(), sd=ea.std(0).tolist())
    # 3. nets against the truth, per placement group (date + rounded truth)
    out['nets'] = {}
    for n in ('v1', 'v5'):
        e = np.array([r[n + '_err'] for r in rows])
        T = np.array([r['truth'] for r in rows])
        out['nets'][n] = dict(n=len(e), slope_y=float(np.polyfit(T[:, 1], np.array([r[n][1] for r in rows]), 1)[0]),
                              slope_x=float(np.polyfit(T[:, 0], np.array([r[n][0] for r in rows]), 1)[0]),
                              err_median=np.median(e, 0).tolist(), gate_ok=int(sum(r[n + '_ok'] for r in rows)))
    # 4. the six mapping poses
    m = {r['run']: r for r in rows if r['run'] in MAPS}
    if len(m) == 6:
        ref = m[MAPS[0]]
        tab = []
        for k in MAPS:
            r = m[k]
            dt = np.array(r['truth']) - np.array(ref['truth'])
            tab.append(dict(run=k, truth=r['truth'], spread=r['spread'], variant=r['variant'], d_truth=dt.tolist(),
                            **{n: r[n] for n in ('v1', 'v5', 'logged')}, v1_ok=r['v1_ok'], v5_ok=r['v5_ok'], logged_ok=r.get('logged_ok'),
                            v1_maha=r['v1_maha'], v5_maha=r['v5_maha'], logged_maha=r.get('logged_maha'),
                            v1_err=r['v1_err'], v5_err=r['v5_err'],
                            v1_dchange=(np.array(r['v1']) - np.array(ref['v1']) - dt).tolist(),
                            v5_dchange=(np.array(r['v5']) - np.array(ref['v5']) - dt).tolist()))
        out['maps'] = tab
        for t in tab:
            f = lambda v: '(' + ', '.join(f'{x:+.1f}' if i < 2 else f'{x:+.1f}' for i, x in enumerate(v)) + ')'  # noqa: E731
            print(f"{t['run'][-12:]:12s} truth ({t['truth'][0]:.1f}, {t['truth'][1]:.1f}, {t['truth'][2]:.1f}) +/-({t['spread'][0]:.1f}, {t['spread'][1]:.1f}, {t['spread'][2]:.1f}) [{t['variant']}] "
                  f"d {f(t['d_truth'])} | v1 err {f(t['v1_err'])} ok {t['v1_ok']:d} dchg {f(t['v1_dchange'])} | v5 err {f(t['v5_err'])} ok {t['v5_ok']:d} dchg {f(t['v5_dchange'])}")
    print(json.dumps({k: v for k, v in out.items() if k not in ('rows', 'maps')}, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
