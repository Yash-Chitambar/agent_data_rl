"""Metrics of the classical port vision from eval_frames.csv (scripts/real/port_vision_fit.py eval): per gap band and split.

    python scripts/real/port_vision_report.py     -> results/real/port_vision/metrics.json, printed tables

Slope: least-squares output per mm of label (same as summary.json), insensitive to a constant offset. Residual SD: after removing
a constant over the band. False detection: ok=True and more than 2 mm across or 3 mm up/gap from the label after removing that
split's constant; on negative frames (no plug in the hand): any ok=True. Bands by label gap. Moving frames: excluded from slopes.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/real/port_vision'
BANDS = [(-99, 5), (5, 10), (10, 15), (15, 25), (25, 500)]
SHIFT = {'x': 'lx', 'y': 'ly', 'z': 'lz'}


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def stats(d, grasp_z=False):
    out = {}
    A = d[d.ok == 1]
    off = {a: float((A[a + s] - A[b]).median()) for a, b in SHIFT.items() for s in ['f' if grasp_z else '']} if len(A) else {}
    for lo, hi in BANDS:
        B = d[(d.lz >= lo) & (d.lz < hi)]
        O = B[B.ok == 1]
        s = 'f' if grasp_z else ''
        rec = dict(n=len(B), detected=int(len(O)), rate=round(len(O) / len(B), 3) if len(B) else None, rate_ci95=wilson(len(O), len(B)))
        if len(O):
            e = {a: O[a + s] - O[b] - off[a] for a, b in SHIFT.items()}
            bad = (e['x'].abs() > 2) | (e['y'].abs() > 3) | (e['z'].abs() > 3)
            rec['false'] = int(bad.sum())
            rec['false_rate'] = round(float(bad.mean()), 3)
            M = O[O.moving != True]   # noqa: E712
            rec['n_settled'] = len(M)
            if len(M) >= 3:
                for a, b in SHIFT.items():
                    if M[b].std() > 1e-6:
                        rec['slope_' + a] = round(float(np.polyfit(M[b], M[a + s], 1)[0]), 3)
                    rec['res_' + a] = round(float((M[a + s] - M[b]).std(ddof=1)), 3)
                    rec['mean_' + a] = round(float((M[a + s] - M[b]).mean()), 2)
        out[f'{lo if lo >= 0 else 0}-{hi if hi < 500 else "+"}'] = rec
    return out


def main():
    d = pd.read_csv(OUT / 'eval_frames.csv')
    d['moving'] = d.moving.map({True: True, False: False, 'True': True, 'False': False})
    from deploy.bimanual import port_vision as PV
    K = PV._camera('right')[0]
    O = d[d.ok == 1]
    Zr = O.groupby('run').Zw.median()
    rec = []
    for _, r in O.iterrows():
        R = cv2.Rodrigues(np.array([r.r0, r.r1, r.r2]))[0]
        t = np.array([r.t0, r.t1, r.t2])
        Z = Zr[r.run]
        Zt = Z + PV.TIP_DEEPER_MM
        v = r.ve - PV.TIP_ABOVE_END_PX * PV.Z0_TIP / Z
        tip = np.array([(r.u - K[0, 2]) / K[0, 0] * Zt, (v - K[1, 2]) / K[1, 1] * Zt, Zt]) + R @ np.array([0, -PV.SHELL_T_MM / 2, 0])
        rec.append(R.T @ (tip - t) - np.array(PV.load_model().get('port_center_in_model_mm', PV.PORT_CENTER_MM)))
    rec = np.array(rec)
    for i, a in enumerate('xyz'):
        d[a + 'f'] = np.nan
        d.loc[O.index, a + 'f'] = rec[:, i]
    res = {}
    sets = {'collect tune (blocks 0,2,4)': d[(d.source == 'collect') & (d.split == 'tune')],
            'collect held out (blocks 1,3,5)': d[(d.source == 'collect') & (d.split == 'test')],
            'same_spot (A, A2)': d[d.source == 'same_spot'], 'self (ai4)': d[d.source == 'self']}
    for name, s in sets.items():
        res[name] = dict(frame_width=stats(s), grasp_median_width=stats(s, True))
        print('\n==', name, 'n', len(s), 'ok', int(s.ok.sum()))
        for mode in ('frame_width', 'grasp_median_width'):
            print(' ', mode)
            for b, r in res[name][mode].items():
                if r['n']:
                    print(f"   {b:6s} n {r['n']:3d} det {r['detected']:3d} rate {r['rate']} CI {r['rate_ci95']} false {r.get('false')}"
                          f" slope {r.get('slope_x')},{r.get('slope_y')},{r.get('slope_z')} res {r.get('res_x')},{r.get('res_y')},{r.get('res_z')} mean {r.get('mean_x')},{r.get('mean_y')},{r.get('mean_z')}")
    res['unlabeled'] = {run: dict(n=len(g), ok=int(g.ok.sum())) for run, g in d[d.source == 'unlabeled'].groupby('run')}
    res['unlabeled_plug_found'] = {run: int(g.ve.notna().sum()) for run, g in d[d.source == 'unlabeled'].groupby('run')}
    N = d[d.source == 'negative']
    res['negatives'] = dict(n=len(N), ok_true=int(N.ok.sum()), ci95=wilson(int(N.ok.sum()), len(N)), plug_found=int(N.ve.notna().sum()), reasons=N.reason.fillna('').str.replace(r'[\d\.\-]+', '#', regex=True).str[:40].value_counts().to_dict())
    print('\nnegatives', res['negatives'])
    print('unlabeled', res['unlabeled'], res['unlabeled_plug_found'])
    print('ms', d.ms.median(), d.ms.quantile(.95), 'overall ok frames', d.ok.sum())
    res['time_ms'] = dict(median=float(d.ms.median()), p95=float(d.ms.quantile(.95)), n=len(d))
    (OUT / 'metrics.json').write_text(json.dumps(res, indent=1))
    d.to_csv(OUT / 'eval_frames_full.csv', index=False)


if __name__ == '__main__':
    main()
