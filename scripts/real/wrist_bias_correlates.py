"""Wrist-net "up" bias at the align hand-over against candidate causes (docs/real/PERCEPTION_BIAS_20261115.md, Part B).

Data: every real run (meta dry_run false) with align measurements and the secondary wrist net wrist_r10f_ft_abs
(2026-11-03..13). Per align measurement: port vision (across, up, gap; the primary), the raw net output (the logged
`net` event of the same tick: across, up, gap, rx, ry, rz), q_b, grip, conf, sigma, panel rotation. Per run: the
pick grasp (grasp_v3 squeeze_end: rot_deg, roll_deg, rise_mm, reading; touchdown z), the in-hand net inhand_ih5 at the
approach end, the align in-hand refresh (align_inhand after), brightness of the right wrist policy frame of the
measurement (mean grey; mean grey of the centre crop the net reads).

Response per run: d_up = raw net up - vision up, median over the measurements with vision gap <= 10 mm (the hand-over
region), and the value at the last measurement before the hand-over. Spearman and Pearson r with n per covariate;
leave-one-run-out check of any linear correction.

    PYTHONPATH=. python scripts/real/wrist_bias_correlates.py --out results/placement/perception_bias/wrist_bias.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
DATES = ['2026-11-03', '2026-11-06', '2026-11-10', '2026-11-11', '2026-11-13']
NET = 'wrist_r10f_ft_abs'


def events(run):
    out = []
    p = run / 'events.jsonl'
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def frame_index(run):
    idx = {}
    p = run / 'frames.jsonl'
    if not p.exists():
        return idx
    for line in p.read_text().splitlines():
        try:
            f = json.loads(line)
        except json.JSONDecodeError:
            continue
        if f.get('cam') == 'wrist_right' and f.get('kind') == 'policy':
            idx.setdefault(f.get('tick'), []).append(f['file'])
    return idx


def brightness(run, files):
    g, c = [], []
    for f in files:
        im = cv2.imread(str(run / f))
        if im is None:
            continue
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g.append(gr.mean())
        c.append(gr[40:232, 64:256].mean())
    return (float(np.mean(g)), float(np.mean(c))) if g else (None, None)


def pv_rows(run, ticks):
    """Port vision re-run on the raw right wrist frames of each measurement (tick - 6 .. tick): the plug tip in the
    camera frame (= hand frame: the in-hand plug position, mm), plug and port pixels, shell width and depth."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deploy.bimanual import port_vision as PV
    fr = [json.loads(x) for x in (run / 'frames.jsonl').read_text().splitlines() if x.strip()]
    fr = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw' and f.get('tick') is not None]
    out = {}
    for tk in ticks:
        R = []
        for f in fr:
            if not tk - 6 <= f['tick'] <= tk:
                continue
            img = cv2.imread(str(run / f['file']))
            m = PV.measure(img) if img is not None else None
            if m is None or not m.ok:
                continue
            pl = m.debug['plug']
            R.append([*m.debug['tip_cam'], *m.plug_px, *m.port_px, pl['w'], m.debug['Z_w'], *m.debug['t']])
        if R:
            out[tk] = np.median(np.array(R, float), 0).tolist()
    return out


def stop_tip(run):
    """The plug tip y in the camera at the align stop, from the prepare frames (tag align_depth_*): the shell depth of
    the grasp (port_vision.plug_depth) and port_vision.measure on each frame, median tip_cam. This is what the
    deploy option align.secondary.bias_model computes on the rig (one value per grasp, before any step)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deploy.bimanual import port_vision as PV
    fr = [json.loads(x) for x in (run / 'frames.jsonl').read_text().splitlines() if x.strip()]
    fs = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw' and str(f.get('tag', '')).startswith('align_depth')]
    imgs = [im for im in (cv2.imread(str(run / f['file'])) for f in fs) if im is not None]
    if len(imgs) < 3:
        return None
    z = PV.plug_depth(imgs)
    if z is None:
        return None
    T = [m.debug['tip_cam'] for m in (PV.measure(im, plug_z_mm=z) for im in imgs) if m.ok]
    return float(np.median([t[1] for t in T])) if T else None


PV_KEYS = ['tip_cx', 'tip_cy', 'tip_cz', 'plug_u', 'plug_v', 'port_u', 'port_v', 'shell_w', 'shell_z', 'panel_tx', 'panel_ty', 'panel_tz']


def run_rows(run, pv=False):
    meta = json.loads((run / 'meta.json').read_text())
    if meta.get('dry_run', True):
        return None
    ev = events(run)
    nets = {}
    for e in ev:
        if e.get('kind') == 'net' and NET in e.get('net', '') and '@' in e.get('frame_tag', ''):
            nets[int(e['frame_tag'].split('@')[1])] = e['out'][0]
    am = [e for e in ev if e.get('kind') == 'align_measure' and e.get('ok') and e.get('up') is not None]
    if not am:
        return None
    fidx = frame_index(run)
    pvr = pv_rows(run, [e['tick'] for e in am if e['gap'] <= 10.]) if pv else {}
    meas = []
    for e in am:
        raw = nets.get(e['tick'])
        if raw is None:
            continue
        b = brightness(run, fidx.get(e['tick'], [])[:3])
        meas.append(dict(tick=e['tick'], cycle=e.get('cycle'), across=e['across'], up=e['up'], gap=e['gap'], conf=e.get('conf'), sigma=e.get('sigma'),
                         spread=e.get('spread'), rv_deg=e.get('rv_deg'), q_b=e.get('q_b'), grip=e.get('grip'), decision=e.get('decision'),
                         tip_rel=e.get('tip_rel'), net=raw, d=[raw[0] - e['across'], raw[1] - e['up'], raw[2] - e['gap']],
                         bright=b[0], bright_crop=b[1], pv=pvr.get(e['tick'])))
    if not meas:
        return None
    g = {}
    for e in ev:
        if e.get('kind') == 'grasp_v3' and e.get('event') in ('squeeze_end', 'close_from_settle', 'end_hold'):
            g[e['event']] = e            # last attempt wins
    ih_net = [e['out'][0] for e in ev if e.get('kind') == 'net' and e.get('net') == 'inhand_ih5']
    ih_al = [e for e in ev if e.get('kind') == 'align_inhand']
    ho = [e for e in ev if e.get('kind') == 'align_handover']
    res = [e for e in ev if e.get('kind') == 'align_result']
    return dict(date=run.parent.name, run=run.name, n_meas=len(meas), meas=meas, tip_cy_stop=stop_tip(run) if pv else None,
                handover=ho[-1] if ho else None, result=res[-1].get('reason') if res else None,
                grasp=dict(rot_deg=g.get('squeeze_end', {}).get('rot_deg'), roll_deg=g.get('squeeze_end', {}).get('roll_deg'),
                           rise_mm=g.get('squeeze_end', {}).get('rise_mm'), reading=g.get('squeeze_end', {}).get('reading'),
                           z_close=g.get('close_from_settle', {}).get('z'), rot_end_hold=g.get('end_hold', {}).get('rot_deg')),
                inhand_net=ih_net[-1] if ih_net else None, inhand_align=ih_al[-1]['after'] if ih_al else None,
                inhand_align_before=ih_al[-1]['before'] if ih_al else None)


def spearman(x, y):
    from scipy.stats import spearmanr, pearsonr
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return dict(n=int(m.sum()))
    rs, ps = spearmanr(x[m], y[m])
    rp, pp = pearsonr(x[m], y[m])
    k = np.polyfit(x[m], y[m], 1)
    return dict(n=int(m.sum()), spearman=round(float(rs), 3), p_spearman=round(float(ps), 4), pearson=round(float(rp), 3),
                p_pearson=round(float(pp), 4), slope=round(float(k[0]), 4), intercept=round(float(k[1]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='+', default=DATES)
    ap.add_argument('--near-gap', type=float, default=10.)
    ap.add_argument('--pv', action='store_true', help='re-run port vision on the raw wrist frames (in-hand plug position)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs = []
    for d in a.dates:
        for r in sorted((RUNS / d).glob('*')):
            if r.is_dir() and (r / 'meta.json').exists():
                x = run_rows(r, pv=a.pv)
                if x:
                    runs.append(x)
    table = []
    for r in runs:
        near = [m for m in r['meas'] if m['gap'] <= a.near_gap]
        if not near:
            continue
        last = near[-1]
        med = lambda k, i=None: float(np.median([(m[k] if i is None else m[k][i]) for m in near if m[k] is not None])) if any(m[k] is not None for m in near) else float('nan')
        rv = [m['rv_deg'] for m in near if m['rv_deg']]
        row = dict(date=r['date'], run=r['run'], n_near=len(near), tip_cy_stop=r['tip_cy_stop'] if r['tip_cy_stop'] is not None else float('nan'), handover=r['handover'] is not None, result=r['result'],
                   d_up=med('d', 1), d_up_last=last['d'][1], d_across=med('d', 0), d_gap=med('d', 2),
                   vis_up=med('up'), vis_across=med('across'), vis_gap=med('gap'), net_up=med('net', 1),
                   net_rx=med('net', 3), net_ry=med('net', 4), net_rz=med('net', 5),
                   vis_rx=float(np.median([v[0] for v in rv])) if rv else float('nan'),
                   vis_ry=float(np.median([v[1] for v in rv])) if rv else float('nan'),
                   vis_rz=float(np.median([v[2] for v in rv])) if rv else float('nan'),
                   conf=med('conf'), sigma=med('sigma'), bright=med('bright'), bright_crop=med('bright_crop'), grip=med('grip'),
                   **{f'q{j + 1}': med('q_b', j) for j in range(6)},
                   **{k: (float(np.median([m['pv'][j] for m in near if m.get('pv')])) if any(m.get('pv') for m in near) else float('nan')) for j, k in enumerate(PV_KEYS)},
                   **{f'g_{k}': (v if v is not None else float('nan')) for k, v in r['grasp'].items()},
                   **{f'ihn_{j}': (r['inhand_net'][j] if r['inhand_net'] else float('nan')) for j in range(6)},
                   **{f'iha_{j}': (r['inhand_align'][j] if r['inhand_align'] else float('nan')) for j in range(6)})
        table.append(row)
    keys = [k for k in table[0] if k not in ('date', 'run', 'n_near', 'handover', 'result', 'd_up', 'd_up_last')]
    corr = {k: spearman([t[k] for t in table], [t['d_up'] for t in table]) for k in keys}
    # within-run: per measurement, d_up against the gap and brightness (all measurements, run-centred)
    W = []
    for r in runs:
        ds = np.array([m['d'][1] for m in r['meas']])
        for m in r['meas']:
            W.append((m['gap'], m['bright_crop'] if m['bright_crop'] is not None else np.nan, m['up'], m['d'][1] - ds.mean(), m['d'][1]))
    W = np.array(W, float)
    within = dict(n_meas=len(W), gap=spearman(W[:, 0], W[:, 3]), bright_crop=spearman(W[:, 1], W[:, 3]), vis_up=spearman(W[:, 2], W[:, 3]),
                  pooled_gap=spearman(W[:, 0], W[:, 4]))
    H = [t for t in table if t['handover']]
    corr_ho = {k: spearman([t[k] for t in H], [t['d_up'] for t in H]) for k in keys}
    loo = {}
    for k in ('tip_cy_stop', 'tip_cy'):
        for name, S in (('handover', H), ('all', table)):
            S = [t for t in S if np.isfinite(t[k])]
            if len(S) < 5:
                continue
            x, y = np.array([t[k] for t in S]), np.array([t['d_up'] for t in S])
            res = np.array([y[i] - np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) for i in range(len(S))])
            loo[f'{k}/{name}'] = dict(n=len(S), fit=np.polyfit(x, y, 1).round(4).tolist(), loo_resid=[round(float(res.min()), 2), round(float(res.max()), 2)],
                                      loo_sd=round(float(res.std()), 2), const_resid=[round(float((y - y.mean()).min()), 2), round(float((y - y.mean()).max()), 2)])
    print('LOO', json.dumps(loo))
    ranked = sorted(((k, v) for k, v in corr.items() if 'spearman' in v), key=lambda kv: -abs(kv[1]['spearman']))
    print(f'runs {len(table)} (with a measurement at gap <= {a.near_gap} mm), measurements {len(W)}')
    print('d_up per run: ' + ' '.join(f"{t['d_up']:+.2f}" for t in table))
    for k, v in ranked[:20]:
        print(f'{k:14s} n {v["n"]:3d} rho {v["spearman"]:+.2f} (p {v["p_spearman"]:.3f}) r {v["pearson"]:+.2f} slope {v["slope"]:+.3f}')
    print('within-run', json.dumps(within))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__.strip(), near_gap_mm=a.near_gap, table=table, corr=corr, within=within, corr_handover=corr_ho, loo=loo), indent=0, default=float))


if __name__ == '__main__':
    main()
