"""Depth signals without the operator (ideas 2 and 5 of AGENTIC_RL_FORMULA_20261113.md), on the logged real insertions. No motion, CPU.

    PYTHONPATH=. python scripts/real/depth_signals.py [--out results/real_runs_review]

(a) Contact head: the policy's contact head flips from "touching" to "inside" (inside share > 0.5) at the port mouth. Depth from the mouth at rest =
    rest depth (FK advance of the TCP since the hand-over at the end of the insertion stage, minus the vision gap: the units of insert.py
    rest_depth_mm) minus the seat-2 depth at the flip tick.
(b) Plug band: the visible silver shell between the reader neck and the case in the last right-wrist stream frame before the release. The shell is
    located on the hand-over frames (port_vision.detect_plug: rim edges ul, ur; tip edge vt; sleeve end ve). On the last frame, in the shell columns:
    the neck edge (strongest bright-to-dark step near ve) and the band top (half-way crossing between the band peak and the case level above).
    band_px = neck edge - band top. mm: band_px / ((ve - vt) / 5.8) with ve - vt from the hand-over frames (the full 5.8 mm shell in view):
    a projection along the plug axis, not corrected for the case lip.
Comparisons: Spearman rank correlation against the operator labels (2 fully in, 1 seated / not stated, 0 short or hanging) and against the hand
reading of the band in docs/real/FULL_DEPTH_20261113.md (2026-11-11, n = 6). Outputs: depth_signals.json / .csv / .png.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual import port_vision as PV  # noqa: E402

DATA = '/home/yashchitambar/peg_test/data/real_runs'
DATES = ('2026-11-03', '2026-11-06', '2026-11-10', '2026-11-11')
SHELL_MM = 5.8
# operator: 2 = fully in / flush / perfect, 1 = seated with no depth statement, 0 = short or hanging (session documents; FULL_DEPTH_20261113.md)
OPER = {'full_155211_align_full2': 2, 'full_173941_align_full9': 2, 'full_145824_kjrc_1': 0, 'full_150628_kjrc_2': 0, 'full_152149_kjrc_3_push': 0,
        'full_152717_k4_1': 0, 'full_155541_k4_3': 2, 'full_155351_demo_1': 0, 'full_160310_demo_2': 1, 'full_161223_demo_3': 1, 'full_161705_demo_4': 2,
        'full_163144_speed075_1': 1, 'full_171905_fastmove_1': 2}
# hand reading of the band (shell_end_frames.jpg, FULL_DEPTH_20261113.md): thinnest first. Rank 1 = thinnest band = deepest.
HAND_BAND_ORDER = ['full_161705_demo_4', 'full_171905_fastmove_1', 'full_155351_demo_1', 'full_160310_demo_2', 'full_161223_demo_3', 'full_163144_speed075_1']
MIN_CONTRAST = 12.0        # grey levels: a band peak less than this above the case level reads as "no visible band"


def load_run(run):
    ep = os.path.join(run, 'events.jsonl')
    if not os.path.exists(ep) or not os.path.getsize(ep):
        return None
    ev = [json.loads(x) for x in open(ep)]
    if not any(e['kind'] == 'insert_start' for e in ev):
        return None
    by = {}
    for e in ev:
        by.setdefault(e['kind'], []).append(e)
    st_t = {e['stage']: e['t'] for e in by.get('stage', [])}
    return dict(ev=ev, by=by, st_t=st_t, name=os.path.basename(run), run=run)


def contact_signal(r):
    by = r['by']
    ho, ir = by.get('align_handover', [None])[-1], by.get('insert_result', [None])[-1]
    out = dict(run=r['name'], date=Path(r['run']).parent.name, policy='K4' if 'insert_vision_prior' in by else 'K-J', outcome=ir and ir['outcome'],
               gap_mm=ho and ho.get('gap_mm'))
    p = os.path.join(r['run'], 'policy.npz')
    g = os.path.join(r['run'], 'guard.npz')
    if not (ho and ho.get('gap_mm') is not None and os.path.exists(p) and os.path.exists(g)):
        return out
    pol, gd = np.load(p), np.load(g)
    cs, d2 = pol['contact_share'], gd['depth2_mm']
    n = min(len(cs), len(d2))
    cs, d2 = cs[:n], d2[:n]
    ins = np.where(cs[:, 2] > .5)[0]
    tch = np.where(cs[:, 1] > .5)[0]
    out.update(n_ticks=n, inside_share_end=float(cs[-1, 2]), max_depth2_mm=float(np.nanmax(d2)))
    if len(tch):
        out.update(touch_tick=int(tch[0]), touch_depth_mm=float(d2[tch[0]]))
    if len(ins):
        out.update(mouth_tick=int(ins[0]), mouth_s=float(ins[0] * .02), mouth_depth_mm=float(d2[ins[0]]))
    if ir and ir['outcome'] == 'seated_detected' and ir.get('advance_mm') is not None:
        out['rest_depth_mm'] = float(ir['advance_mm'] - ho['gap_mm'])
        if 'mouth_depth_mm' in out:
            out['depth_from_mouth_mm'] = out['rest_depth_mm'] - out['mouth_depth_mm']
    return out


def _cross(p, i, level, step):
    """Sub-pixel index where p falls below level, walking from i by step."""
    if p[i] <= level:
        return None
    j = i
    while 0 <= j + step < len(p) and p[j + step] > level:
        j += step
    k = j + step
    if not 0 <= k < len(p):
        return None
    a, b = p[j], p[k]
    return j + step * (a - level) / max(a - b, 1e-6)


def band_in_frame(und, ul, ur, ve):
    """Band of one undistorted frame in the shell columns (hand-over ul, ur) near the hand-over sleeve end ve. Returns dict or None."""
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r0, r1 = int(ve) - 45, int(ve) + 15
    prof = gray[r0:r1, int(ul) + 6:int(ur) - 5].mean(1)
    gr = -np.convolve(prof, [1, 1, 0, -1, -1], mode='same') / 2.         # np.convolve flips the kernel: > 0 here = falling (bright above, dark below)
    lo, hi = 20, len(prof) - 4                                          # neck edge between ve - 25 and ve + 11
    i = lo + int(np.argmax(gr[lo:hi]))
    if gr[i] < 8:
        return None
    pk = max(i - 12, 0) + int(np.argmax(prof[max(i - 12, 0):i + 1]))     # band peak just above the neck edge
    case = float(np.median(prof[max(pk - 22, 0):max(pk - 14, 1)])) if pk >= 16 else float(prof[0])
    neck = float(np.median(prof[min(i + 3, len(prof) - 1):min(i + 10, len(prof))]))
    peak = float(prof[pk])
    contrast = peak - case
    bot = _cross(prof, pk, .5 * (peak + neck), +1)
    top = _cross(prof, pk, .5 * (peak + case), -1) if contrast > 0 else None
    # second estimate of the top: the strongest dark-to-bright step (going down) 2-18 rows above the neck edge (does not need the case level)
    tg = None
    if bot is not None:
        a0, a1 = max(int(bot) - 18, 3), max(int(bot) - 1, 4)
        j = a0 + int(np.argmin(gr[a0:a1]))
        if -gr[j] > 3:
            tg = float(j)
    res = dict(neck_row=None if bot is None else float(bot + r0), top_row=None if top is None else float(top + r0), top_grad_row=None if tg is None else tg + r0,
               contrast=contrast, peak=peak, case=case, neck=neck, edge_step=float(gr[i]),
               band_grad_px=None if (tg is None or bot is None) else float(bot - tg))
    if contrast < MIN_CONTRAST:
        res['band_px'] = 0.0
        res['visible'] = False
    elif bot is not None and top is not None:
        res['band_px'] = float(bot - top)
        res['visible'] = True
    return res


def band_signal(r):
    fp = os.path.join(r['run'], 'frames.jsonl')
    st = r['st_t']
    if not os.path.exists(fp) or '5 insert' not in st or '6 release' not in st:
        return dict(run=r['name'], band_why='no frames or no release')
    fr = [json.loads(x) for x in open(fp)]
    wr = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw']
    ho = [f for f in wr if f['tag'].startswith('align_0') and f['t_capture'] < st['5 insert']][-5:]
    last = [f for f in wr if f['tag'] == 'stream' and st['5 insert'] < f['t_capture'] < st['6 release']]
    if not last:
        return dict(run=r['name'], band_why='no stream frame in the insertion (no --stream)')
    pls = [p for p in (PV.detect_plug(PV.undistort(cv2.imread(os.path.join(r['run'], f['file'])))[0])[0] for f in ho) if p and p.get('vt') is not None]
    if len(pls) < 2:
        return dict(run=r['name'], band_why='no shell on the hand-over frames')
    ul, ur, vt, ve = (float(np.median([p[k] for p in pls])) for k in ('ul', 'ur', 'vt', 've'))
    px_mm = (ve - vt) / SHELL_MM
    out = dict(run=r['name'], shell_ul=ul, shell_ur=ur, shell_len_px=ve - vt, px_per_mm=px_mm, n_stream=len(last))
    vals = []
    for k, f in enumerate(last[-2:][::-1]):                 # last frame first, then the one before
        und, _ = PV.undistort(cv2.imread(os.path.join(r['run'], f['file'])))
        b = band_in_frame(und, ul, ur, ve)
        dt = f['t_capture'] - st['6 release']
        if k == 0:
            out.update(frame=f['file'], frame_dt_s=dt)
            if b:
                out.update({f'band_{x}': b[x] for x in ('neck_row', 'top_row', 'top_grad_row', 'contrast', 'visible', 'edge_step') if x in b})
                if b.get('band_grad_px') is not None:
                    out['band_grad_mm'] = b['band_grad_px'] / px_mm
        if b and 'band_px' in b:
            vals.append(b['band_px'])
            out[f'band_px_f{k}'] = b['band_px']
    if vals:
        out['band_px'] = vals[0]
        out['band_mm'] = vals[0] / px_mm
        if len(vals) > 1:
            out['band_mm_prev'] = vals[1] / px_mm
    return out


def spearman(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()), rho=None, p=None)
    res = stats.spearmanr(x[ok], y[ok])
    out = dict(n=int(ok.sum()), rho=round(float(res.statistic), 3), p=round(float(res.pvalue), 3))
    if ok.sum() <= 8 and len(set(x[ok])) == ok.sum() and len(set(y[ok])) == ok.sum():      # exact permutation p for small n, no ties
        import itertools
        rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
        r0 = abs(float(np.corrcoef(rx, ry)[0, 1]))
        perms = list(itertools.permutations(ry))
        out['p_exact'] = round(sum(abs(float(np.corrcoef(rx, p)[0, 1])) >= r0 - 1e-12 for p in perms) / len(perms), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DATA)
    ap.add_argument('--out', default='results/real_runs_review')
    a = ap.parse_args()
    rows = []
    for d in DATES:
        for run in sorted(glob.glob(os.path.join(a.root, d, 'full_*'))):
            if '_dry' in os.path.basename(run):
                continue
            r = load_run(run)
            if r is None:
                continue
            row = contact_signal(r)
            row.update(band_signal(r))
            row['operator'] = OPER.get(r['name'])
            row['hand_band_rank'] = HAND_BAND_ORDER.index(r['name']) + 1 if r['name'] in HAND_BAND_ORDER else None
            rows.append(row)
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k not in ('frame',)}), flush=True)
    seated = [x for x in rows if x.get('rest_depth_mm') is not None]
    get = lambda rs, k: [np.nan if x.get(k) is None else x[k] for x in rs]  # noqa: E731
    comp = {}
    lab = [x for x in seated if x.get('operator') is not None]
    for k in ('rest_depth_mm', 'depth_from_mouth_mm', 'mouth_depth_mm', 'band_mm', 'band_grad_mm'):
        sgn = -1 if k in ('band_mm', 'band_grad_mm', 'mouth_depth_mm') else 1              # deeper = thinner band
        comp[f'operator_vs_{k}'] = spearman(get(lab, 'operator'), [sgn * v for v in get(lab, k)])
        comp[f'operator_vs_{k}_K4'] = spearman(get([x for x in lab if x['policy'] == 'K4'], 'operator'), [sgn * v for v in get([x for x in lab if x['policy'] == 'K4'], k)])
        hb = [x for x in seated if x.get('hand_band_rank') is not None]
        comp[f'hand_band_vs_{k}'] = spearman([-v for v in get(hb, 'hand_band_rank')], [sgn * v for v in get(hb, k)])
    k4 = [x for x in seated if x['policy'] == 'K4']
    for k in ('rest_depth_mm', 'depth_from_mouth_mm'):
        for b in ('band_mm', 'band_grad_mm'):
            comp[f'{b}_vs_{k}'] = spearman([-v for v in get(seated, b)], get(seated, k))
            comp[f'{b}_vs_{k}_K4'] = spearman([-v for v in get(k4, b)], get(k4, k))
    # full (2) against short (0): does a threshold separate them?
    sep = {}
    for k in ('rest_depth_mm', 'depth_from_mouth_mm', 'band_mm', 'band_grad_mm'):
        f2 = [x[k] for x in seated if x.get('operator') == 2 and x.get(k) is not None]
        f0 = [x[k] for x in seated if x.get('operator') == 0 and x.get(k) is not None]
        if f2 and f0:
            sep[k] = dict(full=[round(v, 2) for v in sorted(f2)], short=[round(v, 2) for v in sorted(f0)],
                          separated=(min(f2) > max(f0)) if not k.startswith('band') else (max(f2) < min(f0)))
    comp['full_vs_short'] = sep
    comp['note'] = 'K-J contact head: the inside share ends at 0.00-0.08 on the seated K-J runs (K4: 0.74-0.96), so (a) is valid for K4 only'
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(rows=rows, comparisons=comp, method=__doc__), open(os.path.join(a.out, 'depth_signals.json'), 'w'), indent=1, default=float)
    keys = ['run', 'date', 'policy', 'outcome', 'operator', 'hand_band_rank', 'gap_mm', 'touch_depth_mm', 'mouth_s', 'mouth_depth_mm', 'rest_depth_mm', 'depth_from_mouth_mm',
            'inside_share_end', 'band_px', 'band_mm', 'band_mm_prev', 'band_grad_mm', 'band_contrast', 'band_visible', 'px_per_mm', 'frame_dt_s', 'band_why']
    with open(os.path.join(a.out, 'depth_signals.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for x in rows:
            w.writerow({k: (round(x[k], 3) if isinstance(x.get(k), float) else x.get(k)) for k in keys})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {2: 'tab:green', 1: 'tab:gray', 0: 'tab:red', None: 'k'}
    f_, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for x in seated:
        c = col[x.get('operator')]
        mk = 'o' if x['policy'] == 'K4' else 's'
        lbl = x['run'].split('_', 2)[-1]
        if x.get('depth_from_mouth_mm') is not None:
            ax[0].scatter(x['rest_depth_mm'], x['depth_from_mouth_mm'], c=c, marker=mk)
            ax[0].annotate(lbl, (x['rest_depth_mm'], x['depth_from_mouth_mm']), fontsize=6)
        if x.get('band_mm') is not None:
            ax[1].scatter(x['rest_depth_mm'], x['band_mm'], c=c, marker=mk)
            ax[1].annotate(lbl, (x['rest_depth_mm'], x['band_mm']), fontsize=6)
            if x.get('depth_from_mouth_mm') is not None:
                ax[2].scatter(x['depth_from_mouth_mm'], x['band_mm'], c=c, marker=mk)
                ax[2].annotate(lbl, (x['depth_from_mouth_mm'], x['band_mm']), fontsize=6)
    ax[0].set(xlabel='rest depth mm (FK advance - vision gap)', ylabel='depth from the mouth mm (contact head)', title=f'(a) n = {sum(x.get("depth_from_mouth_mm") is not None for x in seated)}')
    ax[1].set(xlabel='rest depth mm', ylabel='visible band mm (last wrist frame)', title=f'(b) n = {sum(x.get("band_mm") is not None for x in seated)}')
    ax[2].set(xlabel='depth from the mouth mm', ylabel='visible band mm', title='(a) against (b)')
    ax[0].axvline(7.2, color='k', lw=.5, ls=':')
    for a_ in ax:
        a_.grid(alpha=.3)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], marker='o', ls='', color=col[k], label=t) for k, t in ((2, 'operator: fully in'), (1, 'seated, not stated'), (0, 'short / hanging'))]
                 + [Line2D([], [], marker='s', ls='', color='k', label='K-J (square)')], fontsize=7)
    f_.suptitle('Real insertions 2026-11-03..11 (logged seats only); color = operator label', fontsize=9)
    f_.tight_layout()
    f_.savefig(os.path.join(a.out, 'depth_signals.png'), dpi=90)
    print(json.dumps(comp, indent=1))


if __name__ == '__main__':
    main()
