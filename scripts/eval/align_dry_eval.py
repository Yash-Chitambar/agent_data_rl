"""Summarise dry runs of the align stage (deploy/run_bimanual.py --dry-run, many --episodes): how close and how aligned at the hand-over (sim truth), gates, insertion.

    python scripts/eval/align_dry_eval.py <dir with run dirs> [--match name] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def boot(x, f=np.median, B=2000, seed=0):
    x = np.asarray(x, float)
    if not len(x):
        return [None, None]
    g = np.random.default_rng(seed)
    v = [f(x[g.integers(0, len(x), len(x))]) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def one(run):
    ev = [json.loads(x) for x in open(run / 'events.jsonl')]
    r = dict(run=run.name)
    res = [e for e in ev if e['kind'] == 'align_result']
    if not res:
        r['align'] = None
        return r
    ar = res[-1]
    r.update(align=ar['ok'], reason=ar.get('reason'), cycles=ar.get('cycles'), seconds=ar.get('seconds'), slope=ar.get('slope'))
    ms = [e for e in ev if e['kind'] == 'align_measure']
    r['entry_truth'] = ms[0].get('truth_mm') if ms else None
    r['n_step'] = sum(1 for e in ev if e['kind'] == 'align_step')
    st = [e for e in ev if e['kind'] == 'align_step']
    r['frac'] = [e['frac'] for e in st]
    if ms:
        r['truth'] = ms[-1].get('truth_mm')
        r['rot'] = ms[-1].get('truth_rot_deg')
        r['meas'] = [ms[-1].get('across'), ms[-1].get('up'), ms[-1].get('gap')]
    ins = [e for e in ev if e['kind'] == 'insert_result']
    if ins:
        r.update(insert=ins[-1]['outcome'], true_seated=ins[-1].get('true_seated_end'), t_seat=ins[-1].get('t_seat_s'), depth=ins[-1].get('depth_mm'))
    r['stop'] = [e.get('outcome') for e in ev if e['kind'] == 'insert_stop']
    meta = json.loads((run / 'meta.json').read_text())
    r['completed'] = (meta.get('result') or {}).get('completed')
    r['seated_after_release'] = ((meta.get('result') or {}).get('results') or {}).get('true_seated_after_release')
    r['stopmsg'] = (meta.get('result') or {}).get('stop')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--match', default='')
    ap.add_argument('--json', default=None)
    A = ap.parse_args()
    runs = [p for p in Path(A.root).rglob('events.jsonl') if A.match in p.parent.name]
    R = [one(p.parent) for p in sorted(runs)]
    R = [r for r in R if r.get('align') is not None]
    n = len(R)
    ok = [r for r in R if r['align']]
    out = dict(n=n, handed_over=len(ok), handed_over_ci=wilson(len(ok), n))
    reasons = {}
    for r in R:
        if not r['align']:
            key = (r['reason'] or '').split(':')[0].split(' (')[0]
            reasons[key] = reasons.get(key, 0) + 1
    out['stop_reasons'] = reasons
    if ok:
        T = np.array([r['truth'] for r in ok], float)
        g = T[:, 2]
        out['at_handover_truth_mm'] = dict(across_abs_p50=float(np.median(np.abs(T[:, 0]))), across_abs_p95=float(np.percentile(np.abs(T[:, 0]), 95)), across_abs_p50_ci=boot(np.abs(T[:, 0])), up_abs_p50=float(np.median(np.abs(T[:, 1]))), up_abs_p95=float(np.percentile(np.abs(T[:, 1]), 95)), up_abs_p50_ci=boot(np.abs(T[:, 1])),
                                           lateral_p50=float(np.median(np.hypot(T[:, 0], T[:, 1]))), lateral_p95=float(np.percentile(np.hypot(T[:, 0], T[:, 1]), 95)), lateral_p50_ci=boot(np.hypot(T[:, 0], T[:, 1])), gap_p5=float(np.percentile(g, 5)), gap_p50=float(np.median(g)), gap_p95=float(np.percentile(g, 95)), gap_min=float(g.min()), gap_max=float(g.max()))
        rot = np.array([r['rot'] for r in ok], float)
        out['rot_abs_p95_deg'] = np.percentile(np.abs(rot), 95, axis=0).round(2).tolist()
        out['cycles_p50_p95'] = [float(np.percentile([r['cycles'] for r in ok], x)) for x in (50, 95)]
        out['seconds_p50_p95'] = [float(np.percentile([r['seconds'] for r in ok], x)) for x in (50, 95)]
        ins = [r for r in ok if 'insert' in r]
        if ins:
            s = [r for r in ins if r.get('true_seated')]
            det = [r for r in ins if r['insert'] == 'seated_detected']
            out['insert'] = dict(n=len(ins), true_seated=len(s), true_seated_ci=wilson(len(s), len(ins)), detected=len(det), false_detect=sum(1 for r in det if not r.get('true_seated')), missed=sum(1 for r in ins if r.get('true_seated') and r['insert'] != 'seated_detected'), seated_after_release=sum(1 for r in ins if r.get('seated_after_release')), 
                                 outcomes={k: sum(1 for r in ins if r['insert'] == k) for k in {r['insert'] for r in ins}})
    entry = [r['entry_truth'] for r in R if r.get('entry_truth')]
    if entry:
        out['entry_truth_mean_mm'] = np.mean(entry, 0).round(2).tolist()
    fr = [f for r in R for f in r.get('frac', [])]
    if fr:
        out['step_moved_fraction_p10_p50_p90'] = np.percentile(fr, [10, 50, 90]).round(2).tolist()
    out['crashed'] = sum(1 for r in R if r.get('stopmsg'))
    out['stop_msgs'] = sorted({str(r['stopmsg'])[:120] for r in R if r.get('stopmsg')})
    print(json.dumps(out, indent=1))
    if A.json:
        Path(A.json).parent.mkdir(parents=True, exist_ok=True)
        Path(A.json).write_text(json.dumps(dict(summary=out, runs=R), indent=0))


if __name__ == '__main__':
    main()
