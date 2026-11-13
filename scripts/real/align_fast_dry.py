"""Table of align-stage dry runs (sim): cycles, sim time (control ticks), steps and timeouts, hand-over truth, insertion. No motion.

    python scripts/real/align_fast_dry.py data/real_runs/<date>/*t2_fast_* [--json out.json]

Sim time counts control ticks (20 ms); the sim measurement takes no wall time, so the real stage is longer (docs/real/ALIGN_FAST_20261113.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def row(run):
    ev = [json.loads(x) for x in open(Path(run) / 'events.jsonl')]
    ms = [e for e in ev if e['kind'] == 'align_measure']
    st = [e for e in ev if e['kind'] == 'align_step']
    res = [e for e in ev if e['kind'] == 'align_result']
    ho = [e for e in ev if e['kind'] == 'align_handover']
    ins = [e for e in ev if e['kind'] == 'insert_result']
    start = [e for e in ev if e['kind'] == 'align_start']
    if not res:
        return dict(run=Path(run).name, done=False)
    r = res[0]
    se = [e for e in ev if e['kind'] == 'align_servo_end']
    if se:                                  # servo stage: time, truth and tip speed (0.1 s window, believed FK)
        sv = np.load(Path(run) / 'servo.npz') if (Path(run) / 'servo.npz').exists() else None
        sp = None if sv is None else np.percentile(sv['speed'], [50, 99]).round(1).tolist()
        return dict(run=Path(run).name, done=True, servo=True, ok=r['ok'], reason=r['reason'], seconds=[e['seconds'] for e in se], frames=[e['frames'] for e in se],
                    handover_truth=se[-1].get('truth_mm'), est=se[-1].get('est'), tip_speed_p50_p99=sp)
    pre = [m for m in ms if m['cycle'] < 40]
    post = [m for m in ms if m['cycle'] >= 40]
    t_pre = pre[-1]['t_stage'] if pre else None
    t_all = (r['tick'] - (ms[0]['tick'] - int(round(ms[0]['t_stage'] / .02)))) * .02 if ms else None
    last = (post or pre)[-1]
    truth = last.get('truth_mm')
    out = dict(run=Path(run).name, done=True, ok=r['ok'], reason=r['reason'], cycles=len(pre), post_cycles=len(post), probes=sum(m['decision_reason'] == 'probe' for m in pre),
               t_pre_s=t_pre, t_stage_s=t_all, steps=len(st), timeouts=sum(not s['reached'] for s in st), step_ticks_p50=float(np.median([s['ticks'] for s in st])) if st else None,
               handover_est=ho[0].get('lat_mm', None) and [*ho[0]['lat_mm'], ho[0]['gap_mm']] if ho else None, handover_truth=truth,
               entry_gap=pre[0]['gap'] if pre else None, slope=r.get('slope'))
    if ins:
        out.update(insert=ins[0].get('outcome'), true_seated_end=ins[0].get('true_seated_end'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--json')
    a = ap.parse_args()
    rows = [row(r) for r in a.runs]
    for x in rows:
        if not x['done']:
            print(x['run'], 'no align result')
            continue
        tr = x['handover_truth']
        if x.get('servo'):
            print(f"{x['run'][-28:]:28s} servo ok {x['ok']} {x['seconds']} s, frames {x['frames']}, tip speed p50/p99 {x['tip_speed_p50_p99']} mm/s | truth {None if tr is None else np.round(tr, 2).tolist()} | {x['reason'][:50]}")
            continue
        print(f"{x['run'][-28:]:28s} ok {x['ok']!s:5s} cyc {x['cycles']:2d}+{x['post_cycles']} probes {x['probes']} steps {x['steps']:2d} to {x['timeouts']:2d} step p50 {x['step_ticks_p50']} ticks "
              f"t_pre {x['t_pre_s']:.1f} t_stage {x['t_stage_s']:.1f} s | truth {None if tr is None else np.round(tr, 2).tolist()} | {x.get('insert')} {x.get('true_seated_end')} | {x['reason'][:60]}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
