"""One run of the align stage, from its log: every measurement, step, decision; then the hand-over and the insertion guards. No motion.

    python scripts/real/align_log.py data/real_runs/<date>/full_<id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def fmt(v, n=2):
    return 'none' if v is None else f'{v:+.{n}f}'


def main(run):
    run = Path(run)
    ev = [json.loads(x) for x in open(run / 'events.jsonl')]
    wind = [e for e in ev if e['kind'] == 'align_windup_at_entry']
    if wind:
        print('target lead at entry (mrad):', np.round(np.array(wind[0]['lead']) * 1e3, 1).tolist())
    print('cyc  t_s   ok across    up    gap  conf spread | tip against entry (mm)      | decision')
    for e in ev:
        if e['kind'] == 'align_measure':
            nok = sum(1 for f in e['frames'] if f['ok'])
            print(f"{e['cycle']:3d} {e['t_stage']:5.1f} {nok}/{len(e['frames'])} {fmt(e['across'])} {fmt(e['up'])} {fmt(e['gap'])} {e['conf']:.2f} {fmt(e['spread'])} | {np.round(e['tip_rel'], 2).tolist()} | {e['decision']} {np.round(e['delta'], 2).tolist()} {e['decision_reason'] or ''}"
                  + (f" | truth {np.round(e['truth_mm'], 2).tolist()}" if e.get('truth_mm') else ''))
        elif e['kind'] == 'align_step':
            print(f"       step asked {np.round(e['delta'], 2).tolist()} moved {np.round(e['moved'], 2).tolist()} fraction {e['frac']:.2f} reached {e['reached']} ticks {e['ticks']} lead(mrad) {np.round(np.array(e['lead']) * 1e3, 1).tolist()}")
    st = [e['frac'] for e in ev if e['kind'] == 'align_step']
    if st:
        print('moved fraction p10 p50 p90:', np.percentile(st, [10, 50, 90]).round(2).tolist(), 'n', len(st), 'below 0.25:', sum(1 for f in st if f < .25), 'above 3:', sum(1 for f in st if f > 3))
    for k in ('align_result', 'align_handover', 'align_inhand', 'insert_stop', 'seated_detected', 'insert_result', 'back_off'):
        for e in ev:
            if e['kind'] == k:
                print(k, json.dumps({a: b for a, b in e.items() if a not in ('kind', 't', 'cfg')}))
    if (run / 'guard.npz').exists():
        g = np.load(run / 'guard.npz')
        r = g['rel_m']
        print(f"insertion: tip against the hand-over, across {r[:, 0].min():+.1f}..{r[:, 0].max():+.1f}, up {r[:, 1].min():+.1f}..{r[:, 1].max():+.1f}, in {-r[:, 2].max():+.1f}..{-r[:, 2].min():+.1f} mm; ticks with the bound active {int(g['clamped'][-1])}; depth2 end {g['depth2_mm'][-1]:+.2f} mm")


if __name__ == '__main__':
    for x in sys.argv[1:]:
        main(x)
