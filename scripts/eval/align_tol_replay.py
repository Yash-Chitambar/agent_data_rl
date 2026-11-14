"""Offline replay of real alignment stages (docs/real/CONTACT_SENS_20261115.md, test 2): when would the stage have stopped with
a looser lateral tolerance? Read-only on the real run logs (events.jsonl, 'align_measure' records). No robot, no sim.

Per run and tolerance: the first cycle after the last slope probe where |across|, |up| <= tol (fused, mm), |gap - 6| <= 1 mm and
|rx|, |ry| <= 1 deg (rv_fused), followed by one confirm cycle (confirm_n 2: the stage repeats the measurement without a
motion). Stop time = t_stage of that confirm cycle. This is an upper bound of the saving: the real logic would have taken the
same steps up to that cycle (same measurements), so the replay is exact up to it.

    python scripts/eval/align_tol_replay.py --out results/real_align/contact_sens/align_tol_replay.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
DATES = ('2026-11-10', '2026-11-11', '2026-11-13')


def cycles(run):
    out = []
    for line in (run / 'events.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('kind') == 'align_measure' and e.get('fused') is not None:
            out.append(e)
    return out


def first_stop(cyc, tol, gap_tol=1.0, ang_tol=1.0):
    last_probe = max([i for i, e in enumerate(cyc) if e.get('decision_reason') in ('probe', 'extra probe')], default=-1)
    for i in range(last_probe + 1, len(cyc)):
        f = np.asarray(cyc[i]['fused'], float)
        rv = cyc[i].get('rv_fused')
        ok = (np.abs(f[:2]) <= tol).all() and abs(f[2] - 6.0) <= gap_tol and (rv is None or (np.abs(np.asarray(rv)[:2]) <= ang_tol).all())
        if ok:
            j = min(i + 1, len(cyc) - 1)       # one confirm measurement
            return j, float(cyc[j]['t_stage'])
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tols', default='1,1.5,2')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    tols = [float(x) for x in A.tols.split(',')]
    rows = []
    for d in DATES:
        for run in sorted((RUNS / d).glob('full_*')):
            if '_dry_' in run.name or not (run / 'events.jsonl').exists():
                continue
            cyc = cycles(run)
            done = [i for i, e in enumerate(cyc) if e.get('decision') == 'done']
            if not done:
                continue
            i_done = done[0]        # first hand-over decision (before the press-on re-check)
            r = dict(run=f'{d}/{run.name}', n_cycles=len(cyc), done_cycle=i_done, t_done_s=float(cyc[i_done]['t_stage']),
                     fused_at_done=np.round(cyc[i_done]['fused'], 2).tolist())
            for t in tols:
                j, ts = first_stop(cyc, t)
                r[f'tol{t:g}'] = dict(cycle=j, t_s=ts, fused=None if j is None else np.round(cyc[j]['fused'], 2).tolist())
            rows.append(r)
    out = Path(A.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r['run'], 'done', r['done_cycle'], f"{r['t_done_s']:.1f} s", r['fused_at_done'], ' | '.join(
            f"tol {t:g}: {r[f'tol{t:g}']['cycle']} {r[f'tol{t:g}']['t_s']}" for t in tols))


if __name__ == '__main__':
    main()
