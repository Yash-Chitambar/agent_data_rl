"""One table per test batch from the run logs: outcome, roll at the close, rise from close to stall, grip readings.

    .venv/bin/python scripts/real/pick_batch_table.py data/real_runs/2026-11-03                 # every batch of that day
    .venv/bin/python scripts/real/pick_batch_table.py data/real_runs/2026-11-03 --batch A --csv batchA.csv

Reads events.jsonl and meta.json of pick_* and full_* runs with pick.grasp_v3 (dry runs only with --dry).
Outcome = the run's `pick_outcome` event (held / lost_in_lift / closed_empty); `operator` = what you typed at the end.
Wilson 95 % CI on the held count: ten tries show a big jump, not 80 % against 90 %.
"""
import argparse
import csv
import glob
import json
import math
import os


def wilson(k, n, z=1.96):
    if not n:
        return 0., 0.
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def read_run(d):
    ev = [json.loads(x) for x in open(os.path.join(d, 'events.jsonl'))]
    meta = json.load(open(os.path.join(d, 'meta.json')))
    g = {}
    for e in ev:
        if e.get('kind') == 'grasp_v3':
            g.setdefault(e['event'], []).append(e)     # all attempts; the last close counts
    pout = next((e for e in reversed(ev) if e.get('kind') == 'pick_outcome'), None)
    if pout is None:
        return None
    last = lambda name, key: (g.get(name) or [{}])[-1].get(key)   # noqa: E731
    cfg = meta.get('config', {})
    out = meta.get('outcome') or {}
    return dict(run=os.path.basename(d), batch=cfg.get('pick', {}).get('batch') or '-', open_mm=round(cfg.get('gripper', {}).get('right_approach_open_m', float('nan')) * 1e3, 1),
                outcome=pout.get('outcome'), attempts=pout.get('attempts'),
                roll_close_deg=last('close_start', 'roll_deg') if last('close_start', 'roll_deg') is not None else last('squeeze_start', 'roll_deg'),
                rise_to_stall_mm=last('close_stall', 'rise_mm') if last('close_stall', 'rise_mm') is not None else last('squeeze_end', 'rise_mm') if last('squeeze_end', 'rise_mm') is not None else last('close_fell_through', 'rise_mm'),
                grip_stall=last('close_stall', 'reading'), grip_squeeze=last('squeeze_end', 'reading'), grip_test=last('test_result', 'reading'),
                grip_end=last('done', 'reading') if last('done', 'reading') is not None else last('lost_in_lift', 'reading'),
                touch_d_mm=last('touchdown', 'd_mm'), tip_fk_mm=last('touchdown', 'tip_fk_mm'), operator=out.get('label') if isinstance(out, dict) else out,
                notes=(out.get('failure_labels') if isinstance(out, dict) else '') or meta.get('operator_notes', ''))


def fmt(v):
    return '' if v is None else (f'{v:.3g}' if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--batch', default=None)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--csv', default='')
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.root, '*_*'))):
        if not os.path.exists(os.path.join(d, 'events.jsonl')) or (('_dry_' in d) != a.dry):
            continue
        try:
            r = read_run(d)
        except (OSError, ValueError, KeyError):
            continue
        if r is not None and (a.batch is None or r['batch'] == a.batch):
            rows.append(r)
    cols = ('run', 'open_mm', 'outcome', 'attempts', 'roll_close_deg', 'rise_to_stall_mm', 'grip_stall', 'grip_squeeze', 'grip_test', 'grip_end', 'touch_d_mm', 'tip_fk_mm', 'operator', 'notes')
    for b in sorted({r['batch'] for r in rows}):
        sel = [r for r in rows if r['batch'] == b]
        print(f'\n## Batch {b}\n\n| try | ' + ' | '.join(cols) + ' |\n|---|' + '---|' * len(cols))
        for i, r in enumerate(sel, 1):
            print(f'| {i} | ' + ' | '.join(fmt(r[c]) for c in cols) + ' |')
        k, n = sum(r['outcome'] == 'held' for r in sel), len(sel)
        lo, hi = wilson(k, n)
        cnt = {o: sum(r['outcome'] == o for r in sel) for o in sorted({r['outcome'] for r in sel})}
        print(f'\nheld {k} of {n} (Wilson 95 % {lo * 100:.0f}-{hi * 100:.0f} %); {cnt}')
    if a.csv and rows:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
