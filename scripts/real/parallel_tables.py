"""Tables for docs/real/PARALLEL_SCHEDULE_20261115.md from scripts/real/parallel_clearance.py output.

    python scripts/real/parallel_tables.py --dir results/real_align/parallel
Writes tables.json (and prints markdown): clearance per overlap and offset (min, p10, median over runs), the time
saved per offset, the stage timeline (log open, capture start, last move end) and the press budget (latch -> release).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), min=round(float(x.min()), 1), p10=round(float(np.percentile(x, 10)), 1),
                median=round(float(np.median(x)), 1), max=round(float(x.max()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/real_align/parallel')
    a = ap.parse_args()
    d = REPO / a.dir
    rows = json.loads((d / 'clearance_rows.json').read_text())
    tl = json.loads((d / 'timeline.json').read_text())
    out = {}
    # P1 / P2: fixed offsets, plus the serial row per run
    for ov in ('P1 capture', 'P2 home'):
        rs = [r for r in rows if r['overlap'] == ov]
        runs = sorted({r['run'] for r in rs})
        ser = {r['run']: r for r in rs if r['serial']}
        tab = {}
        for off in sorted({r['offset_s'] for r in rs if not r['serial']}):
            q = [r for r in rs if not r['serial'] and r['offset_s'] == off]
            saved = [ser[r['run']]['makespan_s'] - r['makespan_s'] for r in q if r['run'] in ser]
            worst = min(q, key=lambda r: r['arms_mm'])
            tab[f'{off:+.1f}'] = dict(arms_mm=stats([r['arms_mm'] for r in q]), saved_s=stats(saved),
                                      worst=dict(run=worst['run'], pair=worst['pair'], t_min_s=worst['t_min_s']),
                                      pass50=sum(r['arms_mm'] >= 50 for r in q))
        q = list(ser.values())
        tab['serial'] = dict(arms_mm=stats([r['arms_mm'] for r in q]), makespan_s=stats([r['makespan_s'] for r in q]),
                             pass50=sum(r['arms_mm'] >= 50 for r in q))
        objs = [v for r in rs for v in (r.get('pc_right_mm'), r.get('pc_left_mm')) if v is not None]
        rds = [v for r in rs for v in (r.get('reader_right_mm'), r.get('reader_left_mm')) if v is not None]
        tab['objects'] = dict(pc_mm=stats(objs) if objs else None, reader_mm=stats(rds) if rds else None)
        out[ov] = dict(runs=len(runs), table=tab)
    # P3: per run three offsets (start of the pick, 6 s before its end, swing ends with the pick) + today's serial swing
    rs = [r for r in rows if r['overlap'] == 'P3 swing in pick']
    by = {}
    for r in rs:
        by.setdefault(r['run'], []).append(r)
    kinds = {'at pick start': [], '6 s before pick end': [], 'ends at pick end': [], 'serial (today)': []}
    for run, q in by.items():
        par = sorted([r for r in q if not r['serial']], key=lambda r: r['offset_s'])
        ser = [r for r in q if r['serial']]
        if len(par) == 3:
            for k, r in zip(('at pick start', '6 s before pick end', 'ends at pick end'), par):
                kinds[k].append(r)
        kinds['serial (today)'] += ser
    out['P3 swing in pick'] = {k: dict(arms_mm=stats([r['arms_mm'] for r in v]),
                                       pass50=sum(r['arms_mm'] >= 50 for r in v),
                                       view_runs=sum(1 for r in v if r.get('view_ticks')),
                                       view_bodies=sorted({b for r in v for b in (r.get('view_bodies') or [])}),
                                       pc_left_mm=stats([r['pc_left_mm'] for r in v if r.get('pc_left_mm') is not None]) if v and v[0].get('pc_left_mm') is not None else None)
                               for k, v in kinds.items() if v}
    # timeline and press budget
    full = [t for t in tl if t.get('capture_to_end_s') and t.get('home_serial_offset_s')]
    out['timeline'] = dict(runs_with_capture_and_home=len(full),
                           capture_to_end_s=stats([t['capture_to_end_s'] for t in full]) if full else None,
                           open_to_end_s=stats([t['open_to_end_s'] for t in full]) if full else None,
                           open_minus_capture_s=stats([t['open_to_end_s'] - t['capture_to_end_s'] for t in full]) if full else None,
                           connect_s=stats([t['t_connect_s'] for t in full]) if full else None)
    pr = [t for t in tl if t.get('press_s') is not None]
    out['press'] = dict(press_s=stats([t['press_s'] for t in pr]) if pr else None,
                        runs=[(t['run'], t['press_s']) for t in pr],
                        pick_s=stats([t['pick_s'] for t in tl if t.get('pick_s')]),
                        swing_s=stats([t['swing_s'] for t in tl if t.get('swing_s')]))
    # press clock (E-A3): latch -> end of the release stage, minus the align stage's press-off intervals
    RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
    pc = []
    for t in tl:
        f = RUNS / t['run'] / 'events.jsonl'
        if not f.exists():
            continue
        ev = [json.loads(x) for x in open(f)]
        lat = next((e for e in ev if e['kind'] == 'hold_result' and e.get('latched')), None)
        if lat is None:
            continue
        st = [e for e in ev if e['kind'] == 'stage']
        t_home = next((e['t'] for e in st if e['stage'] == '7 home'), None)
        if t_home is None:
            continue
        on, t_on, total = True, lat['t'], 0.
        for e in ev:
            if e['t'] <= lat['t'] or e['t'] >= t_home:
                continue
            if e['kind'] == 'align_press_off' and on:
                total += e['t'] - t_on
                on = False
            elif e['kind'] == 'align_press_on' and not on:
                on, t_on = True, e['t']
        if on:
            total += t_home - t_on
        pc.append(dict(run=t['run'], press_on_s=round(total, 2), latch_to_home_s=round(t_home - lat['t'], 2),
                       press_off_used=any(e['kind'] == 'align_press_off' for e in ev), pick_s=t.get('pick_s')))
    out['press_clock'] = dict(all=stats([x['press_on_s'] for x in pc]) if pc else None,
                              with_press_off=stats([x['press_on_s'] for x in pc if x['press_off_used']]) if any(x['press_off_used'] for x in pc) else None,
                              without_press_off=stats([x['press_on_s'] for x in pc if not x['press_off_used']]) if any(not x['press_off_used'] for x in pc) else None,
                              runs=pc)
    out['per_run'] = tl
    (d / 'tables.json').write_text(json.dumps(out, indent=1))
    for k in ('P1 capture', 'P2 home'):
        print(f"\n### {k} (runs {out[k]['runs']})\n| offset s | n | arms min mm | p10 | median | pass >= 50 mm | saved s (median) | worst pair |\n|---|---|---|---|---|---|---|---|")
        for off, v in out[k]['table'].items():
            if off == 'objects':
                continue
            s = v['arms_mm']
            sv = v.get('saved_s', {}).get('median', '') if 'saved_s' in v else '0'
            wp = '/'.join(v['worst']['pair'] or []) if 'worst' in v else ''
            print(f"| {off} | {s['n']} | {s['min']} | {s['p10']} | {s['median']} | {v['pass50']} of {s['n']} | {sv} | {wp} |")
        print('objects', out[k]['table']['objects'])
    print('\n### P3')
    for k, v in out['P3 swing in pick'].items():
        print(k, v)
    print('\ntimeline', out['timeline'])
    print('press', {k: v for k, v in out['press'].items() if k != 'runs'})
    print('press clock', {k: v for k, v in out['press_clock'].items() if k != 'runs'})


if __name__ == '__main__':
    main()
