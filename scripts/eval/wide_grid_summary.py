"""Tables for the wide-placement grid (docs/real/WIDE_PLACEMENT_READY_20261116.md).

Reads the run folders of scripts/eval/placement_grid.py (classify() of that script) and adds the capture-stage edge-fit
record, the yaw errors of the gate net (zed_v5 in the wide configs), the seat state at the end of the insertion
(before the release: the sim release is a known artefact, K8_20261115.md) and after the release, and the clearance
minima of scripts/eval/wide_clearance.py. Writes rows.json and prints markdown tables.

    PYTHONPATH=. python scripts/eval/wide_grid_summary.py --grid results/placement/wide/grid \
        --clearance results/placement/wide/clearance.json --out results/placement/wide/summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
X0, Y0, T0 = 0.231, 0.394, 4.0          # K7 box centre = the 2026-11-10/11 v1 real spot (layout task_layout_real_v2)
RD0 = (0.131, 0.448, 92.0)
# Seat at the end of the insertion: the sim's strict seat (geometric_seated: within 0.25 mm of full depth, tilt <= 2 deg)
# or a true plug depth >= 5.3 mm (within about 0.5 mm of full: the operator's 'straight, no band' class). Both logged.
DEPTH_IN_MM = 5.3


def pg():
    spec = importlib.util.spec_from_file_location('placement_grid', ROOT / 'scripts/eval/placement_grid.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def node(r):
    """First failing node of the chain, in chain order."""
    if r.get('exception'):
        return 'exception'
    st = r['stage']
    if st == 'zed':
        ef = r.get('edge_fit') or {}
        return 'zed: fit gate' if r.get('gate_net_ok') and not ef.get('ok', True) else 'zed: gate net'
    if st == 'reader_zed':
        return 'reader ZED gate'
    if st in ('pick', 'hold'):
        return st
    if st == 'transport/approach':
        return 'approach'
    if st == 'align':
        return 'align'
    if st == 'insert':
        return 'insert'
    if r.get('insert_outcome') in ('seated', 'seated_detected'):
        if r.get('true_seated_end') is False and not (r.get('true_depth_end_mm') or 0) >= DEPTH_IN_MM:
            return 'insert (detected, plug short)'
        if r.get('true_seated') is False:
            return 'release'
        return 'ok'
    return str(st)


def one(d, P):
    p = json.loads((d / 'point.json').read_text())
    r = dict(tag=d.name, **p, **P.classify(d))
    evs = []
    for f in d.rglob('events.jsonl'):
        evs = [json.loads(x) for x in open(f)]
        break
    z = next((e for e in evs if e['kind'] == 'estimate' and e.get('est') == 'zed_pc'), None)
    if z:
        net = z.get('net') or {}
        ef = z.get('edge_fit') or {}
        r.update(gate_net_ok=net.get('ok', z.get('ok')), gate_net_err_mm=net.get('err_xy_mm', z.get('err_xy_mm')),
                 gate_yaw_err_deg=(net.get('yaw_deg', z.get('yaw_deg')) - z['true_yaw_deg']) if 'true_yaw_deg' in z else None,
                 mahalanobis=z.get('mahalanobis'), disagree_xy_mm=z.get('disagree_xy_mm'), presence=z.get('presence'),
                 edge_fit=dict(ok=ef.get('ok'), why=ef.get('why'), shift_mm=ef.get('shift_mm'), err_xy_mm=ef.get('err_xy_mm')),
                 used_err_mm=z.get('err_xy_mm'))
    ir = next((e for e in evs if e['kind'] == 'insert_result'), {})
    r.update(true_seated_end=ir.get('true_seated_end'), true_depth_end_mm=ir.get('true_depth_mm_end'), t_seat_s=ir.get('t_seat_s'))
    hr = next((e for e in evs if e['kind'] == 'hold_result'), {})
    r.update(hold_point=hr.get('point'), hold_yaw=hr.get('yaw_deg'))
    r['release_widened'] = any(e['kind'] == 'sim_stage' and e.get('key') == 'sim.release_open_m' for e in evs)
    r['node'] = node(r)
    r['d'] = [round((p['pc'][0] - X0) * 1e3), round((p['pc'][1] - Y0) * 1e3), round(p['pc'][2] - T0)]
    r['rd'] = [round((p['reader'][0] - RD0[0]) * 1e3), round((p['reader'][1] - RD0[1]) * 1e3), round(p['reader'][2])]
    r['policy'] = 'k7' if 'k7' in Path(p['config']).name else ('k4' if 'k4' in Path(p['config']).name else Path(p['config']).stem)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', default=str(ROOT / 'results/placement/wide/grid'))
    ap.add_argument('--clearance', default=None)
    ap.add_argument('--out', default=str(ROOT / 'results/placement/wide/summary.json'))
    a = ap.parse_args()
    P = pg()
    rows = [one(d, P) for d in sorted(Path(a.grid).iterdir()) if (d / 'point.json').exists() and (d / 'result.json').exists()]
    cl = {}
    if a.clearance and Path(a.clearance).exists():
        cl = {r['tag']: r['mins'] for r in json.loads(Path(a.clearance).read_text())['rows']}
    for r in rows:
        r['clearance'] = cl.get(r['tag'])
    Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
    f = lambda v, n=1: '-' if v is None else f'{v:.{n}f}'  # noqa: E731
    print('| run | PC d (x, y mm; yaw deg) | reader d (x, y mm; yaw) | gate net ok / err mm / yaw err deg | fit shift mm | node | align s | seat end | seated after release |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rows, key=lambda r: (r['policy'], r['tag'])):
        print(f"| {r['tag']} | {r['d']} | {r['rd']} | {r.get('gate_net_ok')} / {f(r.get('gate_net_err_mm'))} / {f(r.get('gate_yaw_err_deg'))} | "
              f"{f((r.get('edge_fit') or {}).get('shift_mm'))} | {r['node']} | {f(r.get('align_s'))} | {r.get('true_seated_end')} | {r.get('true_seated')} |")
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        agg[r['policy']][r['node']] += 1
    print()
    for k, v in agg.items():
        print(k, dict(v), 'n =', sum(v.values()))


if __name__ == '__main__':
    main()
