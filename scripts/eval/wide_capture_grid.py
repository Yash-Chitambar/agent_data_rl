"""Capture-stage dry runs (deploy.run_bimanual --dry-run --stage capture) over a PC pose grid: the ZED gate net, the
edge fit (on the sim render with the dark edge-fit panel, docs/real/SIM_RELEASE_PANEL_20261116.md) and the reader
estimate, per pose. No arm motion beyond the capture pose in the chain; sim only.
docs/real/WIDE_PLACEMENT_READY_20261116.md.

    # one pose (worker)
    PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/wide_capture_grid.py one --pc 0.231 0.394 4 --reader 0.131 0.448 92 \
        --seed 7043 --config deploy/config/bimanual_align11_wide_k7.json --out results/placement/wide/capture --tag t
    # the grid: dx, dy in {-25, -12, 0, 12, 25} mm, dyaw in {-15, -8, 0, 8, 15} deg around (231, 394) mm, 4 deg
    PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/wide_capture_grid.py grid --workers 6 --config ... --out ...
    PYTHONPATH=. python scripts/eval/wide_capture_grid.py table --out results/placement/wide/capture
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
X0, Y0, T0 = 0.231, 0.394, 4.0


def pg():
    spec = importlib.util.spec_from_file_location('placement_grid', ROOT / 'scripts/eval/placement_grid.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def one(a):
    pg().patch_layout(a.pc, a.reader)
    from deploy import run_bimanual as RB
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / 'point.json').write_text(json.dumps(dict(pc=a.pc, reader=a.reader, seed=a.seed, config=a.config)))
    argv = ['--dry-run', '--stage', 'capture', '--seed', str(a.seed), '--config', a.config, '--yes', '--name', a.tag,
            '--out', str(out), '--no-home']
    try:
        res = RB.main(argv)
        (out / 'result.json').write_text(json.dumps(res, default=str))
    except BaseException as e:  # noqa: BLE001
        (out / 'result.json').write_text(json.dumps({'exception': repr(e)}))
        raise


def grid(a):
    pts = []
    for dx, dy, dt in itertools.product((-25, -12, 0, 12, 25), (-25, -12, 0, 12, 25), (-15, -8, 0, 8, 15)):
        pts.append(dict(tag=f'cap{dx:+d}_{dy:+d}_{dt:+d}', pc=[round(X0 + dx / 1e3, 4), round(Y0 + dy / 1e3, 4), T0 + dt]))
    procs, todo = [], [p for p in pts if not (Path(a.out) / p['tag'] / 'result.json').exists()]
    logd = Path(a.out) / '_logs'
    logd.mkdir(parents=True, exist_ok=True)
    while todo or procs:
        while todo and len(procs) < a.workers:
            p = todo.pop(0)
            cmd = [sys.executable, __file__, 'one', '--pc', *map(str, p['pc']), '--reader', *map(str, a.reader), '--seed', str(a.seed),
                   '--config', a.config, '--out', a.out, '--tag', p['tag']]
            f = open(logd / f"{p['tag']}.log", 'w')
            procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy()), f, p['tag']))
        time.sleep(2)
        for pr in list(procs):
            if pr[0].poll() is not None:
                pr[1].close()
                procs.remove(pr)


def table(a):
    rows = []
    for d in sorted(Path(a.out).iterdir()):
        if not (d / 'point.json').exists():
            continue
        p = json.loads((d / 'point.json').read_text())
        evs = []
        for f in d.rglob('events.jsonl'):
            evs = [json.loads(x) for x in open(f)]
            break
        z = next((e for e in evs if e['kind'] == 'estimate' and e.get('est') == 'zed_pc'), None)
        rd = next((e for e in evs if e['kind'] == 'estimate' and e.get('est') == 'zed_reader'), None)
        if z is None:
            rows.append(dict(tag=d.name, pc=p['pc'], error='no estimate'))
            continue
        net, ef = z.get('net') or {}, z.get('edge_fit') or {}
        rows.append(dict(tag=d.name, pc=p['pc'], ok=z.get('ok'), gate_net_ok=net.get('ok'), gate_net_err_mm=net.get('err_xy_mm'),
                         gate_yaw_err_deg=(net.get('yaw_deg', 0.) - z['true_yaw_deg']), mahalanobis=z.get('mahalanobis'),
                         disagree_xy_mm=z.get('disagree_xy_mm'), presence=z.get('presence'),
                         fit_ok=ef.get('ok'), fit_why=ef.get('why'), fit_err_mm=ef.get('err_mm'), fit_err_yaw_deg=ef.get('err_yaw_deg'),
                         shift_mm=ef.get('shift_mm'), edges_used=ef.get('edges_used'), rms_px=ef.get('inlier_rms_px'),
                         used_err_mm=z.get('err_xy_mm'), reader_ok=None if rd is None else rd.get('ok'),
                         reader_err_mm=None if rd is None else rd.get('err_mm')))
    Path(a.out, 'table.json').write_text(json.dumps(rows, indent=1, default=str))
    for r in rows:
        print(r['tag'], r.get('ok'), 'net', r.get('gate_net_ok'), r.get('gate_net_err_mm'), 'fit', r.get('fit_ok'), r.get('fit_err_mm'), r.get('fit_why'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('one', 'grid', 'table'))
    ap.add_argument('--pc', type=float, nargs=3)
    ap.add_argument('--reader', type=float, nargs=3, default=[0.131, 0.448, 92.0])
    ap.add_argument('--seed', type=int, default=7043)
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align11_wide_k7.json'))
    ap.add_argument('--out', default=str(ROOT / 'results/placement/wide/capture'))
    ap.add_argument('--tag', default='p')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    {'one': one, 'grid': grid, 'table': table}[a.mode](a)


if __name__ == '__main__':
    main()
