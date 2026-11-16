"""A/B of the sim release jaw opening in the dry run (docs/real/SIM_RELEASE_PANEL_20261116.md). Sim only.

Each run: scripts/eval/k8_release_dry.py (truth trace of the insertion and the release) around deploy.run_bimanual --dry-run
--stage full --align-mode full, with --sim-release-open-m X (0 = the old 24 mm dry run; 0.027 = the SimIO default).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/sim_release_ab.py run \
        --configs bimanual_align9_all bimanual_align10_all --seeds 7041 7042 --opens 0 0.027 --reps 3 --par 6
    python scripts/eval/sim_release_ab.py table
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/real_align/sim_release'
RUNS = ROOT / 'runs/simrel/dry'


def one(cfg, seed, op, rep):
    tag = f'{cfg}_s{seed}_o{int(round(op * 1e4))}_r{rep}'
    summ = OUT / 'rows' / f'{tag}.json'
    if summ.exists():
        return tag
    summ.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, 'scripts/eval/k8_release_dry.py', '--tag', tag, '--summary', str(summ), '--',
           '--config', f'deploy/config/{cfg}.json', '--stage', 'full', '--align-mode', 'full', '--dry-run', '--seed', str(seed),
           '--out', str(RUNS), '--name', tag, '--sim-release-open-m', str(op)]     # --name: parallel runs get their own folder
    (RUNS / 'logs').mkdir(parents=True, exist_ok=True)
    with open(RUNS / 'logs' / f'{tag}.log', 'w') as f:
        subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), stdout=f, stderr=subprocess.STDOUT)
    return tag


def run(A):
    jobs = list(itertools.product(A.configs, A.seeds, A.opens, range(A.reps)))
    with ThreadPoolExecutor(A.par) as ex:
        for t in ex.map(lambda j: one(*j), jobs):
            print('done', t, flush=True)


def table(A):
    rows = []
    for f in sorted((OUT / 'rows').glob('*.json')):
        for r in json.loads(f.read_text()):
            a = r['argv']
            r['config'] = Path(a[a.index('--config') + 1]).stem
            r['seed'] = int(a[a.index('--seed') + 1])
            r['open_m'] = float(a[a.index('--sim-release-open-m') + 1])
            rows.append(r)
    keys = sorted({(r['config'], r['seed'], r['open_m']) for r in rows})
    out, lines = [], ['| config | seed | release open (mm) | runs | released after a seat | seated at end of insert | seated at release start | seated after release | not seated after release | retreat depth loss (mm, per run) | other outcomes |',
                      '|---|---|---|---|---|---|---|---|---|---|---|']
    for k in keys + sorted({(c, 'all', o) for c, _, o in keys}):
        xs = [r for r in rows if r['config'] == k[0] and r['open_m'] == k[2] and (k[1] == 'all' or r['seed'] == k[1])]
        rel = [r for r in xs if r.get('insert') == 'seated_detected' and r.get('depth_release_start_mm') is not None]   # released after a seat
        s = dict(config=k[0], seed=k[1], open_mm=k[2] * 1e3 if k[2] else 24.0, runs=len(xs), reached_release=len(rel),
                 seated_end_insert=sum(bool(r.get('seated_end_insert')) for r in rel),
                 seated_after_release=sum(bool(r.get('true_seated_after_release')) for r in rel),
                 seated_release_start=sum(bool(r.get('first_unseated_k') != 0) for r in rel),
                 retreat_drop_mm=[round(r['depth_after_open_mm'] - r['depth_after_retreat_mm'], 3) for r in rel if r.get('depth_after_retreat_mm') is not None],
                 stops=sorted(str(r.get('stop') or r.get('insert'))[:40] for r in xs if r not in rel))
        s['not_seated_after_release'] = s['reached_release'] - s['seated_after_release']
        out.append(s)
        lines.append(f"| {s['config']} | {s['seed']} | {s['open_mm']:.0f}{' (old)' if not k[2] else ''} | {s['runs']} | {s['reached_release']} | {s['seated_end_insert']} | "
                     f"{s['seated_release_start']} | {s['seated_after_release']} | {s['not_seated_after_release']} | {s['retreat_drop_mm'] or '-'} | {'; '.join(s['stops']) or '-'} |")
    (OUT / 'ab.json').write_text(json.dumps(dict(groups=out, rows=rows), indent=1))
    (OUT / 'ab.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--configs', nargs='+', default=['bimanual_align9_all', 'bimanual_align10_all'])
    r.add_argument('--seeds', nargs='+', type=int, default=[7041, 7042])
    r.add_argument('--opens', nargs='+', type=float, default=[0.0, 0.027])
    r.add_argument('--reps', type=int, default=3)
    r.add_argument('--par', type=int, default=4)
    sp.add_parser('table')
    A = ap.parse_args()
    run(A) if A.cmd == 'run' else table(A)


if __name__ == '__main__':
    main()
