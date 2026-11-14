"""Paired dry runs, serial vs parallel schedule (docs/real/PARALLEL_SCHEDULE_20261115.md).

    run:     python scripts/eval/parallel_dry_ab.py run --configs serial=deploy/config/bimanual_align10_all.json \
                 par=deploy/config/bimanual_align11_par_home_capture.json --seeds 7041-7070 --jobs 4 --out runs/parallel_dry
    summary: python scripts/eval/parallel_dry_ab.py summary --out runs/parallel_dry --json results/real_align/parallel/dry_ab.json

Each run: python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed S --config C (sim only; the
dry run teleports the slow moves, so the schedule changes the plan check and logs, not the sim physics).
The summary reads meta.json and events.jsonl of every run folder: outcome, seat, depth, hand-over error, gate values,
sim ticks, and the parallel_plan events (planned arm-arm clearance, serial vs parallel move time).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = '/home/yashchitambar/peg_test_bimanual/.venv/bin/python'


def seeds_of(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run(a):
    cfgs = dict(x.split('=', 1) for x in a.configs)
    jobs = [(tag, c, s) for s in seeds_of(a.seeds) for tag, c in cfgs.items()]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), MUJOCO_GL='egl')

    def one(j):
        tag, c, s = j
        d = out / tag
        d.mkdir(exist_ok=True)
        if any(x.is_dir() and (x / 'meta.json').exists() for x in d.glob(f'*/*_{tag}_{s}')):
            return j, 'skip'
        cmd = [PY, '-m', 'deploy.run_bimanual', '--dry-run', '--stage', 'full', '--align-mode', 'full', '--seed', str(s),
               '--config', c, '--name', f'{tag}_{s}', '--out', str(d)]
        with open(d / f'{tag}_{s}.log', 'w') as f:
            r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        return j, r.returncode

    with ThreadPoolExecutor(a.jobs) as ex:
        for j, rc in ex.map(one, jobs):
            print(j, rc, flush=True)


def one_summary(d):
    m = json.loads((d / 'meta.json').read_text())
    r = m.get('result') or {}
    res = r.get('results') or {}
    ev = [json.loads(x) for x in open(d / 'events.jsonl')]
    al = res.get('align') or {}
    ho = al.get('handover') or {}
    ins = res.get('insert') or {}
    zed = next((e for e in ev if e['kind'] == 'estimate' and e.get('est') == 'zed_pc'), {})
    stop = next((e['reason'] for e in ev if e['kind'] == 'stopped'), None)
    plans = [e for e in ev if e['kind'] == 'parallel_plan']
    lat = ho.get('lat_mm')
    return dict(
        run=d.name, seed=m.get('args', {}).get('seed'), completed=r.get('completed'), stop=stop,
        zed_ok=res.get('zed_ok'), zed_err_mm=zed.get('err_xy_mm'), pick_ok=res.get('pick_ok'),
        align_ok=al.get('ok'), align_s=al.get('seconds'), align_cycles=al.get('cycles'),
        handover_lat_mm=None if lat is None else float(np.hypot(*lat[:2])), handover_gap_mm=ho.get('gap_mm'),
        insert=ins.get('outcome'), t_seat_s=ins.get('t_seat_s'), rest_depth_mm=ins.get('rest_depth_mm'),
        seated_after_release=res.get('true_seated_after_release'),
        end_seated=(r.get('sim_truth_end') or {}).get('seated'), end_depth_mm=(r.get('sim_truth_end') or {}).get('tip_depth_mm'),
        sim_s=None if r.get('ticks') is None else r['ticks'] * 0.02,
        plans=[{k: p.get(k) for k in ('label', 'min_mm', 'pair', 'ok', 'serial_s', 'parallel_s', 'move_s', 'lead_s', 'first')} for p in plans])


def summary(a):
    out = ROOT / a.out
    rows = []
    for tagdir in sorted(p for p in out.iterdir() if p.is_dir()):
        for d in sorted(p for p in tagdir.glob('*/*') if p.is_dir() and (p / 'meta.json').exists()):
            try:
                x = one_summary(d)
            except Exception as e:  # noqa: BLE001
                x = dict(run=d.name, error=f'{type(e).__name__}: {e}')
            x['tag'] = tagdir.name
            rows.append(x)
    by = {}
    for x in rows:
        by.setdefault(x['tag'], {})[x.get('seed')] = x
    tags = list(by)
    agg = {}
    for t in tags:
        xs = list(by[t].values())

        def vals(k):
            return [x[k] for x in xs if x.get(k) is not None]

        def med(k):
            v = vals(k)
            return None if not v else round(float(np.median(v)), 3)
        agg[t] = dict(n=len(xs), completed=sum(bool(x.get('completed')) for x in xs),
                      stops=sorted({(x.get('stop') or '')[:80] for x in xs if x.get('stop')}),
                      zed_ok=sum(bool(x.get('zed_ok')) for x in xs), pick_ok=sum(bool(x.get('pick_ok')) for x in xs),
                      align_ok=sum(bool(x.get('align_ok')) for x in xs),
                      seated_detected=sum(x.get('insert') == 'seated_detected' for x in xs),
                      seated_after_release=sum(bool(x.get('seated_after_release')) for x in xs),
                      end_seated=sum(bool(x.get('end_seated')) for x in xs),
                      handover_lat_mm_median=med('handover_lat_mm'), handover_gap_mm_median=med('handover_gap_mm'),
                      zed_err_mm_median=med('zed_err_mm'), align_s_median=med('align_s'), rest_depth_mm_median=med('rest_depth_mm'),
                      end_depth_mm_median=med('end_depth_mm'), t_seat_s_median=med('t_seat_s'), sim_s_median=med('sim_s'))
        pl = [p for x in xs for p in x.get('plans', [])]
        for lab in sorted({p['label'] for p in pl}):
            q = [p for p in pl if p['label'] == lab]
            agg[t]['plan ' + lab] = dict(n=len(q), ok=sum(bool(p['ok']) for p in q),
                                         min_mm=[min(p['min_mm'] for p in q), round(float(np.median([p['min_mm'] for p in q])), 1)],
                                         saved_s_median=round(float(np.median([p['serial_s'] - p['parallel_s'] for p in q])), 2))
    # paired: seeds present in both tags
    paired = {}
    if len(tags) == 2:
        t0, t1 = tags
        common = sorted(set(by[t0]) & set(by[t1]))
        same = lambda k: sum(by[t0][s].get(k) == by[t1][s].get(k) for s in common)  # noqa: E731
        paired = dict(tags=[t0, t1], n=len(common), same_completed=same('completed'), same_insert=same('insert'),
                      same_seated_after_release=same('seated_after_release'), same_pick=same('pick_ok'),
                      same_align_ok=same('align_ok'),
                      differ=[dict(seed=s, **{t: {k: by[t][s].get(k) for k in ('insert', 'seated_after_release', 'stop', 'handover_lat_mm', 'rest_depth_mm')} for t in (t0, t1)})
                              for s in common if any(by[t0][s].get(k) != by[t1][s].get(k) for k in ('insert', 'seated_after_release', 'completed', 'pick_ok', 'align_ok'))])
    res = dict(aggregate=agg, paired=paired, rows=rows)
    if a.json:
        p = ROOT / a.json
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(dict(aggregate=agg, paired={k: v for k, v in paired.items()}), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--configs', nargs='+', required=True, help='tag=config.json')
    r.add_argument('--seeds', default='7041-7045')
    r.add_argument('--jobs', type=int, default=4)
    r.add_argument('--out', default='runs/parallel_dry')
    s = sp.add_parser('summary')
    s.add_argument('--out', default='runs/parallel_dry')
    s.add_argument('--json', default='')
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else summary(a)


if __name__ == '__main__':
    sys.exit(main())
