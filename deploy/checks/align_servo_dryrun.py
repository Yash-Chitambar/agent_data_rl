"""Servo align stage (align.servo) in sim dry runs: run_bimanual --dry-run on several seeds, then from each run's log: outcome, servo time,
frames, final estimate against sim truth, tip speed profile (0.1 s window, from the servo rows), frame age, entry gap, and in full mode
the insertion outcome. No arm.

  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python -m deploy.checks.align_servo_dryrun --config deploy/config/bimanual_align6_servo.json --mode align --seeds 7100,7101,7102
  ... --summarize-only RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def summarize(run):
    run = Path(run)
    ev = [json.loads(x) for x in open(run / 'events.jsonl')]
    out = dict(run=run.name)
    ends = [e for e in ev if e['kind'] == 'align_servo_end']
    if ends:
        e0 = ends[0]
        out.update(ok=e0['ok'], reason=e0['reason'], seconds=round(e0['seconds'], 2), frames=e0['frames'],
                   est=np.round(e0['est'][:3], 2).tolist() if e0['est'] else None, truth=np.round(e0['truth_mm'], 2).tolist() if e0['truth_mm'] else None,
                   truth_rot_deg=np.round(e0['truth_rot_deg'], 2).tolist() if e0.get('truth_rot_deg') else None)
        if e0['truth_mm']:
            t = np.asarray(e0['truth_mm'])
            out['true_err_mm'] = np.round(t - [0, 0, 6], 2).tolist()
        if len(ends) > 1:
            out['post_press'] = dict(ok=ends[1]['ok'], reason=ends[1]['reason'], seconds=round(ends[1]['seconds'], 2),
                                     truth=None if not ends[1]['truth_mm'] else np.round(ends[1]['truth_mm'], 2).tolist())
    fr = [e for e in ev if e['kind'] == 'align_servo_frame']
    good = [e for e in fr if e['ok']]
    if good:
        out['entry_gap_mm'] = round(good[0]['v'][2], 2)
        out['frame_ms_p50'] = round(float(np.median([e['ms'] for e in fr])), 1)
    p = run / 'servo.npz'
    if p.exists():
        d = np.load(p)
        sp = d['speed']
        out.update(speed_p50_p99_max=[round(float(np.percentile(sp, 50)), 1), round(float(np.percentile(sp, 99)), 1), round(float(sp.max()), 1)],
                   age_max_s=round(float(d['age'].max()), 3), ticks=int(len(sp)))
    for kind in ('align_handover', 'insert_result', 'align_result', 'align_entry_too_close', 'back_off'):
        es = [e for e in ev if e['kind'] == kind]
        if es:
            e = es[-1]
            out[kind] = {k: e[k] for k in ('ok', 'reason', 'gap_mm', 'lat_mm', 'outcome', 't_seat_s', 'depth_mm', 'true_seated_end', 'true_depth_mm_end', 'seconds', 'mm') if k in e}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--mode', default='align')
    ap.add_argument('--seeds', default='7100,7101,7102')
    ap.add_argument('--tag', default='servo')
    ap.add_argument('--out')
    ap.add_argument('--align-set', action='append', default=[], help='passed to run_bimanual (k=json)')
    ap.add_argument('--summarize-only', nargs='*')
    a = ap.parse_args()
    rows = []
    if a.summarize_only:
        rows = [summarize(r) for r in a.summarize_only]
    else:
        py = sys.executable
        for s in a.seeds.split(','):
            name = f'dry_{a.tag}_s{s}'
            cmd = [py, '-m', 'deploy.run_bimanual', '--dry-run', '--stage', 'approach_insert', '--seed', s, '--config', a.config, '--align-mode', a.mode, '--name', name] + sum([['--align-set', x] for x in a.align_set], [])
            r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
            runs = sorted(glob.glob(str(ROOT / f'data/real_runs/*/*_{name}')), key=os.path.getmtime)
            if not runs:
                rows.append(dict(seed=s, error=r.stderr[-800:]))
                continue
            row = summarize(runs[-1])
            row['seed'] = s
            row['completed'] = 'completed True' in r.stdout
            rows.append(row)
            print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
    if a.summarize_only:
        for r in rows:
            print(json.dumps(r))


if __name__ == '__main__':
    main()
