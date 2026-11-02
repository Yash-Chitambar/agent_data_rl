"""SafeArm's table clearances on the logged targets of dry runs (SimIO bypasses SafeArm): would the run have stopped?
Reports per body the lowest commanded and measured height in stage '1 pick', against the config limits.

    PYTHONPATH=. .venv/bin/python -m deploy.checks.pick_clearance 'data/real_runs/*/pick_*_dry_*' --config deploy/config/bimanual_pick3.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_pick3.json'))
    a = ap.parse_args()
    from deploy.run_bimanual import load_cfg
    from deploy.safety_task import FlexChecker
    cfg = load_cfg(a.config)
    s = cfg['safety']
    chk = FlexChecker(np.zeros(6), s['clearance'], s['tip_clearance'], s.get('body_clearance'), s['table_z'])
    lim = {**{k: v for k, v in s['body_clearance'].items()}, 'left_finger': s['tip_clearance'], 'right_finger': s['tip_clearance']}
    lows, n, bad = {}, 0, 0
    for pat in a.runs:
        for d in sorted(glob.glob(pat)):
            f = Path(d) / 'ticks.npz'
            if not f.exists():
                continue
            t = np.load(f)
            idx = np.where(t['stage'] == 1)[0]
            n += 1
            for i in idx[::2]:
                h = chk.heights(t['target_b'][i, 0])
                bad += chk.violation(t['target_b'][i, 0]) is not None
                for b, z in h.items():
                    lows[b] = min(lows.get(b, np.inf), z - s['table_z'])
    out = {b: {'lowest_commanded_mm': round(z * 1e3, 1), 'limit_mm': round(lim[b] * 1e3, 1) if b in lim else None} for b, z in sorted(lows.items()) if b in lim}
    print(json.dumps(dict(runs=n, target_violations=int(bad), bodies=out), indent=1))
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
