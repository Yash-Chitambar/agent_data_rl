"""Scale gate against pose-dependent noise, no arm: false stops with slope 1, and how often real scale errors pass. Monte Carlo on AlignLogic with a perfect hand.

    python -m deploy.checks.align_scale_gate [--n 2000] [--json results/real_align/scale_gate.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual.align import DEFAULTS, AlignLogic, PortMeasure  # noqa: E402

CASES = {'slope 1': (1, 1, 1), 'old fine net 0.34 0.17 0.35': (.34, .17, .35), 'old coarse net 0.50 0.32 0.59': (.5, .32, .59), 'up 0.5 only': (1, .5, 1), 'across 0.5 only': (.5, 1, 1), 'gap 0.5 only': (1, 1, .5), 'up 1.8 only': (1, 1.8, 1)}
POSE = np.array([.25, .74, .49])


def run(slope, g, start=(-1., 6., 8.), bias=(0, 0, 0)):
    lg = AlignLogic(dict(DEFAULTS))
    p, tip = np.array(start, float), np.zeros(3)
    slope = np.asarray(slope, float)
    rb = g.normal(0, 1, 3) * [.3, .6, .4]
    d = None
    for _ in range(200):
        v = slope * p + rb + g.normal(0, 1, 3) * POSE
        if slope[2] < .9:    # a low gap slope reads a small gap: keep the case about scale, not the clearance gate
            v[2] += 6 * (1 - slope[2])
        d = lg.update(PortMeasure(True, 'mc', 1., *v, .74), None, tip)
        if d['kind'] in ('done', 'fail'):
            break
        p, tip = p + d['delta'], tip + d['delta']
    return d, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    out = {}
    ok = True
    for name, sl in CASES.items():
        g = np.random.default_rng(1)
        k, rs, cyc = 0, {}, []
        for _ in range(a.n):
            d, lg = run(sl, g)
            if d['kind'] == 'done':
                k += 1
                cyc.append(lg.cycle)
            else:
                key = d['reason'].split(':')[0].split(' (')[0][:30]
                rs[key] = rs.get(key, 0) + 1
        out[name] = dict(n=a.n, done=k, rate=k / a.n, stops=rs, cycles_p50=float(np.median(cyc)) if cyc else None)
        print(name, json.dumps(out[name]), flush=True)
    if out['slope 1']['rate'] < .9 or out['old fine net 0.34 0.17 0.35']['rate'] > .01:
        ok = False
    print('ok' if ok else 'FAILED')
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
