"""Replay of the wrist-net up offset at the hand-over on the real logs (docs/real/PERCEPTION_BIAS_20261115.md, Part B):
constant offset (bimanual_align10_all: 2.6 mm) against the grasp-height model (align.secondary.bias_model, kind
stop_tip_y: up offset = a_mm + b * plug tip y in the camera at the stop, clipped). Input: wrist_bias.json of
wrist_bias_correlates.py --pv. Per hand-over run (n = 25): the raw up difference at the last measurement before the
hand-over (d_up_last) minus the offset = the value the +/-3.5 mm gate sees. Fixed fit and leave-one-run-out fit.

    PYTHONPATH=. python scripts/real/wrist_bias_replay.py --a 0.03 --b 0.395 --clip 0 4
"""
import argparse
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / 'results/placement/perception_bias'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', type=float, default=.03)
    ap.add_argument('--b', type=float, default=.395)
    ap.add_argument('--clip', type=float, nargs=2, default=(0., 4.))
    ap.add_argument('--gate', type=float, default=3.5)
    a = ap.parse_args()
    T = [t for t in json.loads((R / 'wrist_bias.json').read_text())['table'] if t['handover'] and np.isfinite(t['tip_cy_stop'])]
    x = np.array([t['tip_cy_stop'] for t in T]); y = np.array([t['d_up'] for t in T]); yl = np.array([t['d_up_last'] for t in T])
    model = np.clip(a.a + a.b * x, *a.clip)
    loo = np.array([np.clip(np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]), *a.clip) for i in range(len(T))])
    out = dict(n=len(T), gate_mm=a.gate, model=dict(a_mm=a.a, b=a.b, clip_mm=list(a.clip)))
    for name, off in (('raw', np.zeros(len(T))), ('const_2.6', np.full(len(T), 2.6)), ('model_fixed', model), ('model_loo', loo)):
        r = yl - off
        out[name] = dict(min=round(float(r.min()), 2), max=round(float(r.max()), 2), sd=round(float(r.std()), 2),
                         over_gate=int((np.abs(r) > a.gate).sum()), margin_min=round(float(a.gate - np.abs(r).max()), 2))
    out['runs'] = [dict(date=t['date'], run=t['run'], tip_cy_stop=round(t['tip_cy_stop'], 2), d_up_last=round(t['d_up_last'], 2),
                        model=round(float(m), 2), resid_model=round(float(t['d_up_last'] - m), 2), resid_const=round(float(t['d_up_last'] - 2.6), 2))
                   for t, m in zip(T, model)]
    print(json.dumps({k: v for k, v in out.items() if k != 'runs'}, indent=1))
    for r_ in out['runs']:
        print(r_['date'], r_['run'][:26].ljust(26), r_['tip_cy_stop'], r_['d_up_last'], r_['model'], r_['resid_model'], r_['resid_const'])
    (R / 'wrist_bias_replay.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
