"""K-J in the sim from the four real plain-PD hand-overs (2026-11-03 align_full2/5, 2026-11-06 align_full9/10).

All four were gap 5.3-5.8 mm, lateral within 0.62 mm. Conditions: gap 5.5 mm, ring 0.5 mm (angles from the chain pool),
with the real hand-over target lead (joint target - measured, mean of the four runs) or the sim's own, under a
friction scale range (env_extra friction_scale_range) and gravity jitter. Records, per world, the joint lead at 0.6 s and
the outcome, so seats and stalls can be compared. Resumable on --out.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/eval/kj_real4.py --tag train --out results/real_align/kj_real4.json
  ... --tag fric_hi --env_extra '{"friction_scale_range": [1.3, 2.0]}'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.eval.kj_start_map as KSM  # noqa: E402
from scripts.eval.kj_start_map import build, rows_base, run_cond  # noqa: E402

_cond_rows = KSM.cond_rows


def cond_rows_ang(c, base, g):
    """kj_start_map.cond_rows plus angle errors on top of the pool's: rx/ry/roll (deg) with a random sign."""
    r = _cond_rows(c, base, g)
    for key, col in (('rx', 3), ('ry', 4), ('roll', 5)):
        if key in c:
            r[:, col] = r[:, col] + c[key] * g.choice([-1., 1.], len(r))
    return r


KSM.cond_rows = cond_rows_ang

# obs[38:44] at t = 0 (joint target - believed, rad), right arm, real runs full2, full5, full9, full10
REAL_LEAD = np.array([[0.4, -5.1, 6.3, 1.3, 15.1, -0.3], [0.3, -5.1, 9.4, 1.3, 25.0, 1.5],
                      [0.4, -5.1, 9.5, 0.8, 14.8, 1.5], [-0.3, -5.5, 9.5, 0.9, 15.9, -4.5]]).mean(0) * 1e-3
CONDS = [dict(gap=5.5, ring=.5, lead='sim'), dict(gap=5.5, ring=.5, lead='real'),
         dict(gap=5.5, ring=2., lead='real'), dict(gap=5.5, ring=3., lead='real'),
         dict(gap=2.5, ring=.5, lead='real'), dict(gap=9., ring=.5, lead='real'),
         dict(gap=5.5, ring=.5, lead='real', roll=2.), dict(gap=5.5, ring=.5, lead='real', roll=4.),
         dict(gap=5.5, ring=.5, lead='real', rx=3.), dict(gap=5.5, ring=.5, lead='real', rx=6.),
         dict(gap=5.5, ring=2., lead='real', rx=3., roll=2.)]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=3100)
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', default='results/real_align/kj_real4.json')
    A = ap.parse_args()
    out = Path(A.out)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {(d['tag'], json.dumps(d['cond'], sort_keys=True)) for d in done}
    torch.manual_seed(A.seed)
    env, pol = build(A.nworld, A.seed, extra=json.loads(A.env_extra))
    base = rows_base()
    for i, c in enumerate(CONDS):
        if (A.tag, json.dumps(c, sort_keys=True)) in seen:
            continue
        rec = {}

        def hook(kind, e, act=None, c=c, rec=rec):
            if kind == 'reset':
                q0 = e._measured_model_angles()[:, 0]
                if c['lead'] == 'real':
                    e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
                rec['t'] = 0
                return None
            rec['t'] += 1
            if rec['t'] == 30:
                rec['lead06'] = (e.joint_target[:, 0] - e._measured_model_angles()[:, 0]).cpu().numpy() * 1e3
            return act
        t0 = time.time()
        r = run_cond(env, pol, {k: v for k, v in c.items() if k != 'lead'}, base, A.seed + i, on_step=hook)
        r['cond'], r['tag'], r['env_extra'] = c, A.tag, json.loads(A.env_extra)
        if 'lead06' in rec:
            r['lead06_mrad_p50'] = np.median(rec['lead06'], 0).round(1).tolist()
        done.append(r)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(done, indent=1))
        print(A.tag, json.dumps({k: r[k] for k in ('cond', 'k', 'n', 'rate', 'wilson', 'classes', 't_ok_p50')}), f'{time.time() - t0:.0f} s', flush=True)
