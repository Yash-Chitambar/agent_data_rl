"""Insertion policy (K-J, K2, ...) on the acceptance matrix of docs/real/K2_PROGRESS.md. Per cell: seat rate (Wilson 95 %),
failure classes, time to seat, and the commanded plug-tip climb (lateral offset of the target tip from the measured tip,
port axes, believed FK; max over each world's first episode). The policy's own env_json is the base; training-only DR
(extra angles, lead share) is off and the cell sets the start. Plant: --plant train | real (friction x1.3-2.0, gravity 12 %).

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/k2_matrix.py --policy runs/P4-insert-k2-s0/last.pt --plant real --out results/real_align/k2/k2s0_real.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.eval.kj_real4  # noqa: E402,F401  (angle conditions in cond_rows)
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402

CELLS = {'real_ho': dict(gap=5.5, ring=.5), 'ring3': dict(gap=5.5, ring=3.), 'ring4': dict(gap=5.5, ring=4.), 'ring5': dict(gap=5.5, ring=5.),
         'roll3': dict(gap=5.5, ring=.5, roll=3.), 'roll4': dict(gap=5.5, ring=.5, roll=4.), 'pitch4': dict(gap=5.5, ring=.5, rx=4.),
         'gap8': dict(gap=8., ring=.5), 'mix': dict(gap=5.5, ring=2., roll=2., rx=3.)}
PLANT = {'train': {}, 'real': {'friction_scale_range': [1.3, 2.0], 'gravity_scale_jitter': 0.12}}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--plant', default='real', choices=sorted(PLANT))
    ap.add_argument('--cells', default=','.join(CELLS))
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=5100)
    ap.add_argument('--lead', default='real', choices=('real', 'sim'))
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['cell'] for d in done}
    torch.manual_seed(A.seed)
    pol = InsertionPolicy(A.policy, 'cuda:0')
    ej = dict(pol.meta['env_json'], hard_frac=0., scale_range_frac=0., live_frac=0., extra_ang_frac=0., handover_lead_frac=0.)
    ej.update(PLANT[A.plant]); ej.update(json.loads(A.env_extra))
    env = make_env(argparse.Namespace(nworld=A.nworld, seed=A.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    base = rows_base()
    for i, name in enumerate(A.cells.split(',')):
        if name in seen:
            continue
        st = {}

        def hook(kind, e, act=None):
            if kind == 'reset':
                q0 = e._measured_model_angles()[:, 0]
                if A.lead == 'real':
                    e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
                st['climb'] = torch.zeros(e.N, device=e.device)
                st['live'] = torch.ones(e.N, dtype=torch.bool, device=e.device)
                return None
            st['climb'] = torch.where(st['live'], torch.maximum(st['climb'], e._climb_lateral_mm()), st['climb'])
            st['live'] &= e.steps < e.max_steps - 1
            st['live'] &= ~((e.steps == 0) & (st['climb'] > 0))     # a world that reset is done with its first episode
            return act
        t0 = time.time()
        r = run_cond(env, pol, CELLS[name], base, A.seed + i, on_step=hook)
        c = st['climb'].cpu().numpy()
        r.update(cell=name, plant=A.plant, lead=A.lead, policy=A.policy, climb_mm_p50_p95_max=np.percentile(c, [50, 95, 100]).round(1).tolist())
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(name, f"{r['k']}/{r['n']} = {100 * r['rate']:.1f} % [{100 * r['wilson'][0]:.1f}, {100 * r['wilson'][1]:.1f}]",
              {k: v for k, v in r['classes'].items() if k != 'seated'}, 't', r['t_ok_p50'], 'climb', r['climb_mm_p50_p95_max'], f'{time.time() - t0:.0f} s', flush=True)
