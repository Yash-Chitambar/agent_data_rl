"""K-J with the real observation offsets put on top of the sim's actor observation (the plant is unchanged).
Real hand-overs (2026-11-03/06, n = 4) read TCP - nominal (9-11, 29-31, 4.5-6.6) mm and Euler (-0.06..-0.08, -0.05..-0.07, ~0)
rad, in-hand estimate (+4.0..4.5, ~0, -1.0..-2.6 mm; rotvec up to 2.2 deg), while the sim's dry runs read about
(-5.5, 3.5, -5.5) mm, (-0.02, 0, -0.03) rad, in-hand ~0. Each condition sets the observed values to the real ones
(shift = real - this world's value at reset, held for the episode), from gap 5.5 mm, ring 0.5 mm, real target lead.

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/eval/kj_obs_shift.py --out results/real_align/kj_obs_shift.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import build, rows_base, run_cond  # noqa: E402

REAL_TCP = np.array([9.7, 30.0, 6.0]) * 1e-3 * 10          # obs units (m x 10)
REAL_EUL = np.array([-0.072, -0.060, 0.003])
REAL_INHAND = np.array([4.26, 0.11, -1.96, -0.93, 0.0, 1.26])  # mean of the four runs
CONDS = {'none': [], 'tcp': ['tcp'], 'eul': ['eul'], 'pose': ['tcp', 'eul'], 'inhand': ['inhand'], 'all': ['tcp', 'eul', 'inhand'],
         'recenter': ['recenter']}
# recenter: TCP and Euler features set to the batch mean at the hand-over (no per-world absolute position): the deploy-side fix

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=3500)
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    A = ap.parse_args()
    out = Path(A.out)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['shift'] for d in done}
    torch.manual_seed(A.seed)
    env, pol = build(A.nworld, A.seed, extra=json.loads(A.env_extra))
    base_obs = env._actor_obs
    state = {'shift': None, 'keys': []}

    def shifted():
        o = base_obs()
        if state['shift'] is not None:
            o = o + state['shift']
        return o
    env._actor_obs = shifted
    dev = env.device
    for i, (name, keys) in enumerate(CONDS.items()):
        if name in seen or (A.only and name not in A.only.split(',')):
            continue

        def hook(kind, e, act=None, keys=keys):
            if kind == 'reset':
                q0 = e._measured_model_angles()[:, 0]
                e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=dev)
                state['shift'] = None
                o = base_obs()
                s = torch.zeros_like(o)
                if 'tcp' in keys:
                    s[:, 50:53] = torch.as_tensor(REAL_TCP, dtype=o.dtype, device=dev) - o[:, 50:53]
                if 'eul' in keys:
                    s[:, 53:56] = torch.as_tensor(REAL_EUL, dtype=o.dtype, device=dev) - o[:, 53:56]
                if 'inhand' in keys:
                    s[:, 62:68] = torch.as_tensor(REAL_INHAND, dtype=o.dtype, device=dev) - o[:, 62:68]
                if 'recenter' in keys:
                    s[:, 50:56] = o[:, 50:56].mean(0, keepdim=True) - o[:, 50:56]
                state['shift'] = s
                e.hist[:] = (o + s)[:, None]
                return None
            return act
        t0 = time.time()
        r = run_cond(env, pol, {'gap': 5.5, 'ring': .5}, rows_base(), A.seed + i, on_step=hook)
        r['shift'] = name
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(name, json.dumps({k: r[k] for k in ('k', 'n', 'rate', 'wilson', 'classes', 't_ok_p50')}), f'{time.time() - t0:.0f} s', flush=True)
