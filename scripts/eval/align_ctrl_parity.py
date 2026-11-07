"""Parity (float64; float32 differs only at exact threshold ties) of the batched torch precision law (blind_insertion/envs/precision_ctrl_torch.py) with the NumPy law
(precision_ctrl.PrecisionLaw), one world per random trajectory, called at the same rate. Prints the max torque difference.

  PYTHONPATH=. python scripts/eval/align_ctrl_parity.py
"""
import numpy as np
import torch

from blind_insertion.envs.precision_ctrl import PrecisionLaw
from blind_insertion.envs.precision_ctrl_torch import PrecisionLawTorch

P = dict(integral=True, settle_lock=True, friction_ff=True, friction=[0.138, 0.447, 1.087, 0.069, 0.091, 0.087],
         deadband=0.0004, latch_band=0.001, unlatch_band=0.002, freeze=0.012, hz=50.0, kp=[80, 80, 80, 30, 30, 30])


def run(seed, T=600):
    g = np.random.default_rng(seed)
    a, b = PrecisionLaw(P), PrecisionLawTorch(1, P, dtype=torch.float64)
    tgt = np.zeros(6)
    q = np.zeros(6)
    dmax = 0.0
    for k in range(T):
        if g.random() < 0.05:                      # a step or a hold
            tgt = tgt + g.normal(0, 0.004, 6)
        q = q + (tgt - q) * g.uniform(0, 0.3) + g.normal(0, 2e-4, 6)
        ra = a.step((k + 1) / P['hz'], q, tgt)['tau']
        rb = b.step(torch.tensor(q[None], dtype=torch.float64), torch.tensor(tgt[None], dtype=torch.float64), 1 / P['hz'], t_abs=(k + 1) / P['hz'])[0].numpy()
        dmax = max(dmax, float(np.abs(ra - rb).max()))
    return dmax


if __name__ == '__main__':
    d = [run(s) for s in range(20)]
    print('max |tau_numpy - tau_torch| over 20 x 600 steps:', round(max(d), 5), 'N m', 'PASS' if max(d) < 1e-3 else 'FAIL')
