"""K8 fine-tune launcher (docs/real/K8_20261115.md). Sim only.

train_bimanual.py with the K8 contact randomization, set in this process only (no library default changes, no env_json key:
the checkpoint's env_json stays K7's, so every eval script runs K8 under the same conditions as K4 / K7):
  --port_mu lo hi      port friction U(lo, hi) (K7: 0.10 0.20)
  --mu_sched lo h0 h1 T   curriculum instead: U(lo, h), h from h0 to h1 over T env steps
  --grip lo hi         squeeze-force scale U(lo, hi) (K7: 1.0 1.0)
  --fit spec_train     one USB-C spec plug per world (k8_fit.FIT_GRID: 20-90 / 50-80 um per side); 'nom' = K7's plug
  --live_redraw        live-state worlds get new contact values (else they keep the saved K4/K7 values)
All other arguments go to train_bimanual.py.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/k8_train.py --port_mu 0.10 0.60 --grip 0.6 1.0 \
        --fit spec_train --live_redraw -- --name P4-insert-k8-s0 --init runs/insert_K7.pt --steps 30e6 --curriculum 1.5 --thr 0.99 \
        --seed 481 --env_json "$(cat runs/k7_env.json)"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    ap = argparse.ArgumentParser()
    ap.add_argument('--port_mu', type=float, nargs=2, default=None)
    ap.add_argument('--mu_sched', type=float, nargs=4, default=None)
    ap.add_argument('--grip', type=float, nargs=2, default=None)
    ap.add_argument('--fit', default='nom')
    ap.add_argument('--live_redraw', action='store_true')
    argv = sys.argv[1:]
    cut = argv.index('--') if '--' in argv else len(argv)
    A = ap.parse_args(argv[:cut])
    from scripts.eval.k8_fit import install
    from blind_insertion.envs import insertion_task as IT
    from blind_insertion import train_bimanual as TB
    if A.port_mu:
        IT.DEFAULT_CONTACT_DR['port_friction'] = tuple(A.port_mu)
    if A.mu_sched:
        IT.DEFAULT_CONTACT_DR['port_friction'] = (A.mu_sched[0], A.mu_sched[1])
    if A.grip:
        IT.DEFAULT_CONTACT_DR['grip_gain_scale'] = tuple(A.grip)
    install(fit=A.fit, mu_sched=tuple(A.mu_sched) if A.mu_sched else None, live_redraw=A.live_redraw)
    print('[k8_train] contact_dr', IT.DEFAULT_CONTACT_DR, 'fit', A.fit, 'mu_sched', A.mu_sched, 'live_redraw', A.live_redraw, flush=True)
    targs = TB.parser().parse_args(argv[cut + 1:])
    TB.train(targs)
