"""Funnel sizing (docs/real/CONTACT_SENS_20261115.md, test 2): must the camera alignment stop within 1 mm per axis, or is ~2 mm
enough? Sim only. The insertion policy starts from a lateral offset ring (0.5-3 mm) at gap 5.5 mm, rig-spot layout, real-like
plant, real right-arm lead, training contact model; probe as contact_sens.py (seat, full depth, seat time, release).

Two prior modes per ring, n worlds each (groups of one batch):
  truth: the true start is on the ring; the policy's vision prior = truth + N(0, (0.4, 0.95, 0.63) mm) (as in training);
  real : the vision measurement is on the ring (where the alignment stopped) and is the prior exactly; the true start is
         measurement + N(0, (0.4, 0.95, 0.63) mm) (the hand-over model of replay_real_handovers.py).

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/funnel_rings.py --policy runs/insert_K7.pt --tag k7 \
        --groups truth:0.5,truth:1,real:0.5,real:1
Output: results/real_align/contact_sens/raw/funnel_<tag>.json (one row per group, resumable).
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
from scripts.eval.contact_sens import OUT, RAW, build, ring_rows, run_probe, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--groups', required=True, help='comma list of mode:ring_mm, mode in truth|real')
    ap.add_argument('--place', default='real')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=8300)
    ap.add_argument('--out_dir', default=str(OUT / 'raw'))
    A = ap.parse_args()
    out = Path(A.out_dir) / f'funnel_{A.tag}_{A.place}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['group'] for d in done}
    groups = [x for x in A.groups.split(',') if x not in seen]
    if not groups:
        return
    torch.manual_seed(A.seed)
    N = A.n * len(groups)
    env, pol = build(A.policy, N, A.seed, A.place)
    from scripts.eval.kj_start_map import rows_base
    base = rows_base()
    P = env.icfg.pool_per_world
    g = np.random.default_rng(A.seed)
    rows, prior = [], np.full((N, 3), np.nan)
    for k, grp in enumerate(groups):
        mode, ring = grp.split(':')
        r, m = ring_rows(base, g, A.n * P, float(ring), real_prior=(mode == 'real'))
        rows.append(r)
        if mode == 'real':
            prior[k * A.n:(k + 1) * A.n] = m[::P]      # world w's first start is pool entry w * P
    rows = np.concatenate(rows)
    t0 = time.time()
    res, cls, ok_row = run_probe(env, pol, [(x, {}) for x in groups], rows, A.seed, prior=prior)
    seat_mm = env.seat_depth * 1e3
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / f'funnel_{A.tag}_{A.place}_{"-".join(groups).replace(":", "")}.npz', cls=cls.astype(str), ok_row=ok_row, **res)
    for k, grp in enumerate(groups):
        sel = np.zeros(N, bool); sel[k * A.n:(k + 1) * A.n] = True
        sel_ok = sel & ok_row          # a world whose first start is not its row (IK retry) is left out
        s = summarize(res, cls, sel_ok, seat_mm)
        mode, ring = grp.split(':')
        s.update(group=grp, mode=mode, ring_mm=float(ring), tag=A.tag, policy=A.policy, place=A.place, seed=A.seed,
                 dropped_ik_retry=int((sel & ~ok_row).sum()), wall_s=round(time.time() - t0), batch_worlds=N)
        done.append(s)
        print(grp, f"{s['k']}/{s['n']} = {100 * s['rate']:.1f} % full {100 * s['full_depth_rate']:.1f} % t {s['t_seat_p50_p90_p99']} "
              f"rel_in {s['release_in_k']}/{s['release_n']}", {a: b for a, b in s['classes'].items() if a != 'seated'}, flush=True)
    out.write_text(json.dumps(done, indent=1))


if __name__ == '__main__':
    main()
