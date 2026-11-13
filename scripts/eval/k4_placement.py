"""K4 insertion rate against the PC placement (sim, real-like plant, real right-arm hand-over lead, prior as trained).
Every world of a cell has the PC port at one pose (runtime.layout set, pc_pose_range zero); the hand-over state is the
'real_ho' cell of scripts/eval/k3_robust.py (gap 5.5 mm, lateral ring 0.5 mm). n per cell = --nworld (first episode).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/k4_placement.py --out results/placement/k4_placement.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.k3_robust import CELLS, build  # noqa: E402
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402

REAL = (0.231, 0.394, 4.0)
PCS = [('real', 0, 0, 0)] + [(f'dx{d:+d}', d, 0, 0) for d in (-30, -20, 20, 30)] + [(f'dy{d:+d}', 0, d, 0) for d in (-30, -20, 20, 30)] + \
      [(f'yaw{d:+d}', 0, 0, d) for d in (-15, -10, 10, 15, 20)] + [('sim_nominal', -11, -34, -4)] + \
      [(f'c{dx:+d}{dy:+d}{dyw:+d}', dx, dy, dyw) for dx, dy, dyw in ((-20, -20, -10), (20, 20, 15), (20, -20, 15), (-20, 20, -10))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default=str(ROOT / 'runs/insert_K4.pt'))
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seed', type=int, default=6300)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['pc_tag'] for d in done}
    env, pol = build(A.policy, A.nworld, A.seed, {})
    rt = env.runtime
    rt.cfg.pc_pose_range = (0., 0., 0.)
    base = rows_base()

    def hook(kind, e, act=None):
        if kind == 'reset':
            q0 = e._measured_model_angles()[:, 0]
            e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
            e.hist[:] = e._actor_obs()[:, None]
            return None
        return act
    for i, (tag, dx, dy, dyaw) in enumerate(PCS):
        if tag in seen:
            continue
        pc = (REAL[0] + dx * 1e-3, REAL[1] + dy * 1e-3, REAL[2] + dyaw)
        rt.layout['port_xy'], rt.layout['yaw'] = [pc[0], pc[1]], pc[2]
        t0 = time.time()
        r = run_cond(env, pol, CELLS['real_ho'], base, A.seed + i, on_step=hook)
        r.update(pc_tag=tag, pc=list(pc), wall_s=round(time.time() - t0))
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(tag, pc, f"{r['k']}/{r['n']} = {100 * r['rate']:.1f} % [{100 * r['wilson'][0]:.1f}, {100 * r['wilson'][1]:.1f}]",
              {k: v for k, v in r.get('classes', {}).items() if k != 'seated'}, flush=True)


if __name__ == '__main__':
    main()
