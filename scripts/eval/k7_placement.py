"""Insertion policy against the PC placement (K7 evaluation set, docs/real/K7_20261114.md). Same cells and plant as
scripts/eval/k4_placement.py (real-like plant, real right-arm hand-over lead, 'real_ho' hand-over: gap 5.5 mm, ring 0.5 mm,
all worlds of a cell at one PC pose, first episode per world), plus four more corners of the step-2 box and yaw +25 deg.
Differences to k4_placement.py: --env_extra; training-only DR of newer policies is off (vision_prior_sigma_scale) and the
mating force is off unless --env_extra sets it (K6d/K7 carry mate_force_n in their env_json).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/eval/k7_placement.py --policy runs/insert_K4.pt --out results/real_align/k7/k4_place.json
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
from scripts.eval.k4_placement import PCS as PCS_K4, REAL  # noqa: E402
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402

PCS = PCS_K4 + [(f'c{dx:+d}{dy:+d}{dyw:+d}', dx, dy, dyw) for dx, dy, dyw in ((-20, -20, 15), (-20, 20, 15), (20, 20, -15), (20, -20, -10))] + \
      [('yaw+25', 0, 0, 25)]
# step-2 box of docs/real/ROBUSTNESS_PLAN_20261113.md: x, y +/-25 mm, yaw +/-15 deg around the real spot
BOX = [t for t, dx, dy, dyw in PCS if abs(dx) <= 25 and abs(dy) <= 25 and abs(dyw) <= 15 and t != 'sim_nominal']
BASE_EXTRA = {'task_layout': 'results/placement/task_layout_real_v2.json', 'mate_force_n': None, 'vision_prior_sigma_scale': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=6300)
    ap.add_argument('--only', default='', help='comma list of cell tags')
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['pc_tag'] for d in done}
    extra = dict(BASE_EXTRA, **json.loads(A.env_extra))
    torch.manual_seed(A.seed)
    env, pol = build(A.policy, A.nworld, A.seed, extra)
    rt = env.runtime
    rt.cfg.pc_pose_range = (0., 0., 0.)
    base = rows_base()
    only = set(A.only.split(',')) if A.only else None

    def hook(kind, e, act=None):
        if kind == 'reset':
            q0 = e._measured_model_angles()[:, 0]
            e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
            e.hist[:] = e._actor_obs()[:, None]
            return None
        return act
    for i, (tag, dx, dy, dyaw) in enumerate(PCS):
        if tag in seen or (only and tag not in only):
            continue
        pc = (REAL[0] + dx * 1e-3, REAL[1] + dy * 1e-3, REAL[2] + dyaw)
        rt.layout['port_xy'], rt.layout['yaw'] = [pc[0], pc[1]], pc[2]
        t0 = time.time()
        r = run_cond(env, pol, CELLS['real_ho'], base, A.seed + i, on_step=hook)
        r.update(pc_tag=tag, pc=list(pc), in_box=tag in BOX, policy=A.policy, env_extra=extra, wall_s=round(time.time() - t0))
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(tag, pc, f"{r['k']}/{r['n']} = {100 * r['rate']:.1f} % [{100 * r['wilson'][0]:.1f}, {100 * r['wilson'][1]:.1f}]",
              {k: v for k, v in r.get('classes', {}).items() if k != 'seated'}, f"{r['wall_s']} s", flush=True)


if __name__ == '__main__':
    main()
