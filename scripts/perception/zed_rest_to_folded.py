"""v1 ZED error as both arms move from the chain's rest pose (the recorded reference pose,
data/sim_ref/2026-10-25_scene/pointed.json + joint offsets = BimanualRuntime.initial_q) toward folded
(all joints 0, the pose of the real stationary frames). Joint-space interpolation alpha = 0 (rest) .. 1
(folded), right fingers fully open, reader on the Part 1 strip, policy/v1 visual DR, n = 512 per alpha.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python scripts/perception/zed_rest_to_folded.py
Writes results/usbc/policy/approach/zed_rest_to_folded.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch

from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.perception.zed_estimator import ZedEstimator


def main():
    dev = torch.device('cuda:0')
    r = ApproachRenderer(nworld=256, seed=77, task_layout='results/usbc/policy/pick/task_layout.json')
    rt = r.rt
    est = ZedEstimator(str(dev), pose='runs/perception/zed.pt', presence='runs/perception/zed_v1_gate_arm.pt')
    ra, la = rt.indices['right'][0], rt.indices['left'][0]
    rf = [int(rt.mjm.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
    ref_r, ref_l = rt.initial_q[ra].clone(), rt.initial_q[la].clone()
    out = {'rest_pose_rad': {'right': ref_r.tolist(), 'left': ref_l.tolist()},
           'rest_minus_folded_max_abs_rad': {'right': float(ref_r.abs().max()), 'left': float(ref_l.abs().max())}}
    for alpha in (0., .1, .25, .5, .75, 1.):
        errs, oks = [], []
        for _ in range(2):
            r.h.reset()
            rt.q[:, ra] = ref_r * (1 - alpha); rt.q[:, la] = ref_l * (1 - alpha)
            rt.q[:, rf] = .0475
            with wp.ScopedDevice(rt.cfg.device):
                mjw.forward(rt.m, rt.d)
            p, _ = r.h.pc_pose()
            port, _, ok, _ = est(rt.render()['top'], arms=(r.h.believed('left'), r.h.believed('right'), rt.q[:, rf[0]]))
            errs.append(((port - p[:, :2]).norm(dim=1) * 1e3).cpu()); oks.append(ok.cpu())
        e, k = torch.cat(errs).numpy(), torch.cat(oks).numpy()
        out[f'alpha_{alpha:g}'] = {'n': len(e), 'xy_mm_p50': float(np.percentile(e, 50)), 'xy_mm_p99': float(np.percentile(e, 99)),
                                   'gate_accept': float(k.mean())}
        print(alpha, out[f'alpha_{alpha:g}'], flush=True)
    Path('results/usbc/policy/approach/zed_rest_to_folded.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('alpha')}))


if __name__ == '__main__':
    main()
