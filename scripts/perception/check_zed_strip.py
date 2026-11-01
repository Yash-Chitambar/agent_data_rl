"""ZED estimator + gate with the reader lying in the Part 1 strip in front of the PC (x 0.095-0.145,
y 0.335-0.435 m, yaw +/-25 deg), vs the old reader region, on DR renders (n = 1024 each).

    PYTHONPATH=. .venv/bin/python scripts/perception/check_zed_strip.py
Writes results/usbc/policy/approach/zed_strip_check.json
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.perception.zed_estimator import ZedEstimator


def main():
    r = ApproachRenderer(nworld=256, seed=11)
    est = ZedEstimator(r.dev)
    out = {}
    for mode in ('old_region', 'front_strip'):
        errs, yerr, ok = [], [], []
        for b in range(4):
            r.h.reset()
            rt = r.rt
            if mode == 'front_strip':
                n = r.n
                pa = rt.lay.peg_qadr
                u = lambda lo, hi: torch.rand(n, device=r.dev) * (hi - lo) + lo   # noqa: E731
                x, y, yaw = u(.095, .145), u(.335, .435), torch.deg2rad(u(-25, 25))
                # reader body centre at (x, y): keep height and base orientation, turn about z
                R0 = K.quat_to_mat(rt.q[:, pa + 3:pa + 7])
                Rz = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=r.dev).expand(n, 3), yaw)
                # current yaw of the reader removed: base = Rz(-yaw_cur) R0
                ycur = torch.atan2(R0[:, 1, 2], R0[:, 0, 2]) - math.pi   # part z = -x world at yaw 0
                Rb = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=r.dev).expand(n, 3), -ycur) @ R0
                R = Rz @ Rb
                centre = torch.stack([x, y, rt.q[:, pa + 2]], 1)
                bz = .0698                                            # body centre along the part axis
                rt.q[:, pa:pa + 3] = centre - R[:, :, 2] * bz
                rt.q[:, pa + 2] = centre[:, 2]
                rt.q[:, pa + 3:pa + 7] = K.mat_to_quat(R)
            with wp.ScopedDevice(rt.cfg.device):
                mjw.forward(rt.m, rt.d)
            p, R = r.h.pc_pose()
            yaw_t = torch.atan2(-R[:, 1, 2], -R[:, 0, 2])
            port, yaw_e, good, _ = est(rt.render()['top'])
            errs.append(((port - p[:, :2]).norm(dim=1) * 1e3).cpu()); ok.append(good.cpu())
            yerr.append(torch.rad2deg(torch.atan2(torch.sin(yaw_e - yaw_t), torch.cos(yaw_e - yaw_t))).abs().cpu())
        e, y, k = torch.cat(errs).numpy(), torch.cat(yerr).numpy(), torch.cat(ok).numpy()
        out[mode] = {'n': len(e), 'gate_accept_rate': float(k.mean()),
                     'xy_mm_accepted': {f'p{q}': float(np.percentile(e[k], q)) for q in (50, 99, 100)},
                     'yaw_deg_accepted': {f'p{q}': float(np.percentile(y[k], q)) for q in (50, 99)}}
        print(mode, json.dumps(out[mode]))
    Path('results/usbc/policy/approach/zed_strip_check.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
