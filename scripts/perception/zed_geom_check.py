"""Sim check: signed error of the ZED PC nets on renders of the sim PC ('sim') and of the measured real PC geometry
('real', pc_geom_patch.py), arms at the capture pose, the v5 box about the real spot. Tests whether the PC geometry
gap explains the real-frame offset of zed_v5 (docs/real/ZED_MAP_20261114.md). Sim only.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/perception/zed_geom_check.py --geom real --n 2048 --out results/placement/zed_map/geom_real.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/perception'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', choices=('sim', 'real'), default='sim')
    ap.add_argument('--n', type=int, default=2048)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--keys', nargs='+', default=['v1:zed:zed_v1_gate_arm', 'v5::zed_v5_gate'])
    ap.add_argument('--range', type=float, nargs=3, default=(.04, .04, 20.))
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.geom == 'real':
        import pc_geom_patch
        pc_geom_patch.apply()
    from blind_insertion.perception.render_data import ApproachRenderer
    from blind_insertion.perception.zed_estimator import ZedEstimator
    P = ROOT / 'runs/perception'
    nets = {}
    for k in a.keys:
        name, pose, gate = k.split(':')
        nets[name] = ZedEstimator('cuda:0', pose=str(P / f'{pose}.pt') if pose else None, presence=str(P / f'{gate}.pt'))
    r = ApproachRenderer(nworld=a.nworld, seed=a.seed, task_layout='results/placement/task_layout_real_v2.json', pc_pose_range=tuple(a.range))
    lab, est = [], {k: [] for k in nets}
    while sum(len(x) for x in lab) < a.n:
        img, lb = r.sample_top()
        lab.append(torch.cat([lb['port_xy'] * 1e3, torch.rad2deg(lb['port_yaw'])[:, None]], 1).cpu().numpy())
        for k, e in nets.items():
            p, y, ok, info = e(img)
            est[k].append(np.c_[p.cpu().numpy() * 1e3, np.degrees(y.cpu().numpy()), ok.cpu().numpy()])
    L = np.concatenate(lab)[:a.n]
    out = dict(geom=a.geom, n=len(L), range=a.range, nets={})
    print(f'geom {a.geom}  n {len(L)}')
    for k in nets:
        E = np.concatenate(est[k])[:a.n]
        d = E[:, :3] - L
        res = dict(gate_ok=int(E[:, 3].sum()), dx_mean=float(d[:, 0].mean()), dy_mean=float(d[:, 1].mean()), dyaw_mean=float(d[:, 2].mean()),
                   dx_med=float(np.median(d[:, 0])), dy_med=float(np.median(d[:, 1])), dyaw_med=float(np.median(d[:, 2])),
                   xy_p50=float(np.median(np.linalg.norm(d[:, :2], axis=1))), slope_y=float(np.polyfit(L[:, 1], E[:, 1], 1)[0]),
                   slope_x=float(np.polyfit(L[:, 0], E[:, 0], 1)[0]))
        bins = {}
        for lo in range(354, 434, 10):
            m = (L[:, 1] >= lo) & (L[:, 1] < lo + 10)
            if m.sum() >= 10:
                bins[f'{lo}-{lo + 10}'] = dict(n=int(m.sum()), dy_med=float(np.median(d[m, 1])), dx_med=float(np.median(d[m, 0])), ok=int(E[m, 3].sum()))
        res['by_y'] = bins
        out['nets'][k] = res
        print(k, json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in res.items() if kk != 'by_y'}))
        for b, v in bins.items():
            print(f'   y {b}: n {v["n"]:4d} dy med {v["dy_med"]:+.2f} dx med {v["dx_med"]:+.2f} gate {v["ok"]}')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
