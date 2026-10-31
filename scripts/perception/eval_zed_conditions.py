"""ZED pose nets by arm condition (sim, policy/v1 visual DR, reader on the Part 1 strip), n per condition.

Conditions: rest (reference arms, right fingers fully open = rounds 1-4 training), rest_24mm (right fingers
at the pick's 24 mm opening), parked (left arm at Part 1's park pose, right fingers 24 mm = the chain's
start), pressing (left arm pressing the PC top 2 cm in front, right at rest), and wild (vary_arms 2.5).

    PYTHONPATH=. .venv/bin/python scripts/perception/eval_zed_conditions.py --nets zed zed_v3s0 zed_v3s1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch

from blind_insertion.perception.nets import PoseNet
from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.perception.zed_estimator import NOMINAL_PORT_XY, zed_crop

LAYOUT = 'results/usbc/policy/pick/task_layout.json'


def load(name, dev):
    ck = torch.load(f'runs/perception/{name}.pt', map_location=dev)
    net = PoseNet(ck.get('n_out', len(ck['scale'])), ck['size'], width=ck.get('width', 32)).to(dev).eval()
    net.load_state_dict(ck['net'])
    return net, ck['scale'].to(dev)


def frames(r, cond):
    rt, n, dev = r.rt, r.n, r.dev
    r.h.reset()
    rf = [int(rt.mjm.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
    la = rt.indices['left'][0]
    if cond == 'rest':
        rt.q[:, rf] = .0475
    elif cond == 'rest_24mm':
        rt.q[:, rf] = .024
    elif cond == 'parked':
        rt.q[:, rf] = .024
        rt.q[:, la] = torch.tensor(r.LEFT_PARK, device=dev)
    elif cond == 'pressing':
        rt.q[:, rf] = .0475
        r.hold.cfg.along_m = -.02
        p, R = r.h.pc_pose()
        rt.q[:, la] = r.hold.solve(p, R)['q_contact']
        rt.q[:, r.left_fingers] = 0.
    elif cond == 'wild':
        r.vary_arms(2.5)
    with wp.ScopedDevice(rt.cfg.device):
        mjw.forward(rt.m, rt.d)
    p, R = r.h.pc_pose()
    return zed_crop(rt.render()['top']).contiguous(), p[:, :2], torch.atan2(-R[:, 1, 2], -R[:, 0, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['zed'])
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--out', default='results/usbc/policy/approach/zed_conditions.json')
    ap.add_argument('--gate', nargs=3, default=None, metavar=('POSE', 'POSE2', 'GATE'), help='also evaluate ZedEstimator(pose, gate, pose2)')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    nets = {nm: load(nm, dev) for nm in a.nets}
    est = None
    if a.gate:
        from blind_insertion.perception.zed_estimator import ZedEstimator
        est = ZedEstimator(str(dev), pose=f'runs/perception/{a.gate[0]}.pt', presence=f'runs/perception/{a.gate[2]}.pt',
                           pose2=f'runs/perception/{a.gate[1]}.pt' if a.gate[1] != 'none' else None)
    r = ApproachRenderer(nworld=256, seed=555, task_layout=LAYOUT)
    out = {}
    for cond in ('rest', 'rest_24mm', 'parked', 'pressing', 'wild'):
        errs = {nm: [] for nm in nets}; yerr = {nm: [] for nm in nets}; ens = []; gerr, gok = [], []
        for _ in range(a.n // r.n):
            x, pxy, yaw = frames(r, cond)
            if est is not None:
                full = torch.zeros(len(x), 320, 320, 3, dtype=torch.uint8, device=dev)
                from blind_insertion.perception.zed_estimator import ZED_CROP
                full[:, ZED_CROP[1]:ZED_CROP[1] + ZED_CROP[2], ZED_CROP[0]:ZED_CROP[0] + ZED_CROP[2]] = x
                port, _, ok, _ = est(full)
                gerr.append(((port - pxy).norm(dim=1) * 1e3).cpu()); gok.append(ok.cpu())
            ests = []
            for nm, (net, sc) in nets.items():
                with torch.no_grad():
                    e = net(x)[:, :len(sc)] * sc
                port = torch.tensor(NOMINAL_PORT_XY, device=dev) + e[:, :2] * 1e-3
                errs[nm].append(((port - pxy).norm(dim=1) * 1e3).cpu())
                yerr[nm].append(torch.rad2deg(torch.atan2(torch.sin(torch.deg2rad(e[:, 2]) - yaw), torch.cos(torch.deg2rad(e[:, 2]) - yaw))).abs().cpu())
                ests.append(port)
            if len(ests) > 1:
                ens.append(((torch.stack(ests[-2:]).mean(0) - pxy).norm(dim=1) * 1e3).cpu())
        out[cond] = {nm: {'xy_mm_p50': float(np.percentile(torch.cat(errs[nm]), 50)), 'xy_mm_p99': float(np.percentile(torch.cat(errs[nm]), 99)),
                          'yaw_deg_p99': float(np.percentile(torch.cat(yerr[nm]), 99))} for nm in nets}
        if ens:
            e = torch.cat(ens)
            out[cond]['ensemble_last2'] = {'xy_mm_p50': float(np.percentile(e, 50)), 'xy_mm_p99': float(np.percentile(e, 99))}
        if gerr:
            e, k_ = torch.cat(gerr), torch.cat(gok)
            out[cond]['gated_estimator'] = {'accept': float(k_.float().mean()),
                                            'xy_mm_accepted_p50': float(np.percentile(e[k_], 50)) if k_.any() else None,
                                            'xy_mm_accepted_p99': float(np.percentile(e[k_], 99)) if k_.any() else None,
                                            'xy_mm_rejected_p50': float(np.percentile(e[~k_], 50)) if (~k_).any() else None}
        print(cond, json.dumps(out[cond]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
