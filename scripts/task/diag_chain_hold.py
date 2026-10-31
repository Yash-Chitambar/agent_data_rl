"""Diagnose the Part 3 hold inside the chain (scripts/eval/eval_chain.py): reset -> Part 1 expert pick ->
guarded press, with elliptic cones + impratio 30 as the chain and Part 4 run. Reports, per press point:
arm-arm contacts during the engage (which geoms), PC sink / translation / rotation at the engage, and
optionally PC-still under a fixed axial push at the port after the engage (the right arm keeps the reader
at the lift pose; the push is applied to the PC as in eval_left_hold.py).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/task/diag_chain_hold.py --hold-along -0.035 --push 3
Writes results/usbc/policy/hold/chain_hold_<tag>.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts' / 'eval')); sys.path.insert(0, str(ROOT / 'scripts' / 'perception'))
import numpy as np
import torch
import warp as wp

import eval_chain as EC
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.left_hold import HoldCfg, LeftHold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hold-along', type=float, default=-.035)
    ap.add_argument('--hold-force', type=float, default=25.)
    ap.add_argument('--stand-up', type=float, default=0., help='extra stand-off height (m) above the press point before the guarded descent')
    ap.add_argument('--push', type=float, default=0.)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seeds', type=int, nargs='+', default=[7100])
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    ca = EC.parse(['--nworld', str(a.nworld), '--hold-along', str(a.hold_along), '--hold-force', str(a.hold_force)])
    ch = EC.Chain(ca)
    rt, h, dev, n = ch.rt, ch.h, ch.dev, ch.n
    m = rt.mjm
    side = h.geom_side.cpu().numpy()
    rows, pairs = [], {}
    for seed in a.seeds:
        ch.seed = seed
        torch.manual_seed(seed)
        if hasattr(rt.visual, 'gen'):
            rt.visual.gen.manual_seed(seed)
        rt.comp['right'] = None
        ch.pick.reset()
        p_true, R_true = h.pc_pose()
        log = {}
        qr = ch.stage_pick(log)
        yaws = tuple(ch.env.hold.cfg.yaws_deg)
        hcfg = HoldCfg(mode='press', force_n=a.hold_force, across_m=-.0208, along_m=a.hold_along, yaws_deg=yaws,
                       standoff_m=.015 + a.stand_up)
        hold = LeftHold(h, hcfg)
        plan = hold.solve(p_true, R_true)
        ql0 = h.believed('left').clone()
        pc0, R0 = h.pc_pose()
        aa = torch.zeros(n, dtype=torch.bool, device=dev)
        for k in range(int(round(hold.duration / ch.dt))):
            h.step(qr, -1., hold.target(k * ch.dt, ql0), -1.)
            if k % 5 == 0:
                c = h.contacts()
                if c['arm_arm'].any():
                    d = rt.d
                    nac = int(d.nacon.numpy()[0])
                    g = wp.to_torch(d.contact.geom)[:nac].long().cpu().numpy(); wid = wp.to_torch(d.contact.worldid)[:nac].cpu().numpy()
                    dist = wp.to_torch(d.contact.dist)[:nac].cpu().numpy()
                    for (g1, g2), w_, d_ in zip(g, wid, dist):
                        if d_ < 0 and ((side[g1] == 2) != (side[g2] == 2)) and (side[g1] > 0 or side[g2] > 0):
                            key = ' / '.join(sorted([f'{m.body(int(m.geom_bodyid[g1])).name}:{m.geom(int(g1)).name}', f'{m.body(int(m.geom_bodyid[g2])).name}:{m.geom(int(g2)).name}']))
                            pairs[key] = pairs.get(key, 0) + 1
                aa |= c['arm_arm']
        pc1, R1 = h.pc_pose()
        q_hold = hold.hold_target()
        dz = (pc1 - pc0)[:, 2] * 1e3
        still = torch.ones(n, dtype=torch.bool, device=dev)
        if a.push > 0:
            p0, Rp0 = h.pc_pose()
            dpm = torch.zeros(n, device=dev); angm = torch.zeros(n, device=dev)
            lat = (torch.rand(n, 2, device=dev) * 2 - 1) * .15
            for k in range(int(2. / ch.dt)):
                t = k * ch.dt
                pp, RR = h.pc_pose()
                f = -RR[:, :, 2] + RR[:, :, 0] * lat[:, :1] + RR[:, :, 1] * lat[:, 1:]
                h.push_pc(f * a.push * min(1., t / .5), pp)
                h.step(qr, -1., q_hold, -1.)
                pp, RR = h.pc_pose()
                dpm = torch.maximum(dpm, torch.nan_to_num((pp - p0).norm(dim=1), nan=1.))
                angm = torch.maximum(angm, torch.nan_to_num(K.rot_log(RR @ Rp0.transpose(1, 2)).norm(dim=1), nan=1.))
            h.push_pc(torch.zeros(n, 3, device=dev), p0)
            still = (dpm < 5e-4) & (torch.rad2deg(angm) < .5)
        rows.append({'pick_ok': log['pick_ok'].cpu().numpy(), 'arm_arm': aa.cpu().numpy(), 'latched': hold.latched.cpu().numpy(),
                     'sink_mm': dz.cpu().numpy(), 'engage_mm': ((pc1 - pc0).norm(dim=1) * 1e3).cpu().numpy(),
                     'engage_deg': torch.rad2deg(K.rot_log(R1 @ R0.transpose(1, 2)).norm(dim=1)).cpu().numpy(),
                     'still': still.cpu().numpy(), 'j2_nm': (plan['dq_force'][:, 1] * h.kp[1, 1]).abs().cpu().numpy()})
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    q = lambda x, p: float(np.nanpercentile(x, p))
    out = {'hold_along_m': a.hold_along, 'hold_force_n': a.hold_force, 'stand_up_m': a.stand_up, 'n': int(len(cat['arm_arm'])),
           'solver': 'elliptic cones, impratio 30 (InsertionCfg physics)',
           'arm_arm_worlds': int(cat['arm_arm'].sum()), 'arm_arm_pairs_samples': pairs, 'latched': float(cat['latched'].mean()),
           'pick_ok': float(cat['pick_ok'].mean()),
           'pc_sink_mm': {f'p{p}': q(-cat['sink_mm'], p) for p in (1, 50, 99)},
           'engage_translation_mm': {f'p{p}': q(cat['engage_mm'], p) for p in (50, 99, 100)},
           'engage_rotation_deg': {f'p{p}': q(cat['engage_deg'], p) for p in (50, 99, 100)},
           'push_n': a.push, 'pc_still_rate': float(cat['still'].mean()) if a.push > 0 else None,
           'j2_cmd_nm_max': float(cat['j2_nm'].max())}
    tag = a.tag or f"a{abs(a.hold_along) * 1e3:.0f}_f{a.hold_force:.0f}_p{a.push:g}" + (f"_up{a.stand_up * 1e3:.0f}" if a.stand_up else '')
    Path(ROOT / 'results/usbc/policy/hold').mkdir(parents=True, exist_ok=True)
    (ROOT / f'results/usbc/policy/hold/chain_hold_{tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
