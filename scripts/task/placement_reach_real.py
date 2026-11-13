"""Reach map around the real rig spot (no hardware, no physics): which PC and reader poses are kinematically feasible for
both arms, and with what margin.

PC (port x, y in the right-base frame, yaw in the ZED convention): scripts/task/explore_pc_placement.World.evaluate on
the deployed scene profile (the one deploy/bimanual/scene.SceneModel uses): right IK along the insertion line (seated,
6, 30, 60 mm out; the real chain stops at 18 and 6 mm) with in-hand tilts of +/-8 deg, left press/brace IK, joint
margin >= 0.05 rad, arm-arm / arm-PC / arm-table contact with a 5 mm margin. Output per pose: ok, the first failing
check, right and left joint margin (rad) and dexterity.

Reader (the ZED reader estimate's p = plug-tip frame origin, and its yaw): PickExpert.reset's plan (tilt 55 deg grasp,
plug-side flip, pre-grasp / grasp / lift IK from 4 seeds, every joint >= 0.05 rad inside its limits) on the tcp chain,
plus the reader footprint's clearance from the PC footprint at the real PC spot.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/task/placement_reach_real.py --out results/placement/reach
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/task'))

REAL_PC = (0.231, 0.394, 4.0)
REAL_READER = (0.131, 0.448, 92.0)


def _pc_job(args):
    import explore_pc_placement as E
    if 'w' not in E._W:
        E._W['w'] = E.World()
    r = E._W['w'].evaluate(args[:2], args[2])
    return dict(x=args[0], y=args[1], yaw=args[2], **{k: v for k, v in r.items() if k not in ('q_right', 'q_left')})


def pc_map(a):
    xs = np.round(np.arange(REAL_PC[0] - a.pc_dx, REAL_PC[0] + a.pc_dx + 1e-9, a.pc_step), 4)
    ys = np.round(np.arange(REAL_PC[1] - a.pc_dy, REAL_PC[1] + a.pc_dy + 1e-9, a.pc_step), 4)
    yaws = np.arange(REAL_PC[2] - a.pc_dyaw, REAL_PC[2] + a.pc_dyaw + 1e-9, a.yaw_step)
    jobs = [(float(x), float(y), float(t)) for x in xs for y in ys for t in yaws]
    with mp.get_context('fork').Pool(a.workers) as pool:
        rows = pool.map(_pc_job, jobs, chunksize=4)
    return dict(xs=xs.tolist(), ys=ys.tolist(), yaws=yaws.tolist(), rows=rows)


def reader_map(a):
    import torch
    from blind_insertion.envs import kinematics as K
    from blind_insertion.scenes import reader_grasp as G
    from deploy.bimanual.scene import SceneModel
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sc = SceneModel(device=dev)
    ch = sc.tcp_chain
    xs = np.round(np.arange(REAL_READER[0] - a.rd_dx, REAL_READER[0] + a.rd_dx + 1e-9, a.rd_step), 4)
    ys = np.round(np.arange(REAL_READER[1] - a.rd_dy, REAL_READER[1] + a.rd_dy + 1e-9, a.rd_step), 4)
    yaws = np.arange(a.rd_yaw[0], a.rd_yaw[1] + 1e-9, a.rd_yaw_step)
    P = np.array([(x, y, t) for x in xs for y in ys for t in yaws], float)
    n = len(P)
    t = torch.tensor(P, dtype=torch.float32, device=dev)
    R0 = K.quat_to_mat(sc._reader0[3:][None].to(dev))[0]
    yaw = torch.deg2rad(t[:, 2])
    c, s, z, o = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    Rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
    R = Rz @ R0
    p = torch.cat([t[:, :2], torch.full((n, 1), float(sc._reader0[2]), device=dev)], 1)
    plug = -R[:, :2, 2]
    flip = torch.cos(torch.atan2(plug[:, 1], plug[:, 0]) - torch.atan2(p[:, 1], p[:, 0])) < 0
    tilt = G.TILT_DEG
    off = torch.where(flip[:, None], torch.as_tensor(sc.offsets[(tilt, True)], device=dev, dtype=torch.float32),
                      torch.as_tensor(sc.offsets[(tilt, False)], device=dev, dtype=torch.float32))
    W = G.grasp_poses(p, R, off, tilt, flip=flip)
    aim = torch.atan2(p[:, 1], p[:, 0])
    lo, hi = ch.lo, ch.hi
    q_pre, e_pre = None, None
    for sd in ([1.0, 1.0, -.5, 0, 0], [1.6, 1.3, -.9, 0, 0], [2.2, 1.8, -1.2, 0, 0], [.6, .5, -.3, 0, 0]):
        seed = torch.cat([aim[:, None], torch.tensor(sd, device=dev).expand(n, 5)], 1)
        seed = torch.max(torch.min(seed, hi - .05), lo + .05)
        q, pe, re = ch.ik(*W['pregrasp'], seed, iters=80)
        e = pe + .05 * re
        if q_pre is None:
            q_pre, e_pre = q, e
        else:
            b = e < e_pre
            q_pre, e_pre = torch.where(b[:, None], q, q_pre), torch.where(b, e, e_pre)
    q_g, pe_g, re_g = ch.ik(*W['grasp'], q_pre, iters=80)
    q_l, pe_l, re_l = ch.ik(*W['lift'], q_g, iters=80)
    margin = lambda q: torch.minimum(q - lo, hi - q).min(1).values   # noqa: E731
    m = torch.stack([margin(q_pre), margin(q_g), margin(q_l)], 1).min(1).values
    conv = (e_pre < 1e-3) & (pe_g + .05 * re_g < 1e-3) & (pe_l + .05 * re_l < 1e-3)
    ok = conv & (m >= .05)
    # reader footprint vs the PC footprint at the real spot (explore_pc_placement geometry: reader -2..+7 cm about its
    # centre, half width 1.8 cm; the plug tip is 7 cm ahead of the centre)
    import explore_pc_placement as E
    w = E.World()
    cx, cy = w.pc_centre(REAL_PC[:2], REAL_PC[2])
    dep, wid = w.pc['depth']['value'], w.pc['width']['value']
    rows = []
    for i in range(n):
        x, y, tdeg = P[i]
        u = np.array([math.cos(math.radians(tdeg)), math.sin(math.radians(tdeg))])
        centre = np.array([x, y]) - .07 * u          # plug tip -> body centre
        pts = [centre + u * a_ + np.array([-u[1], u[0]]) * b_ for a_ in (-E.READER_BACK, E.READER_FRONT) for b_ in (-E.READER_HALF_W, E.READER_HALF_W)]
        cp, sp = math.cos(math.radians(REAL_PC[2])), math.sin(math.radians(REAL_PC[2]))
        loc = [np.array([[cp, sp], [-sp, cp]]) @ (q_ - [cx, cy]) for q_ in pts]
        # signed clearance of the reader box corners from the PC rectangle (positive = outside), conservative (corners)
        clear = min(max(abs(l_[0]) - dep / 2, abs(l_[1]) - wid / 2) for l_ in loc)
        rows.append(dict(x=float(x), y=float(y), yaw=float(tdeg), ok=bool(ok[i]), converged=bool(conv[i]), margin_rad=float(m[i]),
                         flip=bool(flip[i]), pc_clear_m=float(clear)))
    return dict(xs=xs.tolist(), ys=ys.tolist(), yaws=yaws.tolist(), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'results/placement/reach'))
    ap.add_argument('--pc-dx', type=float, default=.05)
    ap.add_argument('--pc-dy', type=float, default=.05)
    ap.add_argument('--pc-step', type=float, default=.01)
    ap.add_argument('--pc-dyaw', type=float, default=24.)
    ap.add_argument('--yaw-step', type=float, default=4.)
    ap.add_argument('--rd-dx', type=float, default=.06)
    ap.add_argument('--rd-dy', type=float, default=.06)
    ap.add_argument('--rd-step', type=float, default=.01)
    ap.add_argument('--rd-yaw', type=float, nargs=2, default=(40., 140.))
    ap.add_argument('--rd-yaw-step', type=float, default=5.)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--only', choices=('pc', 'reader'), default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.only != 'pc':
        rm = reader_map(a)
        (out / 'reader_map.json').write_text(json.dumps(rm))
        print('reader poses', len(rm['rows']), 'ok', sum(r['ok'] for r in rm['rows']), flush=True)
    if a.only != 'reader':
        pm = pc_map(a)
        (out / 'pc_map.json').write_text(json.dumps(pm))
        print('pc poses', len(pm['rows']), 'ok', sum(r['ok'] for r in pm['rows']), flush=True)


if __name__ == '__main__':
    main()
