"""Collision-aware whole-chain clearance check over the wide PC box (docs/real/WIDE_MOTION_20261116.md). Kinematic,
CPU only, no sim physics: the chain's planned motions (deploy/bimanual/wide_motion.Planner) for each PC pose and
reader pose, on the bimanual scene model (deploy/bimanual/wide_motion.ClearModel: arm meshes, grippers, wrist
cameras, fingers, the reader in the right hand, the PC case, the table; SafeArm's FlexChecker for the table and the
joint box). Two motion sets per pose:
  today : the config without its wide_motion block (lift pose kept in the hold, Cartesian transport with the
          40 / 30 mm via, LeftHold over the policy yaws at along -0.02)
  new   : the config's wide_motion block (clear pose, joint_via transport, hold selection)
Per pose: pass / the failed floors / the first failing node (chain order: hold, transport, port (approach, align,
insert), release, home) and every minimum.

    PYTHONPATH=. python scripts/eval/wide_motion_check.py --config deploy/config/bimanual_align12_wide_k7.json \
        --out results/placement/wide_motion/check.json [--shard 0/4] [--reader 0.131 0.448 92]
    PYTHONPATH=. python scripts/eval/wide_motion_check.py --merge results/placement/wide_motion/check_s*.json --out results/placement/wide_motion/check.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NOMINAL_RUN = ROOT / 'results/placement/wide/grid/k7_pc+0_+0_+0_s7043/2026-11-15/full_211036_dry_s7043_k7_pc+0_+0_+0_s7043'
NOMINAL_READER = (0.131, 0.448, 92.0)          # the reader of that run (body x, y m; yaw deg)
START_RUN = Path('/home/yashchitambar/peg_test/data/real_runs/2026-11-13/full_165400_all9_full_1')
PC0 = (0.231, 0.394, 4.0)
NODE_ORDER = ('press', 'clear', 'swing', 'wait', 'transport', 'port', 'release', 'home')
NODE_NAME = dict(press='hold', clear='hold', swing='hold', wait='hold', transport='transport', port='approach/align/insert',
                 release='release', home='home')


def setup(config, wm_over=None):
    from deploy.bimanual.scene import SceneModel
    from deploy.bimanual.io_sim import chain_argv, eval_chain
    from deploy.bimanual import wide_motion as W
    EC = eval_chain()
    cfg = json.loads(Path(config).read_text())
    consts = json.loads((ROOT / 'deploy/bundle/configs/chain_constants.json').read_text())
    a = EC.parse(chain_argv(cfg, 'runs/insert_K7.pt', 1, []))
    sc = SceneModel('cpu')
    off = sc.encoder_offsets.numpy()
    if wm_over:
        cfg['wide_motion'].update(wm_over)
    wm_new = W.parse(cfg)
    if wm_new is None:
        raise SystemExit('config has no enabled wide_motion block')
    wm_today = dict(W.DEFAULTS, enabled=True)
    P = {'today': W.Planner(sc, a, consts, cfg['insert'], wm_today), 'new': W.Planner(sc, a, consts, cfg['insert'], wm_new)}
    M = W.ClearModel(sc, off, cfg['safety'])
    tk = np.load(NOMINAL_RUN / 'ticks.npz')
    i = int(np.where(tk['stage'] == 2)[0][0])
    q_lift0 = tk['q'][i, 0] + off[0]                     # motor -> believed
    from blind_insertion.envs.bimanual_pick import PickTaskCfg
    q_park = np.asarray(PickTaskCfg().left_park_q, float)
    q_home = {s: np.load(START_RUN / f'moves_{s}.npz')['measured'][0, :6] + off[k] for k, s in enumerate(('right', 'left'))}
    return W, sc, P, M, q_lift0, q_park, q_home


def lift_for_reader(P, q_lift0, reader):
    """Right lift pose for another reader pose: the nominal lift TCP moved rigidly with the reader (planar move and
    turn about the vertical through the reader body), IK seeded at the nominal."""
    ch = P.ch['right']
    q0 = torch.as_tensor(q_lift0, dtype=torch.float32)[None]
    p, R = ch.fk(q0)
    dyaw = math.radians(reader[2] - NOMINAL_READER[2])
    c, s = math.cos(dyaw), math.sin(dyaw)
    Rz = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    r0 = torch.tensor([NOMINAL_READER[0], NOMINAL_READER[1], 0.])
    r1 = torch.tensor([reader[0], reader[1], 0.])
    p1 = (Rz @ (p[0] - r0)) + r1
    q, pe, _ = ch.ik(p1[None], (Rz @ R[0])[None], q0, iters=150)
    return q[0].numpy(), float(pe[0])


def first_node(fails):
    best = None
    for f in fails:
        g = f.split()[1] if f.startswith(('arm-arm', 'reader-table')) else (f.split()[2] if f.startswith(('table', 'joint')) else
                                                                            (f.split()[2] if f.startswith('left body-PC') else 'press'))
        if g in NODE_ORDER and (best is None or NODE_ORDER.index(g) < NODE_ORDER.index(best)):
            best = g
    return NODE_NAME.get(best, best)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align12_wide_k7.json'))
    ap.add_argument('--out', default=str(ROOT / 'results/placement/wide_motion/check.json'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--dx', type=float, nargs='*', default=[-25, -12.5, 0, 12.5, 25])
    ap.add_argument('--dy', type=float, nargs='*', default=[-25, -12.5, 0, 12.5, 25])
    ap.add_argument('--dyaw', type=float, nargs='*', default=[-15, -10, -5, 0, 5, 10, 15])
    ap.add_argument('--reader', type=float, nargs=3, action='append', default=None,
                    help='reader body x y (m) yaw (deg); repeat for several (default the nominal reader)')
    ap.add_argument('--variants', nargs='*', default=['today', 'new'])
    ap.add_argument('--full', action='store_true', help='score every hold candidate completely (slow)')
    ap.add_argument('--merge', nargs='*')
    ap.add_argument('--wm', default=None, help='JSON: override keys of the wide_motion block (study only)')
    a = ap.parse_args()
    if a.merge:
        rows = []
        for f in a.merge:
            rows += json.loads(Path(f).read_text())['rows']
        Path(a.out).write_text(json.dumps(dict(n=len(rows), rows=rows), indent=1))
        return
    W, sc, P, M, q_lift0, q_park, q_home = setup(a.config, json.loads(a.wm) if a.wm else None)
    readers = [tuple(r) for r in (a.reader or [NOMINAL_READER])]
    pts = [(dx, dy, dyaw, rd) for rd in readers for dyaw in a.dyaw for dy in a.dy for dx in a.dx]
    si, sn = map(int, a.shard.split('/'))
    rows = []
    lifts = {}
    for dx, dy, dyaw, rd in pts[si::sn]:
        if rd not in lifts:
            lifts[rd] = lift_for_reader(P['new'], q_lift0, rd)
        q_lift, e_lift = lifts[rd]
        pc = ((PC0[0] + dx * 1e-3, PC0[1] + dy * 1e-3), PC0[2] + dyaw)
        ep, eR = sc.port_frame([pc[0]], [math.radians(pc[1])])
        row = dict(d=[dx, dy, dyaw], pc=[pc[0][0], pc[0][1], pc[1]], reader=list(rd), lift_ik_err_mm=e_lift * 1e3)
        for v in a.variants:
            t0 = time.time()
            r = W.select(P[v], M, ep, eR, q_lift, q_park, full=a.full, q_home=q_home)
            b = r['best']
            if v == 'new' and not a.full and r['ok']:      # the chosen candidate, scored completely (home included)
                plan = P[v].hold_plan(ep.cpu(), eR.cpu(), *r['choice'])
                rm = W.right_mins(M, r['paths'], q_park)
                b = W.score(P[v], M, plan, r['choice'], r['paths'], rm, q_park, quick=False, q_home=q_home)
            elif v == 'new' and not a.full:                 # no candidate passed: the best one, scored completely
                plan = P[v].hold_plan(ep.cpu(), eR.cpu(), *b['key'])
                rm = W.right_mins(M, r['paths'], q_park)
                b = W.score(P[v], M, plan, b['key'], r['paths'], rm, q_park, quick=False, q_home=q_home)
            row[v] = dict(choice=list(b['key']), n_scored=len(r['scored']), first_node=first_node(b['fails']) if b['fails'] else None,
                          transport_ik_err_mm=r['transport_ik_err_mm'], seconds=round(time.time() - t0, 1), **W.summary(b))
        rows.append(row)
        print(row['d'], rd, ' | '.join(f"{v}: {'PASS' if row[v]['ok'] else 'FAIL ' + str(row[v]['first_node'])} {row[v]['choice']} "
                                       f"{row[v]['fails'][:3]} ({row[v]['seconds']} s)" for v in a.variants), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(config=a.config, method=__doc__, n=len(rows), rows=rows), indent=1, default=str))


if __name__ == '__main__':
    main()
