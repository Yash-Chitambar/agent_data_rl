"""Every gate of the align stage and the insertion guards, without an arm or a GPU. Also: configs that do not enable the new stages keep the old behaviour.

    python -m deploy.checks.align_gates
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual.align import DEFAULTS, AlignLogic, PortMeasure, align_cfg  # noqa: E402
from deploy.bimanual.insert_guard import SeatDetector2  # noqa: E402

FAILED = []


def check(name, ok, info=''):
    print(('ok    ' if ok else 'FAILED') + f' {name} {info}', flush=True)
    if not ok:
        FAILED.append(name)


def M(a, u, g, conf=1., sig=.5):  # noqa: N802
    return PortMeasure(True, 't', conf, a, u, g, sig)


def logic(**kw):
    c = dict(DEFAULTS, probe_mm=[0, 0, 0])
    c.update(kw)
    return AlignLogic(c)


def drive(lg, v0, plant_slope=(1, 1, 1), n=60, seed=0, noise=.1):
    """Closed loop with a perfect hand: true error -> measurement -> step. Returns last decision, final true pos."""
    g = np.random.default_rng(seed)
    p, tip = np.array(v0, float), np.zeros(3)
    d = None
    for _ in range(n):
        m = M(*(np.asarray(plant_slope) * p + g.normal(0, noise, 3)))
        d = lg.update(m, None, tip)
        if d['kind'] in ('done', 'fail'):
            break
        p = p + d['delta']
        tip = tip + d['delta']
    return d, p


def main():
    # normal convergence
    lg = logic(probe_mm=[2, 2, 2])
    d, p = drive(lg, (1.5, 7., 11.5))
    check('converges from A-like start', d['kind'] == 'done' and abs(p[0]) < 1.2 and abs(p[1]) < 1.2 and abs(p[2] - 6) < 1.2, f'{d["kind"]} {p.round(2)} {d.get("reason")}')
    # scale gate: net slopes reject
    lg = logic(probe_mm=[2, 2, 2])
    d, p = drive(lg, (1.5, 7., 11.5), plant_slope=(.34, .17, .35))
    check('net slope rejected', d['kind'] == 'fail' and 'scale' in d['reason'], d.get('reason'))
    # a frozen camera stops right after the probes
    lg = logic(probe_mm=[2, 2, 2])
    tip = np.zeros(3)
    for _ in range(20):
        d = lg.update(M(1.5, 7., 11.5), None, tip)
        if d['kind'] == 'fail':
            break
        tip = tip + d['delta']
    check('frozen measurement stops after the probes', d['kind'] == 'fail' and 'after the probes' in d['reason'] and lg.cycle <= 7, f"{d.get('reason')} cycle {lg.cycle}")
    # already aligned with no motion: the scale cannot be checked
    lg = logic(probe_mm=[0, 0, 0], extra_probes=0)
    d, p = drive(lg, (0., 0., 6.))
    check('no motion, no hand-over', d['kind'] == 'fail' and 'scale' in d['reason'], d.get('reason'))
    # no measurement
    lg = logic()
    for _ in range(int(DEFAULTS['max_no_measure'])):
        d = lg.update(PortMeasure(False, 'x'), None, np.zeros(3))
    check('no measurement stops', d['kind'] == 'fail', d.get('reason'))
    # low confidence
    lg = logic()
    for _ in range(int(DEFAULTS['max_no_measure'])):
        d = lg.update(M(0, 0, 6, conf=.1), None, np.zeros(3))
    check('low confidence stops', d['kind'] == 'fail', d.get('reason'))
    # sources disagree
    lg = logic(extra_probes=0)
    lg.update(M(0, 0, 6), M(0, 5, 6), np.zeros(3))
    d = lg.update(M(0, 0, 6), M(0, 5, 6), np.zeros(3))
    check('sources disagree at the hand-over stops', d['kind'] == 'fail' and 'disagree' in d['reason'], d.get('reason'))
    lg = logic(extra_probes=0, secondary_required=True)
    lg.update(M(0, 0, 6), None, np.zeros(3))
    d = lg.update(M(0, 0, 6), None, np.zeros(3))
    check('no second source at the hand-over stops', d['kind'] == 'fail' and 'second source' in d['reason'], d.get('reason'))
    # step limit
    lg = logic(max_cycles=3)
    p = 0
    for _ in range(4):
        d = lg.update(M(0, 5, 8), None, np.zeros(3))
    check('step limit stops', d['kind'] == 'fail' and 'step limit' in d['reason'], d.get('reason'))
    # displacement limit
    lg = logic()
    d = lg.update(M(0, 5, 8), None, np.array([0, 25., 0]))
    check('displacement limit stops', d['kind'] == 'fail', d.get('reason'))
    # drift
    lg = logic(slope_se_max=1e9, extra_probes=0)
    d = lg.update(M(0, 0, 6), None, np.zeros(3))
    d = lg.update(M(0, 0, 6), None, np.array([0, 2., 0]))
    d = lg.update(M(0, 0, 6), None, np.array([0, 4., 0]))
    check('drift with no command stops', d['kind'] == 'fail' and 'drift' in d['reason'], d.get('reason'))
    # correction growing
    lg = logic(grow_abs_mm=100.)
    tip = np.zeros(3)
    d = None
    for i in range(10):
        d = lg.update(M(0, 3 + i * 1.0, 8), None, tip)
        if d['kind'] == 'fail':
            break
        tip = tip + np.array([0, 1.0, 0])
    check('growing correction stops', d['kind'] == 'fail' and 'grows' in d['reason'], d.get('reason'))
    # measurement against FK
    lg = logic()
    d = lg.update(M(0, 5, 9), None, np.zeros(3))
    d = lg.update(M(0, 12, 9), None, np.zeros(3))
    d = lg.update(M(0, 20, 9), None, np.zeros(3))
    check('measurement jumps against FK stops', d['kind'] == 'fail', d.get('reason'))
    # stuck hand
    lg = logic()
    for _ in range(int(DEFAULTS['stuck_n'])):
        d = lg.update(M(0, 5, 9), None, np.zeros(3), stuck=True)
    check('stuck hand stops', d['kind'] == 'fail' and 'move' in d['reason'], d.get('reason'))
    # gap floor, funnel
    lg = logic()
    d = lg.update(M(0, 0, 2), None, np.zeros(3))
    check('gap below clearance stops', d['kind'] == 'fail', d.get('reason'))
    lg = logic()
    d = lg.update(M(0, 6, 7), None, np.zeros(3))
    check('misaligned and close: backs out', d['kind'] == 'step' and d['delta'][2] > 0, str(d['delta']))
    lg = logic()
    d = lg.update(M(0, 6, 12), None, np.zeros(3))
    check('misaligned and far: closes only to the hold gap', d['kind'] == 'step' and d['delta'][2] < 0, str(d['delta']))
    lg = logic()
    d = lg.update(M(5, 0, 8.5), None, np.zeros(3))
    check('steps capped', np.abs(d['delta']).max() <= 1. + 1e-9, str(d['delta']))
    # grip
    lg = logic()
    lg.update(M(0, 5, 9), None, np.zeros(3), grip=.31)
    d = lg.update(M(0, 4, 9), None, np.zeros(3), grip=.27)
    check('gripper change stops', d['kind'] == 'fail', d.get('reason'))

    # seat 2 against the failed real runs and a real seat (values from the logs, results/real_align/seat_real.json)
    det = SeatDetector2(1, torch.device('cpu'), 12.04, [1.57, 7.26])
    for z in np.linspace(0, -10, 60):
        r = det.update(torch.tensor([[0., 0., z]]))
    check('seat 2: plug on the case, gap measured', not bool(r['seated'][0]), f"depth {float(r['depth'][0]):.1f}")
    det = SeatDetector2(1, torch.device('cpu'), 6., [0., 0.])
    for i in range(60):
        r = det.update(torch.tensor([[0., 13., -11.7]]))
    check('seat 2: A2 at +13 mm, wrong gap, lateral gate rejects', not bool(r['seated'][0]))
    det = SeatDetector2(1, torch.device('cpu'), 6., [0., 0.])
    for i in range(60):
        r = det.update(torch.tensor([[.2, .3, -11.7]]))
    check('seat 2: plug at depth, on axis', bool(r['seated'][0]))

    # configs
    pick3 = json.loads((ROOT / 'deploy/config/bimanual_pick3.json').read_text())
    base = json.loads((ROOT / 'deploy/config/bimanual.json').read_text())
    check('bimanual.json: stages off', align_cfg(base) is None and 'seat2' not in base['insert'] and 'guard' not in base['insert'])
    check('bimanual_pick3.json: stages off', align_cfg(pick3) is None and 'seat2' not in pick3['insert'] and 'guard' not in pick3['insert'])
    new = json.loads((ROOT / 'deploy/config/bimanual_align1.json').read_text())
    a = pick3['pick']['batches']['A']
    p = json.loads(json.dumps(pick3))
    p['pick']['grasp_v3'].update(a['grasp_v3'])
    p['safety']['body_clearance'].update(a['body_clearance'])
    p['pick']['batch'] = 'A'
    n = json.loads(json.dumps(new))
    for k in ('_about', 'align'):
        n.pop(k, None)
        p.pop(k, None)
    for k in ('wrist_net', 'fine_net'):       # r10 nets (docs/real/WRIST_NETS_REAL.md)
        check(f'align1 {k} is the r10 net', 'perception_real/wrist_r10' in n['chain_args'][k])
        n['chain_args'][k] = p['chain_args'][k]
    for k in ('seat2', 'guard', 'keep_reader_unless_seated', 'abort_lift_mm', 'abort_v_mm_s'):
        n['insert'].pop(k, None)
    check('bimanual_align1.json = pick3 + batch A + the new keys, limits equal', n == p, '' if n == p else str({k: (n.get(k), p.get(k)) for k in n if n.get(k) != p.get(k)})[:400])
    # code paths: without the keys, run_insertion's old path is taken
    src = (ROOT / 'deploy/bimanual/insert.py').read_text()
    check('insert.py: guards only behind the config keys', "if (ic.get('guard') or ic.get('seat2')) else (None, None)" in src)
    src = (ROOT / 'deploy/run_bimanual.py').read_text()
    check('run_bimanual.py: no-seat branch only behind the config key', "self.cfg['insert'].get('keep_reader_unless_seated')" in src)
    ast.parse(src)
    print('FAILED:', FAILED or 'none', flush=True)
    return not FAILED


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
