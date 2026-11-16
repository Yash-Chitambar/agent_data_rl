"""Reach, joint-limit and clearance check of both arms over the wide-placement grid (docs/real/WIDE_PLACEMENT_READY_20261116.md).

For each dry run of scripts/eval/placement_grid.py (run folder with ticks.npz), at the forced PC pose of that run:
  * joint margin: distance of every joint (measured and target, motor frame) to the SafeArm joint box of
    deploy/safety_task.FlexChecker (Menagerie range + safety.LIMIT_BUFFER, rig offsets), per arm, min over the run;
  * table margin: FlexChecker body clearances (config safety.*), lowest body minus its limit, per arm;
  * arm-arm: smallest distance between any right-arm geom and any left-arm geom (scripts/real/arm_clearance.TwoArm
    geometry through scripts/real/parallel_clearance.Geo);
  * arm-PC: smallest distance of the right arm (all geoms) and of the left arm without its gripper and fingers (the
    left press touches the PC on purpose) to the PC case at the run's true pose;
  * planned moves the dry run teleports (SimIO): capture (start -> ZED capture pose, right then left) and home
    (left first, then right, to the start pose) as straight joint-space paths, the start pose of a real run
    (2026-11-13 full_165400_all9_full_1). Arm-arm and arm-PC on those paths.
All numbers are minima per run and per stage group. Stride: every 5th tick (10 Hz) for the ticks, 50 samples per move.

    PYTHONPATH=. python scripts/eval/wide_clearance.py --grid results/placement/wide/grid --out results/placement/wide/clearance.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
STAGES = ('0 capture', '1 pick', '2 hold', '3 transport', '4 approach', '5 insert', '6 release', '7 home', 'manual', 'align')
GROUP = {'1 pick': 'pick', '2 hold': 'hold', '3 transport': 'transport', '4 approach': 'approach', 'align': 'align',
         '5 insert': 'insert', '6 release': 'release'}
START_RUN = Path('/home/yashchitambar/peg_test/data/real_runs/2026-11-13/full_165400_all9_full_1')
LEFT_SKIP = ('gripper', 'finger', 'pad', 'tip')      # left bodies that press the PC on purpose


def checkers(cfg):
    from deploy.safety_task import FlexChecker
    from deploy import conventions as C
    sc = cfg['safety']
    return {s: FlexChecker(C.joint_offsets(s), clearance=sc['clearance'], tip_clearance=sc['tip_clearance'],
                           body_clearance=sc.get('body_clearance'), table_z=sc['table_z']) for s in ('right', 'left')}


def table_margin(chk, q6):
    """smallest (body bottom - its limit) in m over every checked geom, and the body."""
    chk.d.qpos[:] = 0
    chk.d.qpos[chk.qadr] = np.asarray(q6, float) + chk.off
    import mujoco
    mujoco.mj_kinematics(chk.m, chk.d)
    best, who = 9., None
    for g, verts, zmin in chk.geoms:
        z = chk._bottom(g, verts) - zmin
        if z < best:
            best, who = z, mujoco.mj_id2name(chk.m, mujoco.mjtObj.mjOBJ_BODY, chk.m.geom_bodyid[g])
    return best, who


def joint_margin(chk, q6):
    q6 = np.asarray(q6, float)
    m = np.minimum(q6 - chk.lo, chk.hi - q6)
    j = int(np.argmin(m))
    return float(m[j]), j + 1


def run_one(dp, d, G, chks, q_start, q_cap, stride=5, pc_real=None):
    """dp: the grid point folder (point.json); d: its run folder (events.jsonl, ticks.npz). pc_real: real run only,
    ((x, y) m, yaw deg) of the arm-derived port (results/placement/zed_map/arm_port.json).
    Distances use the TRUE sim arm angles (motor + rig offsets + the run's encoder-offset DR draw, meta sim_truth0)
    and the true PC pose; joint and table margins use the motor frame (what the rig's SafeArm checks)."""
    from parallel_clearance import load_run
    ev, _, tk = load_run(d)
    if tk is None:
        return None
    if pc_real is not None:
        p = dict(pc=[pc_real[0][0], pc_real[0][1], pc_real[1]], reader=None, seed=None, config='real run')
        pc, enc = pc_real, np.zeros((2, 6))
    else:
        p = json.loads((dp / 'point.json').read_text())
        zed = next((e for e in ev if e['kind'] == 'estimate' and e.get('est') == 'zed_pc'), None)
        if zed is not None and 'true_port_xy' in zed:
            pc = (tuple(zed['true_port_xy']), float(zed['true_yaw_deg']))
        else:
            pc = ((p['pc'][0], p['pc'][1]), p['pc'][2])
        meta = json.loads((d / 'meta.json').read_text()) if (d / 'meta.json').exists() else {}
        enc = np.asarray((meta.get('sim_truth0') or {}).get('enc_error', np.zeros((2, 6))), float)
    G.place(pc=pc, reader=None)
    gl_body = [g for g in G.g['left'] if not any(k in (G.bname(g) or '') for k in LEFT_SKIP)]
    out = {}

    def upd(grp, key, val, extra=None):
        o = out.setdefault(grp, {})
        if key not in o or val < o[key][0]:
            o[key] = (float(val), extra)

    def pose_checks(grp, qr, ql, with_chk=True):
        if with_chk:
            for s, q in (('right', qr), ('left', ql)):
                jm, j = joint_margin(chks[s], q)
                upd(grp, f'joint_{s}', jm, j)
                tm, b = table_margin(chks[s], q)
                upd(grp, f'table_{s}', tm, b)
        G.pose(np.asarray(qr, float) + enc[0], np.asarray(ql, float) + enc[1])
        x, pair = G.dmin(G.g['right'], G.g['left'])
        upd(grp, 'arm_arm', x, None if pair is None else [G.bname(pair[0]), G.bname(pair[1])])
        x, pair = G.dmin(G.g['right'], G.gPC)
        upd(grp, 'right_pc', x, None if pair is None else G.bname(pair[0]))
        x, pair = G.dmin(gl_body, G.gPC)
        upd(grp, 'left_body_pc', x, None if pair is None else G.bname(pair[0]))

    st = tk['stage']
    for k in range(0, len(st), stride):
        grp = GROUP.get(STAGES[st[k]] if 0 <= st[k] < len(STAGES) else '', None)
        if grp is None:
            continue
        for key in ('q', 'tgt'):
            pose_checks(grp, tk[key][k, 0], tk[key][k, 1])
    # planned moves the dry run teleports: capture (right then left) and home (left then right)
    lin = lambda a, b, n=50: [a + (b - a) * t for t in np.linspace(0., 1., n)]  # noqa: E731
    for q in lin(q_start['right'], q_cap['right']):
        pose_checks('capture_move', q, q_start['left'], with_chk=False)
    for q in lin(q_start['left'], q_cap['left']):
        pose_checks('capture_move', q_cap['right'], q, with_chk=False)
    qr_end, ql_end = tk['q'][-1, 0], tk['q'][-1, 1]
    for q in lin(ql_end, q_start['left']):
        pose_checks('home_move', qr_end, q, with_chk=False)
    for q in lin(qr_end, q_start['right']):
        pose_checks('home_move', q, q_start['left'], with_chk=False)
    return dict(tag=d.name, pc=p['pc'], reader=p['reader'], seed=p['seed'], config=p['config'],
                last_stage=STAGES[int(st[-1])] if 0 <= st[-1] < len(STAGES) else None,
                mins={g: {k: [round(v[0] * 1e3, 1) if not k.startswith('joint') else round(v[0], 3), v[1]] for k, v in o.items()}
                      for g, o in out.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', default=str(ROOT / 'results/placement/wide/grid'))
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align11_wide_k7.json'))
    ap.add_argument('--out', default=str(ROOT / 'results/placement/wide/clearance.json'))
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--shard', default='0/1', help='i/n: only every n-th grid folder from i (parallel processes; merge the outputs)')
    ap.add_argument('--real', nargs='*', default=None, help='real run folders instead of the grid (PC at the arm-derived port)')
    a = ap.parse_args()
    from parallel_clearance import Geo
    from deploy import conventions as C
    cfg = json.loads(Path(a.config).read_text())
    chks = checkers(cfg)
    G = Geo()
    q_start = {s: np.load(START_RUN / f'moves_{s}.npz')['measured'][0, :6] for s in ('right', 'left')}
    cap = json.loads((ROOT / 'blind_insertion/configs/zed_capture_pose.json').read_text())['model_frame_rad']
    q_cap = {s: np.asarray(cap[s], float) - C.joint_offsets(s) for s in ('right', 'left')}   # believed -> motor
    rows = []
    if a.real is not None:
        ap_ = {r['run']: r for r in json.loads((ROOT / 'results/placement/zed_map/arm_port.json').read_text())['runs']}
        for d in map(Path, a.real):
            if d.name not in ap_ or not (d / 'ticks.npz').exists():
                print(d.name, 'skipped (no arm port or ticks)')
                continue
            e = ap_[d.name]
            r = run_one(d, d, G, chks, q_start, q_cap, a.stride, pc_real=((e['port_mm'][0] * 1e-3, e['port_mm'][1] * 1e-3), e['yaw_deg']))
            r['tag'] = f'{d.parent.name}/{d.name}'
            rows.append(r)
            print(r['tag'], {g: {k: v[0] for k, v in o.items()} for g, o in r['mins'].items() if g in ('hold', 'insert')}, flush=True)
        Path(a.out).write_text(json.dumps(dict(method=__doc__, n=len(rows), rows=rows), indent=1, default=str))
        return
    si, sn = map(int, a.shard.split('/'))
    for d in sorted(Path(a.grid).iterdir())[si::sn]:
        if not (d / 'point.json').exists() or not (d / 'result.json').exists():
            continue
        runs = [x.parent for x in d.rglob('ticks.npz')]
        if not runs:
            continue
        cache = Path(a.out).with_suffix('') / f'{d.name}.json'      # per-run cache: reruns only do new runs
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        r = run_one(d, runs[0], G, chks, q_start, q_cap, a.stride)
        if r is None:
            continue
        r['tag'] = d.name
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(r, default=str))
        rows.append(r)
        print(d.name, {g: {k: v[0] for k, v in o.items()} for g, o in r['mins'].items() if g in ('hold', 'approach', 'insert')}, flush=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__, n=len(rows), rows=rows), indent=1, default=str))


if __name__ == '__main__':
    main()
