"""Pre-insert / seated reachability and clearance of the confirmed 60-deg grasp (CPU, MuJoCo).

For PC poses over the training box (port nominal (0.22, 0.36) m, +/-3 cm, +/-10 deg: corners,
edge midpoints and centre = 27 poses) and grasp tilts from vertical (60 = confirmed), the right
arm is solved by IK for the reader at 10 mm, 6 mm (pre-insert), 0 mm and seated along the port axis,
with the in-hand extremes (+/-3 mm along the reader axis / vertically, +/-8 deg about the closing
axis). The left arm is at its press hold (explore_pc_placement press target). Reported per tilt:
IK success, joint margin, and the minimum signed distance of the right gripper/fingers/links to
the PC case, the table and the left arm (and of the reader body to the PC, excluding the plug).

    PYTHONPATH=. .venv/bin/python scripts/task/check_preinsert_grasp.py
Writes results/usbc/policy/approach/preinsert_grasp_clearance.json
"""
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_pc_placement import World, _rot  # noqa: E402
from blind_insertion.envs.inhand import sample_inhand, site_in_reader  # noqa: E402

GAPS = (.010, .006, 0., -.0058)
TILTS = (60., 55., 50., 45., 40.)
_W = {}


def _init():
    _W['w'] = World()
    w = _W['w']
    w.m.geom_margin[:] = np.maximum(w.m.geom_margin, .02)   # report distances up to 2 cm


def classify(w):
    m = w.m
    cat = {}
    for g in range(m.ngeom):
        b = m.body(int(m.geom_bodyid[g])).name
        if g in w.robot['right']:
            cat[g] = 'right'
        elif g in w.robot['left']:
            cat[g] = 'left'
        elif b == 'fixture':
            cat[g] = 'pc'
        elif m.geom(g).name == 'table':
            cat[g] = 'table'
        elif b == 'peg':
            z = m.geom_aabb[g].reshape(2, 3)[0, 2] + m.geom_pos[g, 2]
            cat[g] = 'plug' if z < .0067 else 'reader'
    return cat


def evaluate(args):
    (x, y, yaw), tilt = args
    w = _W['w']
    m, d = w.m, w.d
    cat = _W.setdefault('cat', classify(w))
    port_p, port_R = w.place_pc((x, y), yaw)
    # left arm at the press hold (best candidate, 3 mm gap as in the placement search)
    ql = None
    for mode, point in w.left_targets((x, y), yaw):
        if mode == 'press':
            ql, _ = w.ik_pad('left', point)
            if ql is not None:
                break
    out = {'pc': [x, y, yaw], 'tilt': tilt, 'left_ok': ql is not None}
    R_sr, p_sr = site_in_reader(tilt)
    Rs, ps, _ = sample_inhand(1, tilt)
    variants = [(R_sr.numpy(), p_sr.numpy())]
    for dz, dy, a in itertools.product((-3, 3), (-3, 3), (-8, 8)):
        from scipy.spatial.transform import Rotation
        Rr = Rotation.from_rotvec([np.radians(a), 0, 0]).as_matrix()
        pad = p_sr.numpy() - R_sr.numpy()[:, 2] * .0096
        variants.append((Rr.T @ R_sr.numpy(), Rr.T @ (p_sr.numpy() - pad) + pad - np.array([0, dy, dz]) * 1e-3))
    worst = {k: 1. for k in ('pc', 'table', 'left', 'reader_pc')}
    margin, ok, seed = 9., True, None
    for vi, (R_sr, p_sr) in enumerate(variants):
        for gap in GAPS:
            reader_R = port_R
            reader_p = port_p + port_R[:, 2] * gap
            R_t = reader_R @ R_sr
            p_t = reader_p + reader_R @ p_sr
            seeds = ([seed] if seed is not None else []) + list(w.arms['right']['seeds'])
            q = w.ik('right', p_t, R_t, seeds)
            if q is None:
                ok = False
                out.setdefault('fail', []).append([vi, gap])
                continue
            seed = q
            _, mg = w.dexterity('right', q)
            margin = min(margin, mg)
            d.qpos[w.arms['right']['qa']] = q
            if ql is not None:
                d.qpos[w.arms['left']['qa']] = ql
            pa = w.lay.peg_qadr
            d.qpos[pa:pa + 3] = reader_p
            qq = np.zeros(4); mujoco.mju_mat2Quat(qq, reader_R.ravel()); d.qpos[pa + 3:pa + 7] = qq
            mujoco.mj_forward(m, d)
            for c in d.contact[:d.ncon]:
                a_, b_ = cat.get(int(c.geom1)), cat.get(int(c.geom2))
                pair = {a_, b_}
                if 'right' in pair:
                    other = (pair - {'right'}) or {'right'}
                    o = other.pop()
                    if o in worst:
                        worst[o] = min(worst[o], float(c.dist))
                if pair == {'reader', 'pc'}:
                    worst['reader_pc'] = min(worst['reader_pc'], float(c.dist))
    out.update(ik_ok=ok, joint_margin_rad=margin, min_dist_m=worst)
    return out


def main():
    xs, ys, yaws = (.19, .22, .25), (.33, .36, .39), (-10., 0., 10.)
    jobs = [((x, y, t), tilt) for tilt in TILTS for x in xs for y in ys for t in yaws]
    with mp.Pool(12, initializer=_init) as pool:
        res = pool.map(evaluate, jobs)
    summary = {}
    for tilt in TILTS:
        r = [x for x in res if x['tilt'] == tilt]
        summary[f'{tilt:.0f}'] = {
            'n_pc_poses': len(r), 'ik_ok': sum(x['ik_ok'] for x in r), 'left_hold_ok': sum(x['left_ok'] for x in r),
            'min_joint_margin_rad': min(x['joint_margin_rad'] for x in r),
            'min_dist_mm': {k: 1e3 * min(x['min_dist_m'][k] for x in r) for k in ('pc', 'table', 'left', 'reader_pc')}}
    rep = {'method': __doc__.strip(), 'gaps_m': GAPS, 'summary_by_tilt_deg': summary, 'cases': res,
           'note': 'distances are signed (negative = penetration), clipped at 20 mm (geom margin); 9 in-hand variants per pose'}
    out = Path('results/usbc/policy/approach/preinsert_grasp_clearance.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
