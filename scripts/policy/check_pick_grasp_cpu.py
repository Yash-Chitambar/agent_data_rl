"""CPU contact check of the Part 1 tipped (60 deg) side grasp (blind_insertion/scenes/reader_grasp.py).

For a few reader poses: IK to pre-grasp / grasp (vertical tool axis), descend open, close, lift
8 cm, hold 2 s. Reports which pad spheres touch the side faces (rows from the tip), pad contact
height range on the 8.75 mm face, fingertip-table clearance before closing, reader tilt after
lift and in-hand drift. Nominal physics (no randomization); the GPU expert eval covers DR.

    PYTHONPATH=. .venv/bin/python scripts/policy/check_pick_grasp_cpu.py
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from blind_insertion.scenes import pc_bench as S, physical_rig as R, reader_grasp as G

OUT = Path('results/usbc/policy/pick/cpu_grasp_check.json')
profile = json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m, l = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=profile,
                     measured_support=True, controller_profile='native_free_settle_20261028')
qa = [int(m.joint(f'joint{i}').qposadr[0]) for i in range(1, 7)]
acts = [m.actuator(f'joint{i}').id for i in range(1, 7)]
grips = [m.actuator('grip_' + s).id for s in ('left', 'right')]
lim = np.array([m.joint(f'joint{i}').range for i in range(1, 7)])
tcp = l.tcp_site
pad = {g: m.geom(g).name for g in range(m.ngeom) if m.geom(g).name.startswith(('left_pad_s', 'right_pad_s'))}
tips = {m.geom('left_tip_hull').id, m.geom('right_tip_hull').id}
table = m.geom('table').id
top = float(m.geom('table').pos[2] + m.geom('table').size[2])
_, _, _, ht = G.reader_dims(l)
_d = mujoco.MjData(m); R.reference_pose(m, _d, l); mujoco.mj_forward(m, _d)
HAND = G.hand_geometry(m, _d)
OFFSET, PAD_H = G.grasp_offset(l, HAND, G.TILT_DEG)


def rot(q):
    r = np.zeros(9); mujoco.mju_quat2Mat(r, q); return r.reshape(3, 3)


def ik(d, p, Rt, q0):
    s = mujoco.MjData(m); s.qpos[:] = d.qpos
    def res(q):
        s.qpos[qa] = q; mujoco.mj_kinematics(m, s)
        return np.r_[s.site_xpos[tcp] - p, .1 * Rotation.from_matrix(s.site_xmat[tcp].reshape(3, 3) @ Rt.T).as_rotvec()]
    best = None
    for start in [q0, *np.random.default_rng(0).uniform(lim[:, 0], lim[:, 1], (6, 6))]:
        f = least_squares(res, np.clip(start, lim[:, 0] + 1e-6, lim[:, 1] - 1e-6), bounds=(lim[:, 0], lim[:, 1]), max_nfev=300)
        if best is None or np.linalg.norm(f.fun) < np.linalg.norm(best.fun): best = f
        if np.linalg.norm(best.fun) < 1e-6: break
    return best.x, float(np.linalg.norm(best.fun[:3]))


def run(xy, yaw):
    d = mujoco.MjData(m); R.reference_pose(m, d, l); S.place_pc(m, d, port_xy=(.22, .36)); S.place_reader(m, d, l, xy=xy, yaw_deg=yaw)
    for pre in ('', 'left_'):
        for i in range(1, 7): d.ctrl[m.actuator(pre + f'joint{i}').id] = d.qpos[m.joint(pre + f'joint{i}').qposadr][0]
        for s in ('left', 'right'): d.ctrl[m.actuator(pre + 'grip_' + s).id] = 4.
    mujoco.mj_forward(m, d)
    pr, Rr = d.qpos[l.peg_qadr:l.peg_qadr + 3].copy(), rot(d.qpos[l.peg_qadr + 3:l.peg_qadr + 7])
    import torch
    Wt = G.grasp_poses(torch.tensor(pr[None]), torch.tensor(Rr[None]), OFFSET, G.TILT_DEG)
    W = {k: (v[0][0].numpy(), v[1][0].numpy()) for k, v in Wt.items()}
    q_pre, e1 = ik(d, *W['pregrasp'], d.qpos[qa].copy())
    q_g, e2 = ik(d, *W['grasp'], q_pre)
    q_l, e3 = ik(d, *W['lift'], q_g)
    out = {'xy': list(xy), 'yaw_deg': yaw, 'ik_err_mm': [e1 * 1e3, e2 * 1e3, e3 * 1e3]}
    last = d.qpos[qa].copy(); integ = np.zeros(6)
    seq = [('pregrasp', q_pre, 2.5, 4.), ('descend', q_g, 1.5, 4.), ('settle', q_g, 1.0, 4.),
           ('close', q_g, 1.0, -12.), ('lift', q_l, 2.5, -12.), ('hold', q_l, 2.0, -12.)]
    for name, tgt, dur, grip in seq:
        n = int(dur / m.opt.timestep)
        for k in range(n):
            t = min(1, (k + 1) / (.8 * n)); b = t * t * (3 - 2 * t)
            if k % 20 == 0 and name in ('settle', 'hold', 'close'):     # settle-before-close: joint integral
                integ = np.clip(integ + .3 * (tgt - d.qpos[qa]), -.08, .08)
            d.ctrl[acts] = last + (tgt - last) * b + integ
            d.ctrl[grips] = grip
            mujoco.mj_step(m, d)
        last = tgt
        mujoco.mj_forward(m, d)
        if name == 'settle':
            tipz = min(d.geom_xpos[g][2] for g in tips)
            dist = []
            for g in tips:   # signed distance tip hull - table
                ft = np.zeros(6); dist.append(mujoco.mj_geomDistance(m, d, g, table, .05, ft))
            pt, Rt = d.site_xpos[tcp], d.site_xmat[tcp].reshape(3, 3)
            out['tip_table_clearance_mm'] = float(min(dist) * 1e3)
            out['tcp_pos_err_mm'] = float(np.linalg.norm(pt - W['grasp'][0]) * 1e3)
            out['tool_axis_tilt_deg'] = float(np.degrees(np.arccos(np.clip(-Rt[2, 2], -1, 1))))
        if name == 'close':
            rows, heights, nf = set(), [], 0
            forces, others = [], []
            for ci in range(d.ncon):
                c = d.contact[ci]
                gs = {int(c.geom1), int(c.geom2)}
                gp = [g for g in gs if g in pad]
                f = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f)
                if gp and gs & set(l.peg_slots):
                    nm = pad[gp[0]]; rows.add((nm.split('_')[0], nm[-2:]))
                    heights.append(float(c.pos[2] - top)); forces.append(float(f[0]))
                elif gs & set(l.peg_slots):
                    others.append((m.geom(int(c.geom1)).name, m.geom(int(c.geom2)).name, round(float(f[0]), 3)))
            out['pad_normal_force_N'] = [round(x, 2) for x in forces]
            out['other_reader_contacts'] = others
            out['pad_rows_in_contact'] = sorted({f'{a}:row{r}' for a, r in rows})
            out['pad_contact_height_mm'] = [float(min(heights) * 1e3), float(max(heights) * 1e3)] if heights else None
            out['n_pad_contacts'] = len(heights)
            out['finger_gap_mm'] = float(sum(d.qpos[m.joint(s + '_finger').qposadr][0] for s in ('left', 'right')) * 1e3)
            prc, Rrc = d.qpos[l.peg_qadr:l.peg_qadr + 3].copy(), rot(d.qpos[l.peg_qadr + 3:l.peg_qadr + 7])
            out['reader_shift_on_close_mm'] = float(np.linalg.norm(prc - pr) * 1e3)
            Rt, pt = d.site_xmat[tcp].reshape(3, 3).copy(), d.site_xpos[tcp].copy()
            rel0 = (Rt.T @ (prc - pt), Rt.T @ Rrc)
    pr2, Rr2 = d.qpos[l.peg_qadr:l.peg_qadr + 3], rot(d.qpos[l.peg_qadr + 3:l.peg_qadr + 7])
    Rt, pt = d.site_xmat[tcp].reshape(3, 3), d.site_xpos[tcp]
    out['lift_mm'] = float((pr2[2] - pr[2]) * 1e3)
    out['reader_tilt_deg'] = float(np.degrees(np.arccos(np.clip(Rr2[:, 1] @ [0, 0, 1], -1, 1))))
    out['inhand_drift_mm'] = float(np.linalg.norm(Rt.T @ (pr2 - pt) - rel0[0]) * 1e3)
    out['inhand_rot_drift_deg'] = float(np.degrees(np.linalg.norm(Rotation.from_matrix((Rt.T @ Rr2) @ rel0[1].T).as_rotvec())))
    out['success'] = bool(out['lift_mm'] > 60 and out['reader_tilt_deg'] < 3 and out['inhand_drift_mm'] < 2)
    return out


cases = [((.36, .10), 0.), ((.33, .05), 25.), ((.42, .15), -25.), ((.38, .06), 10.), ((.31, .16), -10.), ((.40, .12), 20.)]
res = [run(*c) for c in cases]
for r in res: print(json.dumps(r))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'grasp': {'axial_behind_neck_m': G.AXIAL_BEHIND_NECK, 'tip_clearance_m': G.TIP_CLEARANCE,
    'body_half_thickness_m': ht}, 'pad_contact_height_m': PAD_H, 'tilt_deg': G.TILT_DEG, 'cases': res, 'scope': 'CPU nominal physics, no DR'}, indent=1) + '\n')
print('success', sum(r['success'] for r in res), '/', len(res))
