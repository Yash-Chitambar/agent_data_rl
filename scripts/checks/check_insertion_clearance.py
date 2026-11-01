"""Clearance of the right gripper/arm against the PC case and the table at the insertion poses.

For the confirmed 60-deg grasp (insertion_task.inhand_pose, nominal and randomized) and sampled PC
poses, puts the plug at the pre-insert gap and fully seated (IK, CPU MuJoCo) and reports the smallest
distance between the right arm/gripper collision geoms and the PC / table, plus the reader-vs-PC
(non-plug) distance. Also sweeps the tool tilt (60 deg nominal) for the seated pose.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/checks/check_insertion_clearance.py
"""
import json
import math

import mujoco
import numpy as np
import torch

from blind_insertion.envs import insertion_task as IT


def distances(m, d, groups, pc_body, table_geom, cutoff=0.05):
    out = {}
    pc_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == pc_body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    for name, geoms in groups.items():
        best_pc, best_tab = cutoff, cutoff
        for g in geoms:
            best_tab = min(best_tab, mujoco.mj_geomDistance(m, d, g, table_geom, cutoff, None))
            for h in pc_geoms:
                best_pc = min(best_pc, mujoco.mj_geomDistance(m, d, g, h, cutoff, None))
        out[name] = dict(pc_mm=round(best_pc * 1e3, 2), table_mm=round(best_tab * 1e3, 2))
    return out


def main():
    n = 64
    report = {}
    for tilt in (60.0, 50.0, 40.0, 30.0):
        IT.TIP_DEG = tilt
        c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
        IT.GRASP_R = ((0., -1., 0.), (-s, 0., -c), (c, 0., -s))
        cfg = IT.InsertionCfg()
        cfg.task.physics.nworld = 4
        env = IT.InsertionTask(cfg, seed=5)
        rt, m = env.runtime, env.runtime.mjm
        lay = rt.lay
        d = mujoco.MjData(m)
        right = {m.body(b).id for b in ('left_finger', 'right_finger', 'gripper', 'link_6', 'link_5', 'link_4')}
        groups = {'fingers': [], 'gripper_housing': [], 'wrist_links': []}
        for g in range(m.ngeom):
            b = m.body(int(m.geom_bodyid[g])).name
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if b in ('left_finger', 'right_finger'):
                groups['fingers'].append(g)
            elif b == 'gripper':
                groups['gripper_housing'].append(g)
            elif b in ('link_6', 'link_5', 'link_4'):
                groups['wrist_links'].append(g)
        reader_body = [g for g in lay.peg_slots if m.geom(g).name in ('peg_33', 'peg_34')]
        res = {}
        for label, gap, nominal in (('preinsert_5mm_nominal', 0.005, True), ('seated_nominal', -env.seat_depth, True),
                                    ('seated_randomized', -env.seat_depth, False)):
            c0 = env.icfg
            if nominal:
                c0.inhand_pos_m, c0.inhand_rot_deg, c0.lateral_m, c0.angle_deg = (0, 0, 0), (0, 0, 0), (0, 0), 0.
            else:
                c0.inhand_pos_m, c0.inhand_rot_deg, c0.lateral_m, c0.angle_deg = (0.003, 0, 0.003), (1.5, 8, 1.5), (0, 0), 0.
            c0.gap_m = (gap, gap)
            st = env.sample_starts(n)
            worst = None
            mins = {k: dict(pc_mm=1e9, table_mm=1e9) for k in groups}
            mins['reader_body_vs_pc'] = dict(pc_mm=1e9, table_mm=1e9)
            for i in range(n):
                q = rt.initial_q.cpu().numpy().copy()
                q[rt.indices['right'][0]] = st['q_right'][i].cpu().numpy()
                q[rt.indices['left'][0]] = st['q_left_hover'][i].cpu().numpy()
                q[rt.pc_qadr:rt.pc_qadr + 7] = st['pc'][i].cpu().numpy()
                q[lay.peg_qadr:lay.peg_qadr + 7] = st['reader'][i].cpu().numpy()
                q[env.finger_q] = st['fingers'][i].cpu().numpy()
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                dist = distances(m, d, groups | {'reader_body_vs_pc': reader_body}, m.body('fixture').id, m.geom('table').id)
                for k, v in dist.items():
                    mins[k] = {kk: min(mins[k][kk], v[kk]) for kk in v}
            res[label] = dict(min_over_samples=mins, ik_failures=env.ik_failures, n=n)
        res['nominal_right_q'] = env.q_right_nominal.tolist()
        res['grasp'] = env.grasp_info
        report[f'tilt_{tilt:.0f}deg'] = res
        print(tilt, json.dumps(res, indent=1))
    out = 'results/usbc/policy/insertion/clearance_check.json'
    open(out, 'w').write(json.dumps(report, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
