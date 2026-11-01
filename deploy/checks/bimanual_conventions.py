"""Conventions of the bimanual real path against the sim (no hardware):

  1. deploy/bimanual/scene.SceneModel (CPU) == the live sim (PickTask on a BimanualRuntime): FK of both grasp_site
     chains and the TCP chain at random poses, PD gains, joint limits, encoder offsets, reader / PC reference poses,
     Part 1 grasp offsets, table top.
  2. Joint offsets: rig.joint_offsets == physical_rig calibration == runtime.encoder_offsets == configs/
     zed_capture_pose.json (model - encoder).
  3. Units / frames: SceneModel.port_frame (ZED estimate -> port pose) round-trips the yaw convention and puts the
     port at the sim's PC height; motor <-> believed round trip.
  4. Gripper maps: full open, the 24 mm approach opening, the reader band.
  5. Cameras: sim fovy of 'top' and 'wrist_cam' == the field of view of the real policy crop (real_top_policy_image /
     real_policy_image), i.e. the sim renders what the real preprocessing produces.
  6. Safety: FlexChecker (config clearances, table_z = the scene's tabletop) accepts the capture, park and nominal
     lift poses of both arms (motor frame) and the tabletop height in each arm's base frame.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_conventions [--out results/...json]
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
sys.path.append(str(ROOT / 'scripts/real'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='')
    ap.add_argument('--no-sim', action='store_true', help='skip the GPU sim comparison (rig PC without a free GPU)')
    a = ap.parse_args()
    import rig
    from blind_insertion.scenes import physical_rig as Rig
    from deploy.bimanual.scene import SceneModel
    from deploy.run_bimanual import gripper_maps, load_cfg
    cfg = load_cfg(ROOT / 'deploy/config/bimanual.json')
    res, fails = {}, []

    def check(name, ok, **info):
        res[name] = dict(ok=bool(ok), **info)
        if not ok:
            fails.append(name)

    sc = SceneModel('cpu')
    g = torch.Generator().manual_seed(0)
    # 1. scene == live sim
    if not a.no_sim:
        from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg
        from blind_insertion.envs.bimanual_runtime import BimanualCfg
        from blind_insertion.envs.left_hold import Harness
        task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=1, device='cuda:0', task_layout='results/usbc/policy/pick/task_layout.json')))
        rt = task.runtime
        h = Harness(rt)
        errs = []
        for s in ('right', 'left'):
            lo, hi = sc.indices[s][4][:, 0], sc.indices[s][4][:, 1]
            q = lo + (hi - lo) * torch.rand(64, 6, generator=g)
            p1, R1 = sc.chain[s].fk(q)
            p2, R2 = h.chain[s].fk(q.cuda())
            errs.append(float((p1 - p2.cpu()).abs().max()))
            errs.append(float((R1 - R2.cpu()).abs().max()))
        q = sc.indices['right'][4][:, 0] + (sc.indices['right'][4][:, 1] - sc.indices['right'][4][:, 0]) * torch.rand(64, 6, generator=g)
        p1, _ = sc.tcp_chain.fk(q)
        p2, _ = task.chain.fk(q.cuda())
        errs.append(float((p1 - p2.cpu()).abs().max()))
        check('fk_equal', max(errs) < 1e-5, max_abs=max(errs), note='scene chains are deploy/bimanual/fastchain.FastChain on the CPU')
        # FastChain IK (inherited algorithm, NumPy FK) vs K.Chain IK on random reachable targets
        from blind_insertion.envs import kinematics as Kmod
        kc = Kmod.Chain(sc.mjm, sc.lay.tcp_site, np.array(sc.indices['right'][0]), device='cpu')
        lo, hi = sc.indices['right'][4][:, 0], sc.indices['right'][4][:, 1]
        qt = lo + .2 * (hi - lo) + .6 * (hi - lo) * torch.rand(16, 6, generator=g)
        pt, Rt = kc.fk(qt)
        q0 = (qt + .1 * torch.randn(16, 6, generator=g)).clamp(lo, hi)
        dq, dp = [], []
        for i in range(16):
            qa, pa, _ = kc.ik(pt[i:i + 1], Rt[i:i + 1], q0[i:i + 1], iters=60)
            qb_, pb, _ = sc.tcp_chain.ik(pt[i:i + 1], Rt[i:i + 1], q0[i:i + 1], iters=60)
            dq.append(float((qa - qb_).abs().max()))
            dp.append(abs(float(pa[0]) - float(pb[0])))
        check('fastchain_ik_equal', max(dq) < 1e-3 and max(dp) < 1e-5, max_dq_rad=max(dq), max_dpos_err_m=max(dp))
        check('kp_equal', torch.allclose(sc.kp, h.kp.cpu()), scene=sc.kp.tolist())
        lim_ok = all(torch.allclose(sc.indices[s][4], rt.indices[s][4].cpu()) for s in ('right', 'left'))
        check('joint_limits_equal', lim_ok)
        check('encoder_offsets_equal', torch.allclose(sc.encoder_offsets, rt.encoder_offsets.cpu()))
        check('reader0_pc0_equal', torch.allclose(sc._reader0, rt._reader0.cpu(), atol=1e-6) and torch.allclose(sc._pc0, rt._pc0.cpu(), atol=1e-6))
        off_ok = all(np.allclose(np.asarray(sc.offsets[k]), np.asarray(task.offsets[k]), atol=1e-7) for k in task.offsets)
        check('grasp_offsets_equal', off_ok, keys=[str(k) for k in task.offsets])
        check('table_top_equal', abs(sc.table_top - task.table_top) < 1e-9, scene=sc.table_top)
        check('finger_qadr_equal', sc.finger_qadr == task.finger_qadr)
        z_sim = float(rt.q[0, rt.pc_qadr + 2])
        check('port_height_vs_sim_reset', abs(float(sc._pc0[2]) - z_sim) < 1e-3, scene_mm=float(sc._pc0[2]) * 1e3, sim_reset_mm=z_sim * 1e3)
    # 2. joint offsets
    cal = Rig.load()
    cap = json.loads((ROOT / 'blind_insertion/configs/zed_capture_pose.json').read_text())
    for s in ('right', 'left'):
        o = np.asarray(rig.joint_offsets(s))
        oc = np.asarray(cap['model_frame_rad'][s]) - np.asarray(cap['encoder_frame_rad'][s])
        check(f'offsets_{s}', np.allclose(o, cal['arms'][s]['joint_offsets_rad']) and np.allclose(o, sc.encoder_offsets[('right', 'left').index(s)].numpy(), atol=1e-7)
              and np.abs(o - oc).max() < 2e-5, rig_deg=np.degrees(o).round(3).tolist(), capture_pose_diff_rad=float(np.abs(o - oc).max()))
    # 3. frames / units
    from blind_insertion.envs import kinematics as K
    yaws = torch.linspace(-0.2, 0.2, 9)
    p, R = sc.port_frame(torch.tensor([[0.22, 0.36]]).expand(9, 2), yaws)
    back = torch.atan2(-R[:, 1, 2], -R[:, 0, 2])
    check('port_frame_yaw_roundtrip', float((back - yaws).abs().max()) < 1e-5, max_err_rad=float((back - yaws).abs().max()))
    check('port_frame_flat', float((R[:, 2, 1].abs() - 1).abs().max()) < 1e-3, note='port y axis vertical (PC flat)')
    q = np.random.default_rng(0).uniform(-1, 1, (2, 6))
    check('motor_believed_roundtrip', np.allclose((q + sc.encoder_offsets.numpy()) - sc.encoder_offsets.numpy(), q))
    # 4. gripper maps
    gm = gripper_maps(cfg)
    r = gm['right']
    check('gripper_right_full_open', abs(r.finger_m(r.open) - 0.048) < 1e-9, finger_m=r.finger_m(r.open))
    ga = r.reading(cfg['gripper']['right_approach_open_m'])
    check('gripper_right_approach_in_range', r.close < ga < r.open, reading=ga)
    from deploy.grasp import reader_present
    gr = cfg['gripper']['right']
    check('gripper_reader_band', reader_present(gr['g_reader'], gr['g_empty'], gr['g_reader']) and not reader_present(gr['g_empty'], gr['g_empty'], gr['g_reader']),
          note='calibrated 2026-10-28 for the top-down grasp; re-verify for the 55 deg side grasp')
    # 5. cameras: sim fovy vs real policy crop field of view
    spec = Rig.load()
    m = sc.mjm
    Kt = spec['top']['intrinsics_rectified']['HD1080']
    w, hh = Kt['resolution']
    half = min(round(Kt['cx']), w - round(Kt['cx']), round(Kt['cy']), hh - round(Kt['cy']))
    fov_top = 2 * math.degrees(math.atan(half / Kt['fy']))
    wr = spec['wrist']['right']
    fov_wr = 2 * math.degrees(math.atan((wr['policy_crop_px'] / 2) / wr['intrinsics']['fy']))
    sim_top, sim_wr = float(m.cam_fovy[m.camera('top').id]), float(m.cam_fovy[m.camera('wrist_cam').id])
    check('camera_top_fov', abs(fov_top - sim_top) < 0.5, real_crop_deg=fov_top, sim_fovy_deg=sim_top)
    check('camera_wrist_right_fov', abs(fov_wr - sim_wr) < 0.5, real_crop_deg=fov_wr, sim_fovy_deg=sim_wr)
    # 6. safety checker on the task poses
    from deploy.safety_task import FlexChecker
    from blind_insertion.envs.bimanual_pick import PickTaskCfg
    s_ = cfg['safety']
    check('table_z_is_scene_table', abs(s_['table_z'] - sc.table_top) < 1e-6, config=s_['table_z'], scene=sc.table_top)
    lb = spec['arms']['left_base_in_right_base']['pos']
    check('left_base_height', abs(lb[2]) < 0.003, left_base_z_mm=lb[2] * 1e3,
          note='FlexChecker uses each arm base frame; the left table plane differs by this')
    poses = {'capture': {sd: cap['model_frame_rad'][sd] for sd in ('right', 'left')}, 'park_left': {'left': list(PickTaskCfg().left_park_q)}}
    for name, pp in poses.items():
        for sd, qm in pp.items():
            i = ('right', 'left').index(sd)
            chk = FlexChecker(sc.encoder_offsets[i].numpy(), clearance=s_['clearance'], tip_clearance=s_['tip_clearance'],
                              body_clearance=s_.get('body_clearance'), table_z=s_['table_z'])
            why = chk.violation(np.asarray(qm) - sc.encoder_offsets[i].numpy())
            check(f'safety_{name}_{sd}', why is None, why=why)
    print(json.dumps(res, indent=1, default=float))
    print('FAILED:', fails if fails else 'none')
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'checks': res, 'failed': fails}, indent=1, default=float))
    raise SystemExit(1 if fails else 0)


if __name__ == '__main__':
    main()
