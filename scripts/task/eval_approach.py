"""Closed-loop approach to the pre-insert pose (policy Part 2) in the GPU runtime, RGB estimators only.

Per episode, all worlds in parallel, full DR (layout, arm dynamics, encoder-offset error +/-1.5 deg on
joints 2-5, 0-1 step latency, in-hand pose +/-3 mm / +/-8 deg about the closing axis (confirmed 60-deg
grasp), policy/v1 visual DR incl. camera pose/intrinsics error at real scale):
  1. ZED frame with the arms at the reset pose -> PC port pose estimate (x, y, yaw)   [--pc zed|true]
  2. left press hold planned from that estimate, guarded engage (physics)
  3. right arm + reader placed where the real arm would be after an IK move to the look pose
     (reader `look_gap` in front of the estimated port, nominal in-hand pose, believed joint frame;
     the true joints carry the encoder-offset error, the reader carries the true in-hand pose)
  4. servo (--servo wrist|oracle|none): each iteration renders the right wrist frame, estimates the
     plug tip pose relative to port 2, and moves the commanded hand pose by the rigid correction that
     would bring the reader to the next waypoint (gap schedule ending at --stop mm) -- integrating in
     command space, which also takes out the joint-friction deadband and gravity sag;
  5. the plug-tip pose relative to the live port is recorded (privileged) at the end.

    PYTHONPATH=. .venv/bin/python scripts/task/eval_approach.py --nworld 512 --stop 6 --servo wrist
Writes results/usbc/policy/approach/<name>.json (+ .npz per episode).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'perception'))
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
from blind_insertion.envs.inhand import site_in_reader
from blind_insertion.envs.left_hold import Harness, HarnessCfg, HoldCfg, InsertCfg, LeftHold, RightInserter
from blind_insertion.perception.nets import PoseNet, TwoViewPoseNet
from train_approach_estimators import NOMINAL, WRIST_CROP, WRIST_SIZE, zed_crop

DT = .02
# Coulomb feed-forward for the right arm (envs/friction_comp.py; deployable through i2rt's torque field):
# the identified right-arm Coulomb (dynamics_id/profile.json), fading out within 5 mrad of the target.
FRIC_COMP = {'tau': [0.0275, 0.546, 0.819, 0.0463, 0.0433, 0.0467], 'ramp': .005}


def load(which, dev):
    ck = torch.load(f'runs/perception/{which}.pt', map_location=dev)
    load.mode[which] = ck.get('wrist_mode', 'resize')
    net = (TwoViewPoseNet if ck['which'] == 'wrist2' else PoseNet)(len(ck['scale']), ck['size']).to(dev)
    net.load_state_dict(ck['net']); net.eval()
    return net, ck['scale'].to(dev)


load.mode = {}


def rel_pose(h):
    """True plug-tip pose in the live port frame: position (m), rotation vector (rad)."""
    pp, pR = h.pc_pose()
    tip, rR = h.reader_pose()
    return (pR.transpose(1, 2) @ (tip - pp)[..., None])[..., 0], K.rot_log(pR.transpose(1, 2) @ rR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--pc', choices=('zed', 'true'), default='zed')
    ap.add_argument('--servo', choices=('wrist', 'oracle', 'none'), default='wrist')
    ap.add_argument('--stop', type=float, default=6., help='final gap in front of the panel (mm)')
    ap.add_argument('--look-gap', type=float, default=30.)
    ap.add_argument('--iters', type=int, default=6, help='approach waypoints from the look gap to the stop gap')
    ap.add_argument('--final-rot', choices=('none', 'roll', 'all'), default='none',
                    help='angle correction at the stop gap (the last --final-pos-only iterations are position only)')
    ap.add_argument('--final-pos-only', type=int, default=2)
    ap.add_argument('--tilt', type=float, default=60., help='grasp tilt from vertical (deg)')
    ap.add_argument('--inhand-pos', type=float, default=3., help='+/- in-hand position spread (mm)')
    ap.add_argument('--inhand-rot', type=float, default=8., help='+/- in-hand rotation about the closing axis (deg)')
    ap.add_argument('--hold-force', type=float, default=25.)
    ap.add_argument('--pad-mu', type=float, default=1.2, help='right pad friction (Part 1); <= 0 keeps the scene 1.68')
    ap.add_argument('--impratio', type=float, default=30., help='task impratio (Part 1); <= 0 keeps the scene 100')
    ap.add_argument('--inhand', choices=('measured', 'uniform'), default='measured', help='in-hand pose: Part 1 measured spread or uniform +/- --inhand-pos/--inhand-rot')
    ap.add_argument('--save-states', default=None, help='npz path: full qpos at every wrist estimate (DAgger data)')
    ap.add_argument('--final-iters', type=int, default=6, help='extra servo iterations at the stop gap')
    ap.add_argument('--settle-s', type=float, default=.3)
    ap.add_argument('--ramp-s', type=float, default=.5)
    ap.add_argument('--grasp-settle-s', type=float, default=1.)
    ap.add_argument('--gain', type=float, default=1.)
    ap.add_argument('--frames', type=int, default=1, help='wrist frames averaged per estimate')
    ap.add_argument('--fine-below-mm', type=float, default=0., help='use the fine wrist estimator at waypoints <= this gap (0: never)')
    ap.add_argument('--coarse-net', default='wrist', help='wrist estimator used above --fine-below-mm (runs/perception/<name>.pt)')
    ap.add_argument('--fine-net', default='wrist_fine', help='wrist_fine (right wrist) or wrist2_fine (right + left wrist)')
    ap.add_argument('--fric-comp', action='store_true', help='right-arm Coulomb feed-forward (friction_comp.py)')
    ap.add_argument('--enc-offset-deg', type=float, default=1.5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--name', default=None)
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)
    rt = BimanualRuntime(BimanualCfg(nworld=a.nworld, device='cuda:0', render_rgb=True, visual_dr=True,
                                     camera_resolution=320, measured_camera_cadence=False,
                                     right_fric_comp=FRIC_COMP if a.fric_comp else None))
    rt.visual.gen.manual_seed(a.seed) if hasattr(rt.visual, 'gen') else None
    h = Harness(rt, HarnessCfg(enc_offset_deg=a.enc_offset_deg, pad_mu=a.pad_mu if a.pad_mu > 0 else None,
                               impratio=a.impratio if a.impratio > 0 else None))
    n, dev = h.n, h.dev
    h.reset()
    p_true, R_true = h.pc_pose()
    yaw_true = torch.atan2(-R_true[:, 1, 2], -R_true[:, 0, 2])
    # 1. ZED estimate (arms at reset)
    if a.pc == 'zed':
        zed, zs = load('zed', dev)
        with wp.ScopedDevice(rt.cfg.device):
            mjw.forward(rt.m, rt.d)
        with torch.no_grad():
            est = zed(zed_crop(rt.render()['top'])) * zs
        ex = torch.tensor(NOMINAL['port_xy'], device=dev) + est[:, :2] * 1e-3
        eyaw = torch.deg2rad(est[:, 2])
    else:
        ex, eyaw = p_true[:, :2].clone(), yaw_true.clone()
    zed_err = torch.cat([(ex - p_true[:, :2]) * 1e3, torch.rad2deg(torch.atan2(torch.sin(eyaw - yaw_true), torch.cos(eyaw - yaw_true)))[:, None]], 1)
    # estimated port frame: same construction as the true one (port z = -(cos, sin, 0) of the insertion yaw)
    dyaw = eyaw - yaw_true
    Rz = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=dev).expand(n, 3), dyaw)
    eR = Rz @ R_true                       # yaw error only (the PC sits flat on the table)
    ep = torch.cat([ex, p_true[:, 2:]], 1)  # height known from the table and the case
    # 2. left hold from the estimate
    hold = LeftHold(h, HoldCfg(mode='press', force_n=a.hold_force, across_m=-.0208, along_m=-.02))
    plan = hold.solve(ep, eR)
    ql0, qr0 = h.believed('left').clone(), h.believed('right').clone()
    for k in range(int(round(hold.duration / DT))):
        h.step(qr0, 1., hold.target(k * DT, ql0), -1.)
    q_hold = hold.hold_target()
    # 3. reader in hand, arm where an IK move to the look pose (believed) would put it
    ins = RightInserter(h, InsertCfg(tilt_deg=a.tilt, inhand_pos_mm=a.inhand_pos, inhand_rot_deg=a.inhand_rot, inhand=a.inhand))
    if a.inhand == 'measured':                                 # believed in-hand pose: the measured mean
        from blind_insertion.envs.inhand import measured_mean
        R_sr0, p_sr0 = measured_mean(device=dev)
    else:
        R_sr0, p_sr0 = site_in_reader(ins.cfg.tilt_deg, ins.cfg.pad_z_m, device=dev)   # believed (nominal) in-hand
    ins.place()                                                # samples the true in-hand pose, solves a seed
    look = torch.tensor([0., 0., a.look_gap * 1e-3], device=dev)
    site_p = ep + (eR @ look[:, None].expand(n, 3, 1))[..., 0] + (eR @ p_sr0[:, None].expand(n, 3, 1))[..., 0]
    site_R = eR @ R_sr0
    ch = h.chain['right']
    qb, pe, re = ch.ik(site_p, site_R, ins.target.clone(), iters=150)        # believed joint target
    print('look IK pos err mm p50/max', float(pe.median() * 1e3), float(pe.max() * 1e3), 'place ik ok', float(ins.ik_ok.float().mean()), flush=True)
    q_phys = qb + h.enc_error[:, 0]
    qa = rt.indices['right'][0]
    rt.q[:, qa] = q_phys
    sp, sR = ch.fk(q_phys)                                      # true site pose
    reader_R = sR @ ins.R_sr.transpose(1, 2)
    rt.q[:, rt.lay.peg_qadr:rt.lay.peg_qadr + 3] = sp - (reader_R @ ins.p_sr[..., None])[..., 0]
    rt.q[:, rt.lay.peg_qadr + 3:rt.lay.peg_qadr + 7] = K.mat_to_quat(reader_R)
    rt.v[:] = 0
    h.fresh[:] = True                  # teleported: do not replay the latency-delayed pre-teleport command
    with wp.ScopedDevice(rt.cfg.device):
        mjw.forward(rt.m, rt.d)
    cmd = qb.clone()
    if a.debug:
        sp_, sR_ = ch.fk(h.q('right')); tip_, rR_ = h.reader_pose()
        print('placed: tip-from-site err mm', float(((sR_.transpose(1, 2) @ (tip_ - sp_)[..., None])[..., 0] + (ins.R_sr.transpose(1, 2) @ ins.p_sr[..., None])[..., 0]).norm(dim=1).max() * 1e3), flush=True)
        c_ = h.contacts(); print('contacts right-pc', int(c_['right_pc'].sum()), 'arm-arm', int(c_['arm_arm'].sum()), flush=True)
    for k in range(int(a.grasp_settle_s / DT)):
        if a.debug and k in (0, 1, 5, 20, 49):
            tp_, _ = rel_pose(h); print('settle step', k, 'lateral p50 mm', float(tp_[:, :2].norm(dim=1).median() * 1e3), 'gap p50', float(tp_[:, 2].median() * 1e3), flush=True)
        h.step(cmd, -1., q_hold, -1.)
    p_look, rv_look = rel_pose(h)
    # 4. servo
    wrist = load(a.coarse_net, dev) if a.servo == 'wrist' else None
    fine = load(a.fine_net, dev) if a.servo == 'wrist' and a.fine_below_mm > 0 else None
    two = a.fine_net.startswith('wrist2')
    stop = a.stop * 1e-3
    sched = np.geomspace(a.look_gap * 1e-3, stop, max(a.iters - 1, 1)).tolist() + [stop] * a.final_iters if a.servo != 'none' else []
    trace, per_world, states = [], [], []
    if a.servo == 'none':                                        # open loop: IK straight to the stop pose
        site_p2 = ep + (eR @ torch.tensor([0., 0., stop], device=dev)[:, None].expand(n, 3, 1))[..., 0] + (eR @ p_sr0[:, None].expand(n, 3, 1))[..., 0]
        cmd, _, _ = ch.ik(site_p2, site_R, cmd, iters=100)
        for k in range(int(1.5 / DT)):
            h.step(cmd, -1., q_hold, -1.)
    for it, g in enumerate(sched):
        if a.servo == 'wrist':
            ests = []
            for f in range(a.frames):
                fr = rt.render()
                rs = lambda x: F.interpolate(x.permute(0, 3, 1, 2).float(), size=(WRIST_SIZE, WRIST_SIZE), mode='area').round().byte().permute(0, 2, 3, 1)
                net = fine if (fine is not None and g * 1e3 <= a.fine_below_mm) else wrist
                name = a.fine_net if net is fine else a.coarse_net
                if load.mode.get(name) == 'crop':
                    img = fr['wrist_cam'][:, WRIST_CROP[1]:WRIST_CROP[1] + WRIST_SIZE, WRIST_CROP[0]:WRIST_CROP[0] + WRIST_SIZE].contiguous()
                else:
                    img = rs(fr['wrist_cam'])
                if net is fine and two:
                    img = torch.stack([img, rs(fr['left_wrist_cam'])], 1)
                with torch.no_grad():
                    ests.append(net[0](img) * net[1])
                if f + 1 < a.frames:
                    h.step(cmd, -1., q_hold, -1.)
            e = torch.stack(ests).mean(0)
            rel_p_est, rel_rv_est = e[:, :3] * 1e-3, torch.deg2rad(e[:, 3:])
            if a.save_states:
                states.append(rt.q.clone())
        else:
            rel_p_est, rel_rv_est = rel_pose(h)
        # rigid correction of the reader, expressed in the (estimated) port frame: current -> waypoint
        cur_R = K.axis_angle_mat(rel_rv_est / rel_rv_est.norm(dim=1, keepdim=True).clamp(min=1e-9), rel_rv_est.norm(dim=1))
        tgt_p = torch.tensor([0., 0., g], device=dev).expand(n, 3)
        rv = -rel_rv_est * a.gain                                # rotate the reader back onto the axis (port frame)
        final = it >= len(sched) - a.final_iters
        if final and (a.final_rot == 'none' or it >= len(sched) - a.final_pos_only):
            rv = rv * 0
        elif final and a.final_rot == 'roll':
            rv = rv * torch.tensor([0., 0., 1.], device=dev)
        dR_port = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
        # site (hand) moves rigidly with the reader: new = T_world(dR about the tip, then translate)
        qb_now = cmd
        sp_b, sR_b = ch.fk(qb_now)                               # commanded hand pose (believed model)
        tip_b = sp_b - (sR_b @ (R_sr0.transpose(0, 1) @ p_sr0)[:, None])[..., 0]   # believed tip (nominal in-hand)
        dR_w = eR @ dR_port @ eR.transpose(1, 2)
        dt_w = (eR @ ((tgt_p - rel_p_est) * a.gain)[..., None])[..., 0]
        new_R = dR_w @ sR_b
        new_p = tip_b + dt_w + (dR_w @ (sp_b - tip_b)[..., None])[..., 0]
        prev = cmd
        cmd, ik_pe, ik_re = ch.ik(new_p, new_R, cmd, iters=60)
        nr = int(a.ramp_s / DT)
        for k in range(nr):                                      # smooth move: no jerk on the grasp
            u = (k + 1) / nr; u = u * u * (3 - 2 * u)
            h.step(prev + (cmd - prev) * u, -1., q_hold, -1.)
        for k in range(int(a.settle_s / DT)):
            h.step(cmd, -1., q_hold, -1.)
        tp, trv = rel_pose(h)
        fin = torch.isfinite(tp).all(1)
        latn = tp[fin, :2].norm(dim=1) * 1e3
        sp_t, sR_t = ch.fk(h.q('right'))
        _, rR = h.reader_pose()
        slip = torch.rad2deg(K.rot_log(sR_t.transpose(1, 2) @ rR @ ins.R_sr).norm(dim=1))   # in-hand rotation change
        vnorm = h.rt.v[:, h.rt.indices['right'][1]].norm(dim=1)
        qerr = (h.believed('right') - cmd).abs().max(1).values
        per_world.append(torch.stack([tp[:, 0], tp[:, 1], tp[:, 2], ik_pe, slip, vnorm, qerr], 1))
        trace.append({'gap_target_mm': g * 1e3, 'lateral_mm_p50': float(latn.median()),
                      'lateral_mm_p99': float(torch.quantile(latn, .99)), 'ik_pos_err_mm_p99': float(torch.quantile(ik_pe[fin], .99) * 1e3),
                      'inhand_rot_change_deg_p99': float(torch.quantile(slip[fin], .99))})
    # 5. final
    tp, trv = rel_pose(h)
    pc_final, _ = h.pc_pose()
    fin = torch.isfinite(rt.q).all(1)
    lat = tp[:, :2].norm(dim=1) * 1e3
    tilt = torch.rad2deg(trv[:, :2].norm(dim=1))
    w = {'x_mm': tp[:, 0] * 1e3, 'y_mm': tp[:, 1] * 1e3, 'gap_mm': tp[:, 2] * 1e3, 'lateral_mm': lat,
         'rx_deg': torch.rad2deg(trv[:, 0]), 'ry_deg': torch.rad2deg(trv[:, 1]), 'roll_deg': torch.rad2deg(trv[:, 2]),
         'tilt_deg': tilt, 'zed_err': zed_err, 'look_lateral_mm': p_look[:, :2].norm(dim=1) * 1e3,
         'inhand': ins.inhand, 'pc_moved_mm': (pc_final - p_true).norm(dim=1) * 1e3, 'finite': fin,
         'left_latched': hold.latched}
    if per_world:
        w['servo_per_world'] = torch.stack(per_world, 1)     # (n, iters, [x, y, gap, ik_err, inhand_rot_change])
    w = {k: v.detach().cpu().numpy() for k, v in w.items()}
    ok = w['finite'].astype(bool)
    q = lambda x, p: float(np.percentile(x[ok], p))
    pct = (50, 90, 99, 99.5, 100)
    name = a.name or f'approach_pc-{a.pc}_servo-{a.servo}_stop{a.stop:g}'
    rep = {'name': name, 'method': __doc__.strip(), 'n': int(len(ok)), 'n_finite': int(ok.sum()), 'args': vars(a),
           'units': 'mm and deg; plug tip in the port-2 frame (x across the panel, y up, z out of the panel = gap); '
                    'tilt = angle between the plug axis and the port axis; roll about the port axis',
           'lateral_mm': {f'p{p}': q(w['lateral_mm'], p) for p in pct},
           'tilt_deg': {f'p{p}': q(w['tilt_deg'], p) for p in pct},
           'gap_error_mm': {f'p{p}': q(np.abs(w['gap_mm'] - a.stop), p) for p in pct},
           'roll_deg_abs': {f'p{p}': q(np.abs(w['roll_deg']), p) for p in pct},
           'x_mm': {'mean': float(w['x_mm'][ok].mean()), 'std': float(w['x_mm'][ok].std())},
           'y_mm': {'mean': float(w['y_mm'][ok].mean()), 'std': float(w['y_mm'][ok].std())},
           'within_1mm_2deg': float(((w['lateral_mm'] <= 1) & (w['tilt_deg'] <= 2))[ok].mean()),
           'zed_error': {'xy_mm_p50': q(np.linalg.norm(w['zed_err'][:, :2], axis=1), 50),
                         'xy_mm_p99': q(np.linalg.norm(w['zed_err'][:, :2], axis=1), 99),
                         'yaw_deg_p99': q(np.abs(w['zed_err'][:, 2]), 99)},
           'look_pose_lateral_mm': {f'p{p}': q(w['look_lateral_mm'], p) for p in (50, 99)},
           'pc_moved_since_reset_mm': {f'p{p}': q(w['pc_moved_mm'], p) for p in (50, 99, 100)},
           'servo_trace': trace, 'wall_s': time.time() - t0}
    if a.save_states and states:
        np.savez_compressed(a.save_states, qpos=torch.stack(states, 1).cpu().numpy())
    out = Path(f'results/usbc/policy/approach/{name}.json')
    out.write_text(json.dumps(rep, indent=1) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), **w)
    print(json.dumps({k: rep[k] for k in ('name', 'n_finite', 'lateral_mm', 'tilt_deg', 'gap_error_mm', 'within_1mm_2deg', 'zed_error', 'look_pose_lateral_mm', 'servo_trace')}, indent=1))


if __name__ == '__main__':
    main()
