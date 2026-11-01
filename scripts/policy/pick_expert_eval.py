"""Evaluate the Part 1 scripted expert (top-down side grasp) on the GPU bimanual runtime with all DR.

Success per episode: the expert reaches the end of its hold, the reader is lifted >= 60 mm, its
tilt (thickness axis vs world up) is < 3 deg at the start and end of the 1 s hold, and the in-hand
pose moves < 1 mm / 1 deg during the hold. Writes per-episode parquet + summary JSON and the
in-hand pose spread (`results/usbc/policy/pick/inhand_spread.json`, read by reader_grasp.inhand_pose).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/policy/pick_expert_eval.py --episodes 2048 --seed 1000
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import warp as wp
wp.config.quiet = True
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.scenes import reader_grasp as G

ap = argparse.ArgumentParser()
ap.add_argument('--episodes', type=int, default=2048)
ap.add_argument('--nworld', type=int, default=1024)
ap.add_argument('--seed', type=int, default=1000)
ap.add_argument('--out', default='results/usbc/policy/pick/expert')
ap.add_argument('--no_dr', action='store_true', help='nominal physics: no contact/control DR')
ap.add_argument('--write_spread', action='store_true')
ap.add_argument('--impratio', type=float, default=None)
ap.add_argument('--ki', type=float, default=0.03)
ap.add_argument('--reader_wrist', default=None, help='wrist refinement net (scripts/perception/train_reader_wrist.py) applied at the pre-grasp hover')
ap.add_argument('--reader_est', default=None, help='plan with the ZED reader estimator (checkpoint); renders the ZED at reset with visual DR')
ap.add_argument('--pose_err', type=float, nargs=2, default=None, help='planning pose error: xy (m, random direction), yaw (deg, random sign)')
ap.add_argument('--settle_max_s', type=float, default=2.0)
ap.add_argument('--finger_open', type=float, default=-1., help='approach finger opening (m per finger); -1: task default')
ap.add_argument('--lift_v', type=float, default=0.10, help='expert Cartesian speed after closing (m/s)')
ap.add_argument('--hold_s', type=float, default=1.0, help='hold after lift (stability window)')
ap.add_argument('--grip_range', type=float, nargs=2, default=None)
ap.add_argument('--pad_mu_range', type=float, nargs=2, default=None)
ap.add_argument('--layout', default='results/usbc/policy/pick/task_layout.json')
ap.add_argument('--enc_offset_deg', type=float, default=None, help='encoder-offset DR half range (deg, joints 2-5); default the task config (1.5)')
ap.add_argument('--servo', default=None, help='wrist servo net (scripts/perception/train_reader_servo.py): grasp pose relative to the hand at the hover and at --servo_fracs of the descent')
ap.add_argument('--servo_pos_only', action='store_true')
ap.add_argument('--rehover_mm', type=float, default=0., help='re-hover and take another hover frame when the hover correction moves the grasp by more than this (0: off)')
ap.add_argument('--rehover_max', type=int, default=2)
ap.add_argument('--dump_servo', default=None, help='save the servo frames (224 px) with privileged labels (true grasp relative to the true hand) to this .pt (servo DAgger)')
ap.add_argument('--dump_size', type=int, default=224)
ap.add_argument('--servo_near_rot', default=None, help='6-DoF servo net used only for the rotation of the waypoint correction (position from --servo_near)')
ap.add_argument('--servo_near', default=None, help='second servo net used for the waypoint frames (the hover frame keeps --servo)')
ap.add_argument('--servo_fracs', type=float, nargs='*', default=[.5, .8])
ap.add_argument('--believed_frame', action='store_true', help='expert plans/servos on encoder FK (encoder-offset errors hit the grasp, as on the real robot)')
ap.add_argument('--wrist_rebase', choices=('true', 'encoder'), default='true', help='hand pose the wrist estimate is rebased on: true FK (privileged) or encoder FK (as on the real robot: encoder offset DR errors included)')
a = ap.parse_args()


def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


cfg = PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout=a.layout), episode_s=(18. if a.reader_wrist else 21. if a.servo else 12.) + a.hold_s)
if a.enc_offset_deg is not None:
    cfg.enc_offset_deg = a.enc_offset_deg
if a.reader_est:
    cfg.physics.render_rgb = True; cfg.physics.visual_dr = True; cfg.physics.measured_camera_cadence = False
    cfg.physics.camera_resolution = torch.load(a.reader_est, map_location='cpu')['res']
if a.impratio is not None: cfg.impratio = a.impratio
if a.finger_open >= 0: cfg.finger_open = a.finger_open or None
if a.grip_range: cfg.grip_force_scale_range = tuple(a.grip_range)
if a.pad_mu_range: cfg.pad_friction_range = tuple(a.pad_mu_range)
if a.no_dr:
    cfg.pad_friction_range = cfg.reader_table_friction_range = cfg.grip_force_scale_range = None
    cfg.latency_steps = (0,); cfg.enc_offset_deg = 0.
    cfg.physics.friction_scale_range = (1., 1.); cfg.physics.gravity_scale_jitter = 0.
t0 = time.time()
task = PickTask(cfg)
print('built', round(time.time() - t0, 1), 's', flush=True)
torch.manual_seed(a.seed)
rows = []
for batch in range(math.ceil(a.episodes / a.nworld)):
    if a.reader_est: task.cfg.physics.render_rgb = True
    task.reset()
    est_fn, est_err = None, None
    if a.reader_est:
        from blind_insertion.perception.reader_zed import ReaderZedEstimator, reader_pose
        if batch == 0: EST = ReaderZedEstimator(a.reader_est)
        # reader capture pose: right arm at rest (= ZED capture pose), grippers fully open, left arm parked
        rt_ = task.runtime; fq_ = task.finger_qadr
        f_keep = rt_.q[:, fq_].clone(); rt_.q[:, fq_] = .0475
        import mujoco_warp as _mjw
        with wp.ScopedDevice(cfg.physics.device): _mjw.kinematics(rt_.m, rt_.d)
        xy_hat, yaw_hat = EST(task.runtime.render()['top'])
        rt_.q[:, fq_] = f_keep
        with wp.ScopedDevice(cfg.physics.device): _mjw.forward(rt_.m, rt_.d)
        task.cfg.physics.render_rgb = False      # one ZED frame at the rest pose; no rendering while stepping
        p_hat, R_hat = reader_pose(task, xy_hat, yaw_hat)
        est_fn = lambda: (p_hat, R_hat)   # noqa: E731
        p_t, R_t = task.reader_pose(); bz = float(task.lay.variants[0].peg_boxes[0][0][2])
        est_err = ((p_hat + R_hat[:, :, 2] * bz)[:, :2] - (p_t + R_t[:, :, 2] * bz)[:, :2]).norm(dim=1) * 1e3
        est_yaw_err = torch.rad2deg(torch.atan2(-R_hat[:, 1, 2], -R_hat[:, 0, 2]) - torch.atan2(-R_t[:, 1, 2], -R_t[:, 0, 2])).abs()
    ex = PickExpert(task, seed=a.seed + batch, hold_s=a.hold_s, v_max=a.lift_v, integral_gain=a.ki, settle_max_s=a.settle_max_s,
                    pose_err=a.pose_err, reader_pose_fn=est_fn, hover_tol_m=.004 if (a.reader_wrist or a.servo) else None)
    ex.believed_frame = a.believed_frame
    if a.servo: ex.servo_fracs = tuple(a.servo_fracs); ex.servo_rot = not a.servo_pos_only
    ex.reset()
    N = a.nworld
    start_hold = {k: torch.zeros(N, *s, device=task.device) for k, s in (('p', (3,)), ('R', (3, 3)), ('tilt', ()))}
    steps = torch.zeros(N, dtype=torch.long, device=task.device)
    final = None
    if a.reader_wrist:
        import train_reader_wrist as RW
        from blind_insertion.perception.nets import PoseNet as _PN
        if batch == 0:
            _ck = torch.load(a.reader_wrist, map_location=task.device)
            WNET = _PN(3, _ck['size'], width=_ck['width']).to(task.device).eval(); WNET.load_state_dict(_ck['net']); WSC = _ck['scale'].to(task.device)
        wcorr = torch.zeros(N, 3, device=task.device)
    if a.servo:
        from blind_insertion.perception.nets import PoseNet as _PN
        if batch == 0 and a.servo != 'oracle':
            _ck = torch.load(a.servo, map_location=task.device)
            SNET = _PN(_ck.get('n_out', 6), _ck['size'], width=_ck['width']).to(task.device).eval(); SNET.load_state_dict(_ck['net']); SSC = _ck['scale'].to(task.device); SSIZE = _ck['size']
            NNET = None
            DUMP = {'X': [], 'Y': [], 'k': []}
            if a.servo_near:
                _cn = torch.load(a.servo_near, map_location=task.device)
                NNET = _PN(_cn.get('n_out', 6), _cn['size'], width=_cn['width']).to(task.device).eval(); NNET.load_state_dict(_cn['net']); NSC = _cn['scale'].to(task.device)
            RNET = None
            if a.servo_near_rot:
                _cr = torch.load(a.servo_near_rot, map_location=task.device)
                RNET = _PN(_cr.get('n_out', 6), _cr['size'], width=_cr['width']).to(task.device).eval(); RNET.load_state_dict(_cr['net']); RSC = _cr['scale'].to(task.device)
        n_frames = torch.zeros(N, dtype=torch.long, device=task.device)
    g_err = torch.full((N, 2), float('nan'), device=task.device)
    for k in range(task.max_steps - 1):
        prev = ex.phase.clone()
        act = ex.act()
        closing = (ex.phase == 3) & (prev == 2)
        if closing.any():   # true hand vs true grasp pose at the close (diagnostic; privileged)
            _W = G.grasp_poses(*task.reader_pose(), ex.offset, ex.tilt.nan_to_num(G.TILT_DEG), flip=ex.flip)
            _p, _R = task.tcp_pose()
            g_err[closing, 0] = ((_W['grasp'][0] - _p).norm(dim=1) * 1e3)[closing]
            g_err[closing, 1] = torch.rad2deg(K.rot_log(_W['grasp'][1] @ _R.transpose(1, 2)).norm(dim=1))[closing]
        if a.servo:
            want = ((ex.phase == 1) & (prev == 0) & ~ex.refined) | ex.need_frame
            if want.any():
                if a.servo == 'oracle':      # privileged: true grasp relative to the true hand (upper bound)
                    _W = G.grasp_poses(*task.reader_pose(), ex.offset, ex.tilt.nan_to_num(G.TILT_DEG), flip=ex.flip)
                    _p, _R = task.tcp_pose()
                    delta = torch.cat([(_R.transpose(1, 2) @ (_W['grasp'][0] - _p)[..., None])[..., 0], K.rot_log(_R.transpose(1, 2) @ _W['grasp'][1])], 1)
                    ex.servo_update(want, delta); n_frames += want.long()
                    want = want & False
            if want.any():
                task.cfg.physics.render_rgb = True
                with torch.no_grad():
                    fr = task.runtime.render()['wrist_cam']
                    def _run(net, sc):
                        d_ = torch.cat([net(fr[i:i + 256]) for i in range(0, N, 256)]) * sc
                        return torch.cat([d_, torch.full_like(d_, float('nan'))], 1) if d_.shape[1] == 3 else d_
                    delta = _run(SNET, SSC)
                    if NNET is not None and ex.need_frame.any():
                        dn = _run(NNET, NSC)
                        if RNET is not None:
                            dn = torch.cat([dn[:, :3], _run(RNET, RSC)[:, 3:]], 1)
                        delta = torch.where(ex.need_frame[:, None], dn, delta)
                task.cfg.physics.render_rgb = False
                if a.dump_servo:
                    _W = G.grasp_poses(*task.reader_pose(), ex.offset, ex.tilt.nan_to_num(G.TILT_DEG), flip=ex.flip)
                    _p, _R = task.tcp_pose()
                    lab = torch.cat([(_R.transpose(1, 2) @ (_W['grasp'][0] - _p)[..., None])[..., 0], K.rot_log(_R.transpose(1, 2) @ _W['grasp'][1])], 1)
                    small = torch.nn.functional.interpolate(fr[want].permute(0, 3, 1, 2).float(), size=(a.dump_size, a.dump_size), mode='area').round().clamp(0, 255).byte().permute(0, 2, 3, 1)
                    DUMP['X'].append(small.cpu()); DUMP['Y'].append(lab[want].cpu()); DUMP['k'].append(torch.where(ex.need_frame, ex.servo_k, -1)[want].cpu())
                hov = want & ~ex.need_frame; pg_old = ex.p_g.clone()
                ex.servo_update(want, delta)
                n_frames += want.long()
                if a.rehover_mm > 0 and hov.any():
                    big = hov & ((ex.p_g - pg_old).norm(dim=1) > a.rehover_mm * 1e-3) & (ex.hover_iters < a.rehover_max)
                    if big.any(): ex.rehover(big)
        if a.reader_wrist:
            at_hover = (ex.phase == 1) & (prev == 0) & ~ex.refined
            if at_hover.any():
                task.cfg.physics.render_rgb = True
                with torch.no_grad():
                    fr = task.runtime.render()['wrist_cam']
                    corr = torch.cat([WNET(fr[i:i + 256]) for i in range(0, N, 256)]) * WSC
                task.cfg.physics.render_rgb = False
                p_m, R_m = RW.measured_tcp(task); cam = RW.wrist_cam_pos(task)
                if a.wrist_rebase == 'encoder':      # encoder-FK hand pose; the camera rides on it rigidly
                    cam_rel = (R_m.transpose(1, 2) @ (cam - p_m)[..., None])[..., 0]
                    p_m, R_m = task.chain.fk(task._measured_model_angles()[:, 0])
                    cam = p_m + (R_m @ cam_rel[..., None])[..., 0]
                p_eb, R_eb = RW.rebase_estimate(task, ex.p_est, ex.R_est, ex.p_pre, ex.R_g, p_m, R_m, cam)
                p_new, R_new = RW.apply_correction(task, p_eb, R_eb, corr)
                ex.replan(at_hover, p_new, R_new)
                wcorr[at_hover] = corr[at_hover]
        task.step(act)
        m = task.pick_metrics()
        entered = (ex.phase == 5) & (prev == 4)
        start_hold['p'][entered] = m['inhand_p'][entered]; start_hold['R'][entered] = m['inhand_R'][entered]
        start_hold['tilt'][entered] = m['tilt_deg'][entered]
        fin = (ex.phase == 6) & (prev == 5)
        if final is None: final = {kk: v.clone() for kk, v in m.items()}
        for kk, v in m.items(): final[kk][fin] = v[fin]
        steps[fin] = k + 1
        if ex.done.all(): break
    m_now = task.pick_metrics()
    for kk, v in m_now.items(): final[kk][~ex.done] = v[~ex.done]
    dp = (final['inhand_p'] - start_hold['p']).norm(dim=1)
    dR = torch.rad2deg(K.rot_log(final['inhand_R'] @ start_hold['R'].transpose(1, 2)).norm(dim=1))
    nominal_p = torch.zeros(N, 3, device=task.device); nominal_R = torch.zeros(N, 3, 3, device=task.device)
    for (tl, fl), off in task.offsets.items():
        np_, nR = G.nominal_inhand(off, tl, fl); sel = (ex.tilt == tl) & (ex.flip == fl)
        nominal_p[sel] = torch.tensor(np_, dtype=torch.float32, device=task.device); nominal_R[sel] = torch.tensor(nR, dtype=torch.float32, device=task.device)
    nominal_p[~ex.ik_ok] = nominal_p[ex.ik_ok][:1] if ex.ik_ok.any() else 0; nominal_R[~ex.ik_ok] = torch.eye(3, device=task.device)
    rel_p = final['inhand_p'] - nominal_p
    rel_r = K.rot_log(final['inhand_R'] @ nominal_R.transpose(1, 2))
    ok = ex.done & ~task.failed & final['finite'] & (final['lift_m'] > .06) & (final['tilt_deg'] < 3) & (start_hold['tilt'] < 3) & (dp < 1e-3) & (dR < 1.)
    pr = task.reader0
    d = {'batch': batch, 'success': ok, 'done': ex.done, 'ik_ok': ex.ik_ok, 'tilt_cmd_deg': ex.tilt, 'flip': ex.flip, 'failed': task.failed, 'nonfinite': task.nonfinite, 'nonfinite_part': task.nonfinite_part, 'nonfinite_step': task.nonfinite_step, 'phase_end': ex.phase, 'lift_mm': final['lift_m'] * 1e3,
         'tilt_deg': final['tilt_deg'], 'tilt_start_hold_deg': start_hold['tilt'], 'hold_drift_mm': dp * 1e3, 'hold_rot_drift_deg': dR,
         'settle_err_mm': ex.settle_err[:, 0] * 1e3, 'settle_err_deg': ex.settle_err[:, 1], 'steps': steps,
         'reader_x': pr[:, 0], 'reader_y': pr[:, 1],
         'reader_yaw_deg': torch.rad2deg(torch.atan2(-K.quat_to_mat(pr[:, 3:])[:, 1, 2], -K.quat_to_mat(pr[:, 3:])[:, 0, 2])),
         'latency': task.latency, 'grip_scale': task.runtime.grip_scale if task.runtime.grip_scale is not None else torch.ones(N, device=task.device),
         'wrist_corr_mm': torch.linalg.norm(wcorr[:, :2], dim=1) * 1e3 if a.reader_wrist else torch.zeros(N, device=task.device),
         'final_est_xy_err_mm': ((ex.p_est + ex.R_est[:, :, 2] * float(task.lay.variants[0].peg_boxes[0][0][2]))[:, :2] - (task.reader0[:, :3] + K.quat_to_mat(task.reader0[:, 3:])[:, :, 2] * float(task.lay.variants[0].peg_boxes[0][0][2]))[:, :2]).norm(dim=1) * 1e3,
         'est_xy_err_mm': est_err if est_err is not None else torch.zeros(N, device=task.device),
         'est_yaw_err_deg': est_yaw_err if est_err is not None else torch.zeros(N, device=task.device),
         'grasp_err_mm': g_err[:, 0], 'grasp_err_deg': g_err[:, 1],
         'servo_frames': n_frames if a.servo else torch.zeros(N, device=task.device),
         'pad_mu': task.runtime.geom_friction[:, task.pads[0], 0], 'reader_mu': task.runtime.geom_friction[:, task.reader_geoms[0], 0]}
    for i, nm in enumerate('xyz'): d[f'inhand_d{nm}_mm'] = rel_p[:, i] * 1e3
    for i, nm in enumerate('xyz'): d[f'inhand_r{nm}_deg'] = torch.rad2deg(rel_r[:, i])
    d['inhand_p'] = final['inhand_p']; d['inhand_R'] = final['inhand_R']
    rows.append({kk: v.detach().cpu().numpy() if torch.is_tensor(v) else np.full(N, v) for kk, v in d.items()})
    print(f'batch {batch}: success {int(ok.sum())}/{N} done {int(ex.done.sum())} t={time.time()-t0:.0f}s', flush=True)

cat = {k: np.concatenate([r[k] for r in rows])[:a.episodes] for k in rows[0]}
P, R = cat.pop('inhand_p'), cat.pop('inhand_R')
import pandas as pd
df = pd.DataFrame(cat)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
tag = out.name + ('_nodr' if a.no_dr else '') + (('_servo_oracle' if a.servo == 'oracle' else '_servo') + ('pos' if a.servo_pos_only else '') if a.servo else '') + ('_encrebase' if a.reader_wrist and a.wrist_rebase == 'encoder' else '') + ('_believed' if a.believed_frame else '') + (f'_enc{a.enc_offset_deg:g}' if a.enc_offset_deg is not None else '') + f'_s{a.seed}'
df.to_parquet(out.parent / f'{tag}.parquet')
n, k = len(df), int(df.success.sum())
summ = {'n': n, 'success': k, 'rate': k / n, 'wilson95_lo': wilson_lo(k, n), 'seed': a.seed, 'dr': not a.no_dr,
        'failures': {'nonfinite': int(df.nonfinite.sum()), 'not_done': int((~df.done).sum()), 'ik_fail': int((~df.ik_ok).sum()), 'lift<60mm': int((df.lift_mm <= 60).sum()),
                     'tilt>=3': int((df.tilt_deg >= 3).sum()), 'hold_drift>=1mm': int((df.hold_drift_mm >= 1).sum())},
        'tilt_deg_p50_p99_max': [float(np.percentile(df.tilt_deg, q)) for q in (50, 99, 100)],
        'settle_err_mm_p50_p99': [float(np.percentile(df.settle_err_mm, q)) for q in (50, 99)],
        'episode_s_p50_p99': [float(np.percentile(df.steps[df.done], q) * task.dt) for q in (50, 99)] if df.done.any() else None}
okm = df.success.values
if a.reader_est:
    summ['estimator'] = {'ckpt': a.reader_est, **{f'{k}_p{q}': float(np.percentile(df[k], q)) for k in ('est_xy_err_mm', 'est_yaw_err_deg') for q in (50, 99, 99.5, 100)}}
sp = {'n_success': int(okm.sum())}
for c in [c for c in df.columns if c.startswith('inhand_')]:
    v = df[c].values[okm]
    sp[c] = {'mean': float(v.mean()), 'std': float(v.std()), 'p1': float(np.percentile(v, 1)), 'p99': float(np.percentile(v, 99))}
summ['inhand_spread'] = sp
if okm.any():
    dpos = np.sqrt(sum(df[f'inhand_d{c}_mm'].values[okm] ** 2 for c in 'xyz'))
    drot = np.sqrt(sum(df[f'inhand_r{c}_deg'].values[okm] ** 2 for c in 'xyz'))
    pc = lambda v: {'p50': float(np.percentile(v, 50)), 'p99': float(np.percentile(v, 99)), 'max': float(v.max())}   # noqa: E731
    summ['inhand_magnitude'] = {'about': 'mean in-hand pose of the successes', 'pos_mm': pc(np.sqrt(sum((df[f'inhand_d{c}_mm'].values[okm] - df[f'inhand_d{c}_mm'].values[okm].mean()) ** 2 for c in 'xyz'))),
                                'rot_deg': pc(np.sqrt(sum((df[f'inhand_r{c}_deg'].values[okm] - df[f'inhand_r{c}_deg'].values[okm].mean()) ** 2 for c in 'xyz'))),
                                'about_nominal_pos_mm': pc(dpos), 'about_nominal_rot_deg': pc(drot),
                                'hold_drift_mm': pc(df.hold_drift_mm.values[okm]), 'hold_rot_drift_deg': pc(df.hold_rot_drift_deg.values[okm]), 'hold_s': a.hold_s}
print(json.dumps(summ, indent=1))
(out.parent / f'{tag}.json').write_text(json.dumps(summ, indent=1) + '\n')
if a.dump_servo:
    torch.save({k_: torch.cat(v) for k_, v in DUMP.items()}, a.dump_servo); print('dumped', sum(len(x) for x in DUMP['X']), 'servo frames ->', a.dump_servo)
if a.write_spread:
    Pm = P[okm].mean(0); Rm = R[okm].mean(0); U, _, Vt = np.linalg.svd(Rm); Rm = U @ Vt
    from blind_insertion.scenes.reader_grasp import _mat_to_quat
    spread = {'source': f'{tag}.parquet', 'n': int(okm.sum()), 'mean_pos_m': Pm.tolist(), 'mean_quat_wxyz': _mat_to_quat(Rm).tolist(),
              'nominal': {f'tilt{tl:g}_flip{int(fl)}': {'pos_m': G.nominal_inhand(off, tl, fl)[0].tolist(), 'quat_wxyz': _mat_to_quat(G.nominal_inhand(off, tl, fl)[1]).tolist(), 'pad_contact_height_m': task.pad_contact_height[(tl, fl)]} for (tl, fl), off in task.offsets.items()},
              'grasp_share': {f'tilt{tl:g}_flip{int(fl)}': float(((df.tilt_cmd_deg == tl) & (df.flip == fl))[okm].mean()) for (tl, fl) in task.offsets}, 'about_nominal': sp,
              'units': 'inhand_d* in mm (grasp_site frame), inhand_r* rotation vector in deg (grasp_site frame) about the nominal'}
    G.SPREAD_FILE.write_text(json.dumps(spread, indent=1) + '\n')
    print('wrote', G.SPREAD_FILE)
