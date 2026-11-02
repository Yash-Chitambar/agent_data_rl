"""Batched sim evaluation of the guarded grasp (blind_insertion/pipeline/grasp_v3.py) against the current expert.

Physics: the realistic grasp model (BimanualCfg.grasp_fidelity, docs/real/GRASP_FIDELITY.md, first version, under validation).
Injected, from the real logs (results/real/pick_20261101_analysis.md):
  servo roll     --servo_roll_deg r on every world; default the model's bootstrap of the 11 real target poses
  plan too low   --z_bias_mm a b on the guarded modes: the real servo plan sits 3-13 mm below the height where the hand stops
  press          --press_mm on the modes without touch-down (the model's way to reproduce the same real press)
Held = done, not failed, lifted > 60 mm. Attempt 1 alone = `first`; after retries = `success`.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/policy/grasp_v3_eval.py --mode combined --episodes 512 --out results/real/grasp_v3/oracle
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warp as wp

wp.config.quiet = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.envs import kinematics as K  # noqa: E402
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg  # noqa: E402
from blind_insertion.envs.bimanual_runtime import BimanualCfg  # noqa: E402
from blind_insertion.pipeline.grasp_v3 import G_CLOSE, G_HOLD, G_SQUEEZE, G_TD, G_TEST, GraspV3Cfg, GraspV3Expert  # noqa: E402
from blind_insertion.scenes import reader_grasp as G  # noqa: E402
from deploy.conventions import GripperMap  # noqa: E402

OFF = dict(press_from_attempt=None, level_roll=False, roll_gate_deg=None, touchdown=False, hold_from='press', wrist_kp=None, wrist_kd=None, gentle_close=False, holddown_mm=0., max_attempts=1, td_friction_ff=False)
MODES = {
    's0_level': {**OFF, 'level_roll': True},
    's1_touchdown': {**OFF, 'touchdown': True},
    's2_wrist': {**OFF, 'wrist_kp': (40., 30., 30.), 'wrist_kd': (1.5, 2., 2.)},
    's3_gentle_hold': {**OFF, 'hold_from': 'measured', 'gentle_close': True, 'holddown_mm': 3.},
    's4_retry': {**OFF, 'max_attempts': 3},
    'level_gentle': {**OFF, 'level_roll': True, 'hold_from': 'measured', 'gentle_close': True, 'holddown_mm': 3.},
    's1_touchdown_gentle': {**OFF, 'touchdown': True, 'gentle_close': True, 'holddown_mm': 8.},
    'combined': dict(td_friction_ff=False),                                                      # the deploy default
    'combined_tilt': dict(td_friction_ff=False, tilt_fix=True),
    'combined_wrist': dict(td_friction_ff=False, wrist_kp=(40., 30., 30.), wrist_kd=(1.5, 2., 2.)),
}
RELIEF_V3 = {'lift_mm': -3., 'settle_s': .3, 'close_ramp_s': 1., 'dwell_s': .3, 'blend_s': .5}


def wilson(k, n, z=1.96):
    if not n:
        return (0., 0.)
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


class SimHooks:
    """grasp_v3 hooks on the batched PickTask: the same reading map and position-mode jaws as deploy/bimanual ChainHooks + SimIO."""

    def __init__(self, task, gc, gmap):
        self.t, self.gc, self.gmap = task, gc, gmap
        rt, dev = task.runtime, task.device
        self.act = list(rt.indices['right'][2])
        self.gain, self.bias = wp.to_torch(rt.m.actuator_gainprm), wp.to_torch(rt.m.actuator_biasprm)
        self.kp0, self.kd0 = self.gain[:, self.act, 0].clone(), -self.bias[:, self.act, 2].clone()     # this episode's weak-wrist draw
        m = rt.mjm
        nom = torch.tensor(m.actuator_gainprm[self.act, 0], device=dev, dtype=self.kp0.dtype)
        self.scale = self.kp0[:, 3:6] / nom[3:6]
        self.kp = self.kp0.clone()
        N = len(self.kp0)
        if rt.grip_target is None:
            rt.grip_target = torch.full((N,), float('nan'), device=dev)
        self.tgt = torch.full((N,), float('nan'), device=dev)
        self.tg0 = self.tgt.clone()

    def wrist(self, alpha):
        if self.gc.wrist_kp is None:
            return
        kp, kd = self.kp0.clone(), self.kd0.clone()
        sk, sd = (torch.tensor(x, device=kp.device, dtype=kp.dtype) for x in (self.gc.wrist_kp, self.gc.wrist_kd))
        kp[:, 3:6] += alpha[:, None] * (sk * self.scale - kp[:, 3:6])
        kd[:, 3:6] += alpha[:, None] * (sd * self.scale - kd[:, 3:6])
        self.gain[:, self.act, 0] = kp
        self.bias[:, self.act, 1] = -kp
        self.bias[:, self.act, 2] = -kd
        self.kp = kp

    def grip_reading(self):
        fo = self.t.runtime.q[:, self.t.finger_qadr].mean(1)
        return (fo - self.gmap.b) / self.gmap.a

    def grip(self, mode, u):
        rt, c, dt = self.t.runtime, self.gc, self.t.dt
        rd = self.grip_reading()
        off = mode == 0
        self.tgt[off] = float('nan')
        new = (mode > 0) & torch.isnan(self.tgt)
        self.tgt[new] = rd[new]
        m = mode == 1
        run = torch.max(self.tgt - c.gentle_rate * dt, rd - c.gentle_delta)
        self.tgt = torch.where(m, torch.min(self.tgt, run), self.tgt)
        self.tg0 = torch.where(m | new, self.tgt, self.tg0)
        sq = mode == 2
        self.tgt = torch.where(sq, self.tg0 * (1 - u), self.tgt)
        rt.grip_target[:] = self.tgt * self.gmap.a + self.gmap.b       # NaN: the plain sign command

    def friction_ff(self, mask):
        pass

    def tick(self, ex):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('baseline', 'relief_v3', *MODES), default='combined')
    ap.add_argument('--episodes', type=int, default=512)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--servo', default='oracle', choices=('oracle', 'nets'))
    ap.add_argument('--fidelity', default='v2')
    ap.add_argument('--servo_roll_deg', type=float, default=None)
    ap.add_argument('--no_inject', action='store_true', help='no servo_err: the nets alone (or the oracle alone)')
    ap.add_argument('--z_bias_mm', type=float, nargs=2, default=(3., 13.))
    ap.add_argument('--press_mm', type=float, default=16.)
    ap.add_argument('--enc_offset_deg', type=float, default=.25)
    ap.add_argument('--episode_s', type=float, default=80.)
    ap.add_argument('--g_reader', type=float, default=.32)
    ap.add_argument('--tag', default='')
    ap.add_argument('--overrides', default='{}', help='JSON GraspV3Cfg overrides')
    ap.add_argument('--out', default='results/real/grasp_v3/oracle')
    a = ap.parse_args()
    gmap = GripperMap(1., 0., .048, a.g_reader, .019)
    phys = BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout='results/usbc/policy/pick/task_layout.json', batch_arm_gains=True)
    from blind_insertion.envs.grasp_fidelity import GraspFidelityCfg as GFC
    phys.grasp_fidelity = GFC.make(a.fidelity)
    if a.servo_roll_deg is not None:
        phys.grasp_fidelity.servo_err, phys.grasp_fidelity.servo_err_jitter_deg = [[a.servo_roll_deg, 0.]], 0.
    if a.no_inject:
        phys.grasp_fidelity.servo_err = None
    legacy = a.mode in ('baseline', 'relief_v3')
    cfg = PickTaskCfg(physics=phys, episode_s=a.episode_s if not legacy else 30., finger_open=.024, enc_offset_deg=a.enc_offset_deg)
    nets = None
    if a.servo == 'nets':
        from blind_insertion.pipeline.pick_servo import load_servo_net, run_net
        P = ROOT / 'runs/perception'
        phys.render_rgb, phys.visual_dr, phys.measured_camera_cadence = True, True, False
        phys.camera_resolution = 320
    t0 = time.time()
    task = PickTask(cfg)
    if a.servo == 'nets':
        nets = {k: load_servo_net(P / f'reader_servo_{v}.pt', task.device) for k, v in (('hover', 'hover'), ('near', 'near'), ('rot', 'near6'))}
    gc = None
    if not legacy:
        gc = GraspV3Cfg(**{**MODES[a.mode], **json.loads(a.overrides)})
    torch.manual_seed(a.seed)
    rows = []
    up = torch.tensor([0., 0., 1.], device=task.device)
    for batch in range(math.ceil(a.episodes / a.nworld)):
        phys.render_rgb = a.servo == 'nets'
        task.reset()
        phys.render_rgb = False
        N = a.nworld
        hooks = SimHooks(task, gc, gmap) if gc is not None else None
        if gc is not None:
            ex = GraspV3Expert(task, seed=a.seed + batch, hold_s=1., settle_max_s=3., hover_tol_m=.004, cfg=gc, hooks=hooks)
        else:
            ex = PickExpert(task, seed=a.seed + batch, hold_s=1., settle_max_s=3., hover_tol_m=.004)
        ex.believed_frame = True
        guarded = gc is not None and gc.touchdown
        ex.tip_floor = -.013
        ex.press_m = 0. if guarded else a.press_mm * 1e-3       # guarded configs: the fallback press comes from the plan's z bias
        if a.mode == 'relief_v3':
            ex.relief = RELIEF_V3
        ex.servo_fracs, ex.servo_rot = (.5, .8, .92), True
        ex.reset()
        z_bias = (torch.rand(N, device=task.device) * (a.z_bias_mm[1] - a.z_bias_mm[0]) + a.z_bias_mm[0]) * 1e-3 if guarded else torch.zeros(N, device=task.device)
        dev = task.device
        F_press, rot_close, rise, tip_close = torch.zeros(N, device=dev), torch.full((N,), float('nan'), device=dev), torch.full((N,), float('nan'), device=dev), torch.full((N,), float('nan'), device=dev)
        roll_true, roll_b, cen_y, cen_x, tilt_c = rot_close.clone(), rot_close.clone(), rot_close.clone(), rot_close.clone(), rot_close.clone()
        z0 = torch.full((N,), float('nan'), device=dev)
        steps = torch.zeros(N, dtype=torch.long, device=dev)
        kp_nom = wp.to_torch(task.runtime.m.actuator_gainprm)[:, list(task.runtime.indices['right'][2]), 0].clone()
        first_rec = torch.zeros(N, dtype=torch.bool, device=dev)
        tip_off = None
        for k in range(task.max_steps - 1):
            prev = ex.phase.clone()
            gprev = ex.g.clone() if gc is not None else None
            act = ex.act()
            want = ((ex.phase == 1) & (prev == 0) & ~ex.refined) | ex.need_frame
            if want.any():
                if nets is None:
                    W = G.grasp_poses(*task.reader_pose(), ex.offset, ex.tilt.nan_to_num(G.TILT_DEG), flip=ex.flip)
                    p_, R_ = task.tcp_pose()
                    delta = torch.cat([(R_.transpose(1, 2) @ (W['grasp'][0] - p_)[..., None])[..., 0], K.rot_log(R_.transpose(1, 2) @ W['grasp'][1])], 1)
                else:
                    phys.render_rgb = True
                    with torch.no_grad():
                        fr = task.runtime.render()['wrist_cam']
                        delta = run_net(nets['hover'][0], nets['hover'][1], fr)
                        if ex.need_frame.any():
                            dn = run_net(nets['near'][0], nets['near'][1], fr)
                            r = torch.cat([nets['rot'][0](fr[i:i + 256]) for i in range(0, N, 256)]) * nets['rot'][1]
                            dn = torch.cat([dn[:, :3], r[:, 3:]], 1)
                            delta = torch.where(ex.need_frame[:, None], dn, delta)
                    phys.render_rgb = False
                _, Rb = task.chain.fk(ex._q())
                delta[:, :3] += (Rb.transpose(1, 2) @ (-z_bias[:, None] * up)[..., None])[..., 0]
                hov, old = want & ~ex.need_frame, ex.p_g.clone()
                ex.servo_update(want, delta)
                big = hov & ((ex.p_g - old).norm(dim=1) > 8e-3) & (ex.hover_iters < 2)
                if nets is not None and big.any():
                    ex.rehover(big)
            # the close: mechanism metrics on the first close only (baseline: PickExpert phases)
            q = ex._q()
            p_a, R_a, J = task.chain.fk(q, with_jac=True)
            e_r = torch.rad2deg(K.rot_log(ex.R_g @ R_a.transpose(1, 2)).norm(dim=1))
            if gc is None:
                closing, closed, near = (ex.phase == 3) & (prev == 2), (ex.phase == 4) & (prev == 3), (ex.phase >= 2) & (ex.phase <= 3)
            else:
                closing = (ex.g >= G_CLOSE) & (ex.g <= G_SQUEEZE) & ~((gprev >= G_CLOSE) & (gprev <= G_SQUEEZE))
                closed = (ex.g >= G_TEST) & (gprev >= G_CLOSE) & (gprev <= G_SQUEEZE)
                near = ((ex.g >= G_HOLD) & (ex.g <= G_SQUEEZE)) | ((ex.g == G_TD) & (ex.gt > 10))
            cl = closing & ~first_rec
            rot_close = torch.where(cl, e_r, rot_close)
            roll_b = torch.where(cl, torch.rad2deg(torch.asin(R_a[:, 2, 1].clamp(-1, 1))), roll_b)
            roll_true = torch.where(cl, torch.rad2deg(torch.asin(task.tcp_pose()[1][:, 2, 1].clamp(-1, 1))), roll_true)
            z0 = torch.where(cl, p_a[:, 2], z0)
            if cl.any():    # true xy centring at the first close, hand frame (y: closing axis, x: across)
                W_ = G.grasp_poses(*task.reader_pose(), ex.offset, ex.tilt.nan_to_num(G.TILT_DEG), flip=ex.flip)
                pT_, RT_ = task.tcp_pose()
                eh = (RT_.transpose(1, 2) @ (W_['grasp'][0] - pT_)[..., None])[..., 0] * 1e3
                cen_y, cen_x = torch.where(cl, eh[:, 1], cen_y), torch.where(cl, eh[:, 0], cen_x)
                tilt_c = torch.where(cl, torch.rad2deg(torch.acos((-RT_[:, 2, 2]).clamp(-1, 1))), tilt_c)
            first_rec |= cl
            if closed.any():
                ok_r = closed & torch.isnan(rise)
                rise = torch.where(ok_r, (p_a[:, 2] - z0) * 1e3, rise)
                tip_close = torch.where(ok_r, task.tip_height(task.q_true()), tip_close)
            kp_ = hooks.kp if hooks is not None else task.runtime.m and kp_nom
            tau = kp_ * (task.joint_target[:, 0] - q)
            JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, device=dev)
            f = torch.linalg.solve(JJt, (J @ tau[..., None]))[..., 0]
            F_press = torch.where(near, torch.max(F_press, -f[:, 2]), F_press)
            task.step(act)
            d = ex.done if gc is not None else ex.phase >= 6
            steps = torch.where(d & (steps == 0), torch.full_like(steps, k + 1), steps)
            if bool(d.all()):
                break
        m = task.pick_metrics()
        lifted = m['lift_m'] > .06
        done = ex.done
        phys_ok = done & ~task.failed & m['finite'] & lifted
        if gc is not None:
            phys_ok &= ~ex.gave_up
            first = (ex.first_held == 1) & (ex.attempt == 1) & phys_ok
            att = ex.attempt.float()
            t_grasp = ex.t_grasp
        else:
            first, att, t_grasp = phys_ok.clone(), torch.ones(N, device=dev), steps.float() * task.dt
        tip_true_mm = tip_close * 1e3
        rows.append({k: v.detach().cpu().numpy() for k, v in dict(
            success=phys_ok, first=first, attempts=att, t_grasp_s=t_grasp, steps=steps, press_N=F_press, rot_close_deg=rot_close, roll_close_deg=roll_b, roll_close_true_deg=roll_true, rise_mm=rise, centre_y_mm=cen_y, centre_x_mm=cen_x, tilt_close_deg=tilt_c,
            tip_true_mm=tip_true_mm, z_bias_mm=z_bias * 1e3, failed=task.failed, lift_mm=m['lift_m'] * 1e3).items()})
        print(f'batch {batch}: success {int(phys_ok.sum())}/{N} first {int(first.sum())} t={time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame({k: np.concatenate([r[k] for r in rows])[:a.episodes] for k in rows[0]})
    n, k, kf = len(df), int(df.success.sum()), int(df['first'].sum())
    pc = lambda c, q: float(np.nanpercentile(df[c], q))   # noqa: E731
    s = dict(mode=a.mode, servo=a.servo, fidelity=a.fidelity, servo_roll_deg=a.servo_roll_deg, z_bias_mm=list(a.z_bias_mm), press_mm=a.press_mm, n=n, success=k, rate=k / n, wilson95=wilson(k, n),
             first=kf, first_rate=kf / n, first_wilson95=wilson(kf, n), attempts_mean=float(df.attempts.mean()),
             t_grasp_s_p50=pc('t_grasp_s', 50), t_grasp_s_p99=pc('t_grasp_s', 99), press_N_p50=pc('press_N', 50), press_N_p99=pc('press_N', 99),
             rot_close_deg_p50=pc('rot_close_deg', 50), rot_close_deg_p99=pc('rot_close_deg', 99), roll_close_deg_p5=pc('roll_close_deg', 5), roll_close_deg_p50=pc('roll_close_deg', 50), roll_close_deg_p95=pc('roll_close_deg', 95),
             roll_true_deg_p5=pc('roll_close_true_deg', 5), roll_true_deg_p50=pc('roll_close_true_deg', 50), roll_true_deg_p95=pc('roll_close_true_deg', 95), rise_mm_p50=pc('rise_mm', 50), rise_mm_p99=pc('rise_mm', 99),
             centre_y_abs_mm_p50=float(np.nanpercentile(df.centre_y_mm.abs(), 50)), centre_y_abs_mm_p95=float(np.nanpercentile(df.centre_y_mm.abs(), 95)),
             centre_x_abs_mm_p50=float(np.nanpercentile(df.centre_x_mm.abs(), 50)), centre_x_abs_mm_p95=float(np.nanpercentile(df.centre_x_mm.abs(), 95)),
             tilt_close_deg_p5=pc('tilt_close_deg', 5), tilt_close_deg_p50=pc('tilt_close_deg', 50), tilt_close_deg_p95=pc('tilt_close_deg', 95),
             tip_true_mm_p50=pc('tip_true_mm', 50), tip_true_mm_p99=pc('tip_true_mm', 99), overrides=json.loads(a.overrides))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rt_ = 'none' if a.no_inject else 'boot' if a.servo_roll_deg is None else f'{a.servo_roll_deg:g}'
    tag = f'{a.mode}_{a.fidelity}_roll{rt_}_{a.servo}{a.tag}_s{a.seed}'
    df.to_parquet(out / f'{tag}.parquet')
    (out / f'{tag}.json').write_text(json.dumps(s, indent=1) + '\n')
    print(json.dumps(s, indent=1), flush=True)


if __name__ == '__main__':
    main()
