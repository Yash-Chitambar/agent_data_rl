"""Batched sim evaluation of the guarded grasp (blind_insertion/pipeline/grasp_v3.py) against the current expert.

The stock sim grasps ~100 % at every tilt and opening (docs/real/PICK_GRASP_SIM_HANDOFF_20261101.md): its contact does
not fail the way the robot fails. So this eval adds pessimistic perturbations:
  --wrist_soft s      all right-wrist gains x s (calibrated so the baseline press turns the hand about the real 3.5 deg)
  --z_bias_mm a b     the plan sits b ~ U(a, b) mm too low (real tips stop 0.7..4.3 mm above the plan)
and reports the mechanism at the close next to physics success: hand rotation against the plan, PD press force,
hand rise, true fingertip height. `proxy_ok` is an ASSUMED geometric pass (--proxy rot rise tip), not measured physics.

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
from blind_insertion.pipeline.grasp_v3 import G_CLOSE, G_DONE, G_HOLD, G_SQUEEZE, G_TEST, GraspV3Cfg, GraspV3Expert  # noqa: E402
from blind_insertion.scenes import reader_grasp as G  # noqa: E402
from deploy.conventions import GripperMap  # noqa: E402

MODES = {
    's1_touchdown': dict(touchdown=True, wrist_kp=None, wrist_kd=None, gentle_close=False, holddown_mm=0., max_attempts=1, td_friction_ff=False),
    's2_wrist': dict(touchdown=False, hold_from='press', gentle_close=False, holddown_mm=0., max_attempts=1),
    's3_gentle_hold': dict(touchdown=False, hold_from='measured', hold_offset_mm=0., wrist_kp=None, wrist_kd=None, holddown_mm=3., max_attempts=1),
    's4_retry': dict(touchdown=False, hold_from='press', wrist_kp=None, wrist_kd=None, gentle_close=False, holddown_mm=0., max_attempts=3),
    'combined': dict(td_friction_ff=False),
}


def wilson(k, n, z=1.96):
    if not n:
        return (0., 0.)
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


class SimHooks:
    """grasp_v3 hooks on the batched PickTask: same reading map and force rule as deploy/bimanual ChainHooks + SimIO."""

    def __init__(self, task, gc, gmap, soft):
        self.t, self.gc, self.gmap, self.soft = task, gc, gmap, soft
        rt = task.runtime
        self.act = list(rt.indices['right'][2])
        self.gain, self.bias = wp.to_torch(rt.m.actuator_gainprm), wp.to_torch(rt.m.actuator_biasprm)
        if not hasattr(task, '_gains0'):      # nominal gains, once: a later batch must not scale them again
            task._gains0 = (self.gain[:, self.act, 0].clone(), -self.bias[:, self.act, 2].clone())
        self.kp0, self.kd0 = task._gains0
        self.base = rt.grip_scale.clone() if rt.grip_scale is not None else torch.ones(rt.cfg.nworld, device=task.device)
        self.apply(torch.zeros(len(self.kp0), device=task.device))

    def apply(self, alpha):
        kp, kd = self.kp0.clone(), self.kd0.clone()
        if self.gc is not None and self.gc.wrist_kp is not None:
            sk, sd = (torch.tensor(x, device=kp.device, dtype=kp.dtype) for x in (self.gc.wrist_kp, self.gc.wrist_kd))
            kp[:, 3:6] += alpha[:, None] * (sk - kp[:, 3:6])
            kd[:, 3:6] += alpha[:, None] * (sd - kd[:, 3:6])
        kp[:, 3:6] *= self.soft
        kd[:, 3:6] *= math.sqrt(self.soft)
        self.gain[:, self.act, 0] = kp
        self.bias[:, self.act, 1] = -kp
        self.bias[:, self.act, 2] = -kd
        self.kp = kp

    def wrist(self, alpha):
        self.apply(alpha)

    def grip_reading(self):
        fo = self.t.runtime.q[:, self.t.finger_qadr].mean(1)
        return (fo - self.gmap.b) / self.gmap.a

    def grip(self, mode, u):
        rt = self.t.runtime
        frac = torch.ones_like(u)
        low = self.gc.gentle_delta / .32
        frac[mode == 1] = low
        sq = mode == 2
        frac[sq] = low + (1 - low) * u[sq]
        rt.grip_scale = self.base * frac

    def friction_ff(self, mask):
        pass

    def tick(self, ex):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('baseline', *MODES), default='combined')
    ap.add_argument('--episodes', type=int, default=512)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--servo', default='oracle', choices=('oracle', 'nets'))
    ap.add_argument('--wrist_soft', type=float, default=1.)
    ap.add_argument('--z_bias_mm', type=float, nargs=2, default=(0., 0.))
    ap.add_argument('--proxy', type=float, nargs=3, default=(2., 1.5, 3.), help='assumed pinch pass: rotation deg, rise mm, true tip height mm')
    ap.add_argument('--episode_s', type=float, default=80.)
    ap.add_argument('--g_reader', type=float, default=.32)
    ap.add_argument('--tag', default='')
    ap.add_argument('--overrides', default='{}', help='JSON GraspV3Cfg overrides')
    ap.add_argument('--out', default='results/real/grasp_v3/oracle')
    a = ap.parse_args()
    gmap = GripperMap(1., 0., .048, a.g_reader, .019)
    phys = BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout='results/usbc/policy/pick/task_layout.json', batch_arm_gains=True)
    cfg = PickTaskCfg(physics=phys, episode_s=a.episode_s if a.mode != 'baseline' else 25.)
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
    if a.mode != 'baseline':
        gc = GraspV3Cfg(**{**MODES[a.mode], **json.loads(a.overrides)})
    torch.manual_seed(a.seed)
    rows = []
    up = torch.tensor([0., 0., 1.], device=task.device)
    from deploy.safety_task import FlexChecker  # noqa: F401
    fq = task.finger_qadr
    tips = (lambda q: None)
    for batch in range(math.ceil(a.episodes / a.nworld)):
        phys.render_rgb = a.servo == 'nets'
        task.reset()
        phys.render_rgb = False
        N = a.nworld
        hooks = SimHooks(task, gc, gmap, a.wrist_soft)
        if gc is not None:
            ex = GraspV3Expert(task, seed=a.seed + batch, hold_s=1., settle_max_s=3., hover_tol_m=.004, cfg=gc, hooks=hooks)
        else:
            ex = PickExpert(task, seed=a.seed + batch, hold_s=1., settle_max_s=3., hover_tol_m=.004)
        ex.believed_frame = True
        ex.servo_fracs, ex.servo_rot = (.5, .8, .92), True
        ex.reset()
        z_bias = (torch.rand(N, device=task.device) * (a.z_bias_mm[1] - a.z_bias_mm[0]) + a.z_bias_mm[0]) * 1e-3
        dev = task.device
        F_press, rot_close, rise, tip_close = torch.zeros(N, device=dev), torch.full((N,), float('nan'), device=dev), torch.full((N,), float('nan'), device=dev), torch.full((N,), float('nan'), device=dev)
        z0 = torch.full((N,), float('nan'), device=dev)
        steps = torch.zeros(N, dtype=torch.long, device=dev)
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
                near = (ex.g >= G_HOLD) & (ex.g <= G_SQUEEZE) | ((ex.g == 1) & (ex.gt > 10))
            cl = closing & ~first_rec
            rot_close = torch.where(cl, e_r, rot_close)
            z0 = torch.where(cl, p_a[:, 2], z0)
            first_rec |= cl
            q_t = task.q_true()
            if closed.any():
                ok_r = closed & torch.isnan(rise)
                rise = torch.where(ok_r, (p_a[:, 2] - z0) * 1e3, rise)
                pads = task.runtime.q  # noqa: F841
                tip_close = torch.where(ok_r, task.tcp_pose()[0][:, 2], tip_close)
            tau = hooks.kp * (task.joint_target[:, 0] - q)
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
        phys_ok = done & ~task.failed & m['finite'] & lifted & (m['tilt_deg'] < 3)
        if gc is not None:
            phys_ok &= ~ex.gave_up
            first = (ex.first_held == 1) & (ex.attempt == 1) & phys_ok
            att = ex.attempt.float()
            t_grasp = ex.t_grasp
        else:
            first, att, t_grasp = phys_ok.clone(), torch.ones(N, device=dev), steps.float() * task.dt
        tip_true_mm = (tip_close - task.table_top) * 1e3
        rows.append({k: v.detach().cpu().numpy() for k, v in dict(
            success=phys_ok, first=first, attempts=att, t_grasp_s=t_grasp, steps=steps, press_N=F_press, rot_close_deg=rot_close, rise_mm=rise,
            tcp_above_table_mm=tip_true_mm, z_bias_mm=z_bias * 1e3, failed=task.failed, lift_mm=m['lift_m'] * 1e3).items()})
        print(f'batch {batch}: success {int(phys_ok.sum())}/{N} first {int(first.sum())} t={time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame({k: np.concatenate([r[k] for r in rows])[:a.episodes] for k in rows[0]})
    pr = a.proxy
    tcp0 = float(df.tcp_above_table_mm.quantile(.01))     # lowest TCP height over the run = tips about on the table
    df['proxy_ok'] = df.success & (df.rot_close_deg < pr[0]) & (df.rise_mm.abs() < pr[1]) & (df.tcp_above_table_mm - tcp0 < pr[2])
    n, k, kf, kp_ = len(df), int(df.success.sum()), int(df['first'].sum()), int(df.proxy_ok.sum())
    pc = lambda c, q: float(np.nanpercentile(df[c], q))   # noqa: E731
    s = dict(mode=a.mode, servo=a.servo, wrist_soft=a.wrist_soft, z_bias_mm=list(a.z_bias_mm), n=n, success=k, rate=k / n, wilson95=wilson(k, n), first=kf, first_rate=kf / n,
             first_wilson95=wilson(kf, n), proxy_ok=kp_, proxy_rate=kp_ / n, proxy_wilson95=wilson(kp_, n), proxy_assumed=list(pr), attempts_mean=float(df.attempts.mean()),
             t_grasp_s_p50=pc('t_grasp_s', 50), t_grasp_s_p99=pc('t_grasp_s', 99), press_N_p50=pc('press_N', 50), press_N_p99=pc('press_N', 99),
             rot_close_deg_p50=pc('rot_close_deg', 50), rot_close_deg_p99=pc('rot_close_deg', 99), rise_mm_p50=pc('rise_mm', 50), rise_mm_p99=pc('rise_mm', 99),
             tcp_above_table_mm_p50=pc('tcp_above_table_mm', 50), tcp_above_table_mm_p99=pc('tcp_above_table_mm', 99), overrides=json.loads(a.overrides))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.mode}_soft{a.wrist_soft:g}_z{a.z_bias_mm[0]:g}-{a.z_bias_mm[1]:g}_{a.servo}{a.tag}_s{a.seed}'
    df.to_parquet(out / f'{tag}.parquet')
    (out / f'{tag}.json').write_text(json.dumps(s, indent=1) + '\n')
    print(json.dumps(s, indent=1), flush=True)


if __name__ == '__main__':
    main()
