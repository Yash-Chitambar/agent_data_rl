"""Evaluate a scripted left-arm hold of the PC under insertion loads (policy Part 3), GPU runtime.

Per episode (all worlds in parallel, full runtime DR: PC pose +/-3 cm/+/-10 deg, PC-feet friction
U(0.3, 0.9), arm Coulomb x U(0.5, 1.3), gravity-scale jitter; plus encoder-offset error +/-1.5 deg on
joints 2-5 and 0-1 step action latency on both arms):
  1. engage: the left arm moves from the reset pose to a stand-off, then to the hold target
     (IK contact point + Kp^-1 J^T F), settles;     (right arm at its reset pose)
  2. the reader is put in the right gripper at the pre-insert pose (6 mm out), settle 0.3 s;
  3. load: either
       push   -- external force on the PC at the port, along the insertion axis (+ lateral
                 components), ramped over 0.5 s to F ~ U(push_lo, push_hi) N and held;
       insert -- privileged-feedback right-arm insertion (audit_bimanual_pick.py --insert ported)
                 to the seat, optionally overdriven past it (extra axial force through the PD error).
PC motion is measured from the start of the load phase (port point translation, rotation angle).
Gate (POLICY_PLAN Part 3): translation < 0.5 mm and rotation < 0.5 deg in >= 99.5 % of n >= 1000
episodes, no arm-arm contact.

    PYTHONPATH=. .venv/bin/python scripts/task/eval_left_hold.py --mode press --force 25 --load push \
        --push 0 10 --nworld 1024 --out results/usbc/policy/hold/press_f25_push10.json
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import numpy as np
import torch

from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
from blind_insertion.envs.left_hold import Harness, HarnessCfg, HoldCfg, InsertCfg, LeftHold, RightInserter
from blind_insertion.envs import kinematics as K

DT = .02


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def run(a, rt=None, seed=0):
    torch.manual_seed(seed)
    rt = rt or BimanualRuntime(BimanualCfg(nworld=a.nworld, device='cuda:0'))
    h = Harness(rt, HarnessCfg(enc_offset_deg=a.enc_offset_deg))
    h.reset()
    hc = HoldCfg(mode=a.mode, force_n=a.force, preload_n=a.preload, across_m=a.across, along_m=a.along,
                 brace_height_m=a.brace_height, down_n=a.down, tilt_deg=a.tilt, engage_s=a.engage_s,
                 settle_s=a.settle_s)
    hold = LeftHold(h, hc)
    ic = InsertCfg(overdrive_m=a.overdrive)
    ins = RightInserter(h, ic)
    n, dev = h.n, h.dev
    p_reset, R_reset = h.pc_pose()
    # the hold is planned from a PC pose estimate: true pose + estimation error (Part 2 ZED numbers)
    est_p = p_reset + torch.randn(n, 3, device=dev) * a.pc_est_sigma_m * torch.tensor([1., 1., 0.], device=dev)
    yaw_err = torch.randn(n, device=dev) * math.radians(a.pc_est_sigma_deg)
    Rz = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=dev).expand(n, 3), yaw_err)
    plan = hold.solve(est_p, Rz @ R_reset) if a.mode != 'none' else {'ok': torch.ones(n, dtype=torch.bool, device=dev)}
    q_home_l, q_home_r = h.believed('left').clone(), h.believed('right').clone()
    flags = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ('arm_arm', 'left_other', 'nonfinite')}
    T = int(round(hold.duration / DT))
    for k in range(T):
        h.step(q_home_r, 1., hold.target(k * DT, q_home_l), -1. if a.mode != 'none' else 1.)
    c = h.contacts()
    flags['left_other'] |= c['left_other']
    p_eng, R_eng = h.pc_pose()
    engage_dp = (p_eng - p_reset).norm(dim=1)
    engage_ang = K.rot_log(R_eng @ R_reset.transpose(1, 2)).norm(dim=1)
    ins.place()
    q_hold = hold.target(1e9, q_home_l)
    for k in range(int(.3 / DT)):
        h.step(ins.target, -1., q_hold, -1. if a.mode != 'none' else 1.)
    p0, R0 = h.pc_pose()
    dp_max = torch.zeros(n, device=dev); ang_max = torch.zeros(n, device=dev)
    force_sum = torch.zeros(n, 3, device=dev); nf = 0
    push = torch.rand(n, device=dev) * (a.push[1] - a.push[0]) + a.push[0]
    lateral = (torch.rand(n, 2, device=dev) * 2 - 1) * a.lateral_frac
    if a.load == 'push':
        steps = int((.5 + a.push_hold_s) / DT)
    else:
        steps = int(ins.duration / DT)
    for k in range(steps):
        t = k * DT
        port_p, port_R = h.pc_pose()
        if a.load == 'push':
            ramp = min(1., t / .5)
            f = -port_R[:, :, 2] + port_R[:, :, 0] * lateral[:, :1] + port_R[:, :, 1] * lateral[:, 1:]
            h.push_pc(f * (push * ramp)[:, None], port_p)
            tgt = ins.target
        else:
            tgt = ins.track(ins.gap(t))
        h.step(tgt, -1., q_hold, -1. if a.mode != 'none' else 1.)
        pp, RR = h.pc_pose()
        fin = torch.isfinite(rt.q).all(1)
        flags['nonfinite'] |= ~fin
        dp = torch.nan_to_num((pp - p0).norm(dim=1), nan=1.)
        ang = torch.nan_to_num(K.rot_log(RR @ R0.transpose(1, 2)).norm(dim=1), nan=1.)
        dp_max = torch.maximum(dp_max, dp); ang_max = torch.maximum(ang_max, ang)
        if k % 5 == 0:
            c = h.contacts()
            flags['arm_arm'] |= c['arm_arm']; flags['left_other'] |= c['left_other']
            force_sum += c['left_pc_force']; nf += 1
    h.push_pc(torch.zeros(n, 3, device=dev), p0)
    seated = rt.insertion_metrics()['geometric_seated'] if a.load == 'insert' else torch.zeros(n, dtype=torch.bool, device=dev)
    ok = plan['ok'] & ins.ik_ok
    still = (dp_max < a.tol_mm * 1e-3) & (torch.rad2deg(ang_max) < a.tol_deg)
    passed = still & ~flags['arm_arm'] & ~flags['nonfinite']
    fmean = force_sum / max(nf, 1)
    feet_mu = rt.geom_friction[:, rt.pc_feet[0], 0] if rt.cfg.pc_friction_range else torch.full((n,), .6, device=dev)
    w = {'dp_mm': dp_max * 1e3, 'ang_deg': torch.rad2deg(ang_max), 'engage_dp_mm': engage_dp * 1e3,
         'engage_ang_deg': torch.rad2deg(engage_ang), 'push_n': push, 'left_force_n': fmean.norm(dim=1),
         'left_force_down_n': -fmean[:, 2], 'feet_mu': feet_mu, 'passed': passed, 'arm_arm': flags['arm_arm'],
         'left_other': flags['left_other'], 'nonfinite': flags['nonfinite'], 'plan_ok': ok, 'seated': seated}
    return {k: v.detach().cpu().numpy() for k, v in w.items()}, hc, ic


def summarize(w, a, hc, ic, seconds):
    n = len(w['passed']); k = int(w['passed'].sum())
    q = lambda x, p: float(np.percentile(x, p))
    out = {'n': n, 'pass_rate': k / n, 'wilson95_lower': wilson_lower(k, n), 'gate_met': bool(n >= 1000 and k / n >= .995),
           'tolerance': {'translation_mm': a.tol_mm, 'rotation_deg': a.tol_deg},
           'arm_arm_contact_worlds': int(w['arm_arm'].sum()), 'left_touches_other_worlds': int(w['left_other'].sum()),
           'nonfinite_worlds': int(w['nonfinite'].sum()), 'plan_failed_worlds': int((~w['plan_ok']).sum()),
           'pc_translation_mm': {f'p{p}': q(w['dp_mm'], p) for p in (50, 90, 99, 99.5, 100)},
           'pc_rotation_deg': {f'p{p}': q(w['ang_deg'], p) for p in (50, 90, 99, 99.5, 100)},
           'engage_translation_mm': {f'p{p}': q(w['engage_dp_mm'], p) for p in (50, 99, 100)},
           'engage_rotation_deg': {f'p{p}': q(w['engage_ang_deg'], p) for p in (50, 99, 100)},
           'left_contact_force_n': {f'p{p}': q(w['left_force_n'], p) for p in (1, 50, 99)},
           'seated_rate': float(w['seated'].mean()) if a.load == 'insert' else None,
           'hold': asdict(hc), 'insert': asdict(ic) if a.load == 'insert' else None,
           'load': {'type': a.load, 'push_n': a.push if a.load == 'push' else None, 'lateral_frac': a.lateral_frac,
                    'push_hold_s': a.push_hold_s},
           'pc_estimate_error': {'sigma_m': a.pc_est_sigma_m, 'sigma_yaw_deg': a.pc_est_sigma_deg},
           'enc_offset_deg': a.enc_offset_deg, 'wall_s': seconds}
    # failure rate by push force bin and feet friction bin
    if a.load == 'push' and a.push[1] > a.push[0]:
        edges = np.linspace(a.push[0], a.push[1], 6)
        out['pass_by_push_n'] = {f'{lo:.0f}-{hi:.0f}': float(w['passed'][(w['push_n'] >= lo) & (w['push_n'] < hi)].mean())
                                 for lo, hi in zip(edges[:-1], edges[1:])}
    edges = np.linspace(.3, .9, 4)
    out['pass_by_feet_mu'] = {f'{lo:.1f}-{hi:.1f}': float(w['passed'][(w['feet_mu'] >= lo) & (w['feet_mu'] < hi + 1e-9)].mean())
                              for lo, hi in zip(edges[:-1], edges[1:])}
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='press', choices=('press', 'brace', 'none'))
    ap.add_argument('--force', type=float, default=25.)
    ap.add_argument('--preload', type=float, default=0.)
    ap.add_argument('--across', type=float, default=-.0208, help='contact across the panel from the box centre (m); -0.0208 = in line with port 2')
    ap.add_argument('--along', type=float, default=.03)
    ap.add_argument('--brace-height', type=float, default=.02)
    ap.add_argument('--down', type=float, default=0.)
    ap.add_argument('--tilt', type=float, default=0.)
    ap.add_argument('--engage-s', type=float, default=.6)
    ap.add_argument('--settle-s', type=float, default=.5)
    ap.add_argument('--load', default='push', choices=('push', 'insert'))
    ap.add_argument('--push', type=float, nargs=2, default=(0., 10.))
    ap.add_argument('--push-hold-s', type=float, default=1.5)
    ap.add_argument('--lateral-frac', type=float, default=.15)
    ap.add_argument('--overdrive', type=float, default=0.)
    ap.add_argument('--pc-est-sigma-m', type=float, default=.002)
    ap.add_argument('--pc-est-sigma-deg', type=float, default=1.)
    ap.add_argument('--enc-offset-deg', type=float, default=1.5)
    ap.add_argument('--tol-mm', type=float, default=.5)
    ap.add_argument('--tol-deg', type=float, default=.5)
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path)
    return ap


def main():
    a = parser().parse_args()
    t = time.time()
    w, hc, ic = run(a, seed=a.seed)
    s = summarize(w, a, hc, ic, time.time() - t)
    print(json.dumps({k: v for k, v in s.items() if k not in ('hold', 'insert')}, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(s, indent=1) + '\n')
        np.savez_compressed(a.out.with_suffix('.npz'), **w)


if __name__ == '__main__':
    main()
