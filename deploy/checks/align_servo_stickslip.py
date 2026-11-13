"""Continuous align servo against joint stick-slip, no hardware (docs/real/ALIGN_FAST_20261113.md, "Servo sway").

Real 2026-11-10 (servo_1, servo_2): commanded tip speed <= 8.9-11 mm/s, measured tip speed p99 24-25 mm/s; the joints lag their smooth targets
(joint 2 stuck 3.5 s while its target moved 32 mrad, joint 5 25-35 mrad behind) and catch up in jumps. The batched sim has Coulomb joint friction
(static = kinetic), so it has no breakaway jumps (sim dry run p99 13 mm/s). This check: 6 independent joints with static friction Fs > kinetic Fc
(fitted by eye to the real breakaway leads: joint 1 ~0.35, 2 ~1.7, 3 ~1.3, 4 ~0.2, 5 ~0.4, 6 ~0.15 N m incl. the feed-forward), a constant load,
kp/kd of the real arm, 8 ms command latency, 1 kHz physics; YAM kinematics (FastChain); port vision at 23 frames/s, 40 ms latency, smooth
pose error (0.25, 0.74, 0.49) mm rms + 0.1 mm noise. The servo law is ServoStage's (deploy/bimanual/align.py), with its new options.
A simplified plant: it ranks the options; the rig decides.

    python -m deploy.checks.align_servo_stickslip [--n 64] [--json results/real_align/align_fast/servo_stickslip.json]
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
from blind_insertion.envs import friction_comp as FC  # noqa: E402
from blind_insertion.envs.inhand import measured_mean  # noqa: E402
from deploy.bimanual.align import SERVO_DEFAULTS, servo_lead_cut  # noqa: E402
from deploy.bimanual.scene import SceneModel  # noqa: E402

Q0 = np.array([1.5261, 2.1293, 1.2498, -0.4939, 0.9356, 1.4017])       # servo_1 start (believed)
KP = np.array([80., 80., 80., 40., 10., 10.])
KD = np.array([5., 5., 5., 1.5, 1.5, 1.5])
INERTIA = np.array([.08, .08, .06, .01, .005, .005])
FS = np.array([.35, 1.7, 1.3, .2, .4, .15])
FF_TAU = np.array([.0275, .546, .819, .0463, .0433, .0467])


class Arm:
    def __init__(self, g, stick=True, fs_scale=1.):
        self.Fs = FS * fs_scale * g.uniform(.8, 1.25, 6)
        self.Fc = self.Fs * (g.uniform(.35, .7, 6) if stick else 1.)
        self.load = g.uniform(-.5, .5, 6) * self.Fc
        self.q, self.v = Q0.copy(), np.zeros(6)
        self.stuck = np.ones(6, bool)
        self.buf = [(Q0.copy(), np.zeros(6))] * 8

    def control_step(self, tgt, ff):
        for _ in range(20):
            self.buf.append((tgt, ff))
            t, f = self.buf.pop(0)
            tau = KP * (t - self.q) - KD * self.v + self.load + f
            brk = self.stuck & (np.abs(tau) > self.Fs)
            self.stuck &= ~brk
            mv = ~self.stuck
            s = np.where(self.v != 0, np.sign(self.v), np.sign(tau))
            a = (tau - s * self.Fc) / INERTIA
            v2 = self.v + a * 1e-3
            stop = mv & (self.v != 0) & (np.sign(v2) != np.sign(self.v)) & (np.abs(tau) <= self.Fs)
            self.v = np.where(mv & ~stop, v2, 0.)
            self.stuck |= stop
            self.q = self.q + self.v * 1e-3


def run(sv, seed, stick=True, fs_scale=1., max_s=20.):
    g = np.random.default_rng(seed)
    arm = Arm(g, stick, fs_scale)
    chain = SC.chain['right']
    yaw = math.radians(5.)
    zax = np.array([-math.cos(yaw), -math.sin(yaw), 0.])
    eR = np.c_[np.cross([0., 0., 1.], zax), [0., 0., 1.], zax]

    def tip(q):
        p, R = chain.fk(torch.as_tensor(q, dtype=torch.float32)[None])
        return (p[0] - R[0] @ TIP_S).numpy().astype(float)
    start = np.array([g.uniform(-2, 2), g.uniform(-2, 2), g.uniform(16, 19)])
    port = tip(Q0) - eR @ start * 1e-3                          # truth (port axes, mm) = eR^T (tip - port)
    truth = lambda q: eR.T @ (tip(q) - port) * 1e3              # noqa: E731
    k_ph = g.uniform(0, 2 * math.pi, 3)
    amp = np.array([.25, .74, .49]) * math.sqrt(2)
    bias = g.normal(0, 1, 3) * np.array([.3, .6, .4])

    def vision(p):
        return p + bias + amp * np.sin(2 * math.pi * p / 4. + k_ph) + g.normal(0, .1, 3)
    tgt = Q0.copy()
    qp = Q0.copy()
    hist, frames = [], []
    prev_tw, confirm = np.zeros(6), 0
    st = None
    goal0 = np.array([0., 0., 6.])
    n = int(max_s / .02)
    est = None
    for k in range(n):
        q = arm.q.copy()
        tr = truth(q)
        hist.append(tr)
        if k % 2 == 0:
            frames.append((k + 2, k, vision(tr)))                # ready at k + 2 (40 ms), captured at k
        new = False
        while frames and frames[0][0] <= k:
            _, kc, v = frames.pop(0)
            meas = (v, kc)
            new = True
        if new or est is not None:
            v, kc = meas
            est = v + (tr - hist[kc])                              # newest frame + believed motion since its capture
        spd = np.linalg.norm(hist[-1] - hist[-6]) / .1 if len(hist) > 5 else 0.
        if est is None:
            arm.control_step(tgt, FC.feedforward(tgt - q, FF_TAU, .005) if sv['ff'] else np.zeros(6))
            qp = q
            continue
        goal = goal0.copy()
        if np.hypot(*(est[:2] - goal[:2])) > 2.:
            goal[2] = max(goal[2], 9.)
        err = est - goal
        if new:
            confirm = confirm + 1 if ((np.abs(err) <= 1.).all() and spd <= sv['still_mm_s']) else 0
        if confirm >= 2:
            return dict(ok=True, t=k * .02, hist=np.array(hist), true_end=tr)
        inside = np.linalg.norm(err) <= sv['deadband_mm']
        vp = -sv['gain'] * err
        nv = np.linalg.norm(vp)
        if sv['vmin_mm_s'] > 0 and 1e-6 < nv < sv['vmin_mm_s']:
            vp = vp / nv * sv['vmin_mm_s']
        if inside:
            vp[:] = 0.
        vmax = sv['vmax_mm_s']
        if sv.get('near_mm') is not None and np.linalg.norm(err) < sv['near_mm']:
            vmax = sv['vmax_near_mm_s']
        tw = np.r_[np.clip(vp, -vmax, vmax), 0., 0., 0.]
        tw = prev_tw + sv['smooth'] * (tw - prev_tw)
        prev_tw = tw
        p, R, J = chain.fk(torch.as_tensor(q, dtype=torch.float32)[None], with_jac=True)
        r = -(R[0].numpy() @ TIP_S.numpy())
        vw = eR @ (tw[:3] * 1e-3)
        x = np.r_[vw - np.cross(np.zeros(3), r), np.zeros(3)] * .02
        Jn = J[0].numpy().astype(float)
        tgt = tgt + Jn.T @ np.linalg.solve(Jn @ Jn.T + sv['damping'] * np.eye(6), x)
        if sv.get('lead_cut'):
            st = servo_lead_cut(tgt, q, qp, st, sv)
            tgt = st['tgt']
        cap = sv['target_cap_rad']
        tgt = np.clip(tgt, q - cap, q + cap)
        qp = q
        arm.control_step(tgt, FC.feedforward(tgt - q, FF_TAU, .005) if sv['ff'] else np.zeros(6))
    return dict(ok=False, t=max_s, hist=np.array(hist), true_end=hist[-1])


def metrics(rs):
    sp = np.concatenate([np.linalg.norm(r['hist'][5:] - r['hist'][:-5], axis=1) / .1 for r in rs])
    ok = [r for r in rs if r['ok']]
    path = [np.linalg.norm(np.diff(r['hist'], axis=0), axis=1).sum() / max(np.linalg.norm(r['hist'][-1] - r['hist'][0]), 1e-6) for r in rs]
    lat = [float(np.hypot(*r['true_end'][:2])) for r in ok]
    return dict(n=len(rs), done=len(ok), time_p50=float(np.median([r['t'] for r in ok])) if ok else None, time_p90=float(np.percentile([r['t'] for r in ok], 90)) if ok else None,
                tip_speed_p50=float(np.percentile(sp, 50)), tip_speed_p99=float(np.percentile(sp, 99)), run_p99_max=float(max(np.percentile(np.linalg.norm(r['hist'][5:] - r['hist'][:-5], axis=1) / .1, 99) for r in rs)),
                path_ratio_p50=float(np.median(path)), true_lat_p50=float(np.median(lat)) if lat else None, true_gap_err_p90=float(np.percentile([abs(r['true_end'][2] - 6) for r in ok], 90)) if ok else None)


CASES = {
    'servo (align6), Coulomb plant': (dict(), dict(stick=False)),
    'servo (align6), stick-slip plant': (dict(), dict()),
    'servo, feed-forward off': (dict(ff=False), dict()),
    'servo, vmax 4': (dict(vmax_mm_s=4.), dict()),
    'servo, gain 0.8': (dict(gain=.8), dict()),
    'servo, lead cut': (dict(lead_cut=True), dict()),
    'servo, lead cut, vmax 6, near 3 mm -> 3 mm/s': (dict(lead_cut=True, vmax_mm_s=6., near_mm=3., vmax_near_mm_s=3.), dict()),
    'servo, lead cut release 0.5': (dict(lead_cut=True, lead_cut_release=.5), dict()),
    'servo, lead cut release 0.7': (dict(lead_cut=True, lead_cut_release=.7), dict()),
    'servo, lead cut 0.5, vmin 2.5': (dict(lead_cut=True, lead_cut_release=.5, vmin_mm_s=2.5), dict()),
    'servo, lead cut 0.5, Coulomb plant': (dict(lead_cut=True, lead_cut_release=.5), dict(stick=False)),
    'servo (align6), stick-slip plant, Fs x0.7': (dict(), dict(fs_scale=.7)),
    'servo, lead cut 0.5, stick-slip plant, Fs x0.7': (dict(lead_cut=True, lead_cut_release=.5), dict(fs_scale=.7)),
    'servo, lead cut 0.3, vmin 2.5': (dict(lead_cut=True, lead_cut_release=.3, vmin_mm_s=2.5), dict()),
    'servo, lead cut 0.3, vmin 2.5, Fs x0.7': (dict(lead_cut=True, lead_cut_release=.3, vmin_mm_s=2.5), dict(fs_scale=.7)),
    'servo, lead cut 0.5, vmin 2.5, vmax 5': (dict(lead_cut=True, lead_cut_release=.5, vmin_mm_s=2.5, vmax_mm_s=5.), dict()),
    'servo, lead cut 0.5, vmin 2.5, vmax 5, Fs x0.7': (dict(lead_cut=True, lead_cut_release=.5, vmin_mm_s=2.5, vmax_mm_s=5.), dict(fs_scale=.7)),
    'servo, lead cut 0.5, vmin 2.5, Fs x0.7': (dict(lead_cut=True, lead_cut_release=.5, vmin_mm_s=2.5), dict(fs_scale=.7)),
}


def main():
    global SC, TIP_S
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--json')
    ap.add_argument('--only')
    a = ap.parse_args()
    SC = SceneModel('cpu')
    Rs, ps = measured_mean(device='cpu')
    TIP_S = (Rs.reshape(3, 3).float().T @ ps.reshape(3).float())
    out = {}
    for name, (kw, pk) in CASES.items():
        if a.only and not any(o in name for o in a.only.split('|')):
            continue
        sv = {**SERVO_DEFAULTS, 'gain': 1.5, 'vmax_mm_s': 8., 'vmin_mm_s': 1.5, 'deadband_mm': .4, 'smooth': .2, **kw}
        rs = [run(sv, 1000 + i, **pk) for i in range(a.n)]
        out[name] = m = metrics(rs)
        print(f"{name:48s} done {m['done']}/{m['n']} t p50/p90 {m['time_p50']}/{m['time_p90']} tip speed p50/p99 {m['tip_speed_p50']:.1f}/{m['tip_speed_p99']:.1f} "
              f"(worst run p99 {m['run_p99_max']:.1f}) path {m['path_ratio_p50']:.2f} true lat p50 {m['true_lat_p50']}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
