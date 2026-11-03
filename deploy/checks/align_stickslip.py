"""Creep against stick-slip, one joint, no hardware: does a small move reach its goal? Plant fitted by eye to the real joint 2 (kp 80, kd 5, friction band ~1.3 N m,
operator jogs 2026-11-02: asked 0.5-1 mm, moved 0 then 4-11 mm). Static friction Fs, kinetic Fc < Fs, 8 ms output latency, lever 0.25 m from joint to plug tip.

It compares the way the collect stage moves (target ramp, servo on the error) with deploy/bimanual/align.Creep. n random steps per case, random friction. The plant is a simplified model:
it is the argument for the method, not a measurement. The rig decides.

    python -m deploy.checks.align_stickslip [--n 500] [--json results/real_align/stickslip.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual.align import DEFAULTS, Creep  # noqa: E402

KP, KD, LEVER, I = 80., 5., .25, .35


class Plant:
    def __init__(self, g):
        self.Fs = g.uniform(1.0, 1.9)
        self.Fc = self.Fs * g.uniform(.35, .7)
        self.load = g.uniform(-.9, .9)
        self.q, self.v = 0., 0.
        self.t_cmd, self.buf = 0., [0.] * 8
        self.stuck = True

    def control_step(self, tgt):
        """20 ms of 1 kHz physics with 8 ms latency on the target."""
        for _ in range(20):
            self.buf.append(tgt)
            t = self.buf.pop(0)
            tau = KP * (t - self.q) - KD * self.v + self.load
            if self.stuck:
                if abs(tau) > self.Fs:
                    self.stuck = False
                else:
                    self.v = 0.
                    continue
            a = (tau - np.sign(self.v if self.v != 0 else tau) * self.Fc) / I
            v2 = self.v + a * 1e-3
            if self.v != 0 and np.sign(v2) != np.sign(self.v):
                self.v, self.stuck = 0., True
            else:
                self.v = v2
                self.q += self.v * 1e-3


def ramp_step(p, goal, g, iters=3):
    """Collect-like: smooth target ramp to goal + accumulated error, settle .5 s, 3 iterations."""
    tgt = p.t_cmd
    for _ in range(iters):
        err = goal - p.q
        if abs(err) * LEVER * 1e3 < .3:
            break
        t0, t1 = tgt, tgt + err
        n = max(15, int(abs(err) * LEVER * 1e3 / 3. / .02))
        for k in range(n):
            u = (k + 1) / n
            u = u * u * (3 - 2 * u)
            p.control_step(t0 + (t1 - t0) * u)
        tgt = t1
        for _ in range(25):
            p.control_step(tgt)
    p.t_cmd = tgt
    return tgt


def creep_step(p, goal, c, cr):
    cr.goal = np.array([0., goal, 0., 0., 0., 0.])
    for k in range(int(c['step_timeout_s'] / .02)):
        cr.ch.q_b = torch.tensor([[0., p.q, 0., 0., 0., 0.], [0.] * 6])
        cr.tick()
        p.control_step(float(cr.ch.last[1]))
        if abs(goal - p.q) * LEVER * 1e3 < c['move_tol_mm'] and abs(p.v) < 1e-3:
            break
    for _ in range(25):
        cr.ch.q_b = torch.tensor([[0., p.q, 0., 0., 0., 0.], [0.] * 6])
        cr.tick()
        p.control_step(float(cr.ch.last[1]))
    return k


def fake_chain():
    ch = SimpleNamespace(dt=.02, dev=torch.device('cpu'), q_b=torch.zeros(2, 6), last=None)

    def hstep(t, q, label):
        ch.last = t.numpy()
    ch.hstep = hstep
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None)
    A = ap.parse_args()
    c = dict(DEFAULTS)
    out = {}
    for mm in (.3, .5, 1., 2.):
        res = dict(ramp=[], creep=[], ramp_peak=[], creep_peak=[])
        for i in range(A.n):
            g = np.random.default_rng(A.seed + i)
            sign = g.choice([-1., 1.])
            ask = sign * mm / LEVER * 1e-3
            # ramp
            p = Plant(g)
            p.q = g.normal(0, .005)
            q0 = p.q
            p.t_cmd = q0
            ramp_step(p, q0 + ask, g)
            res['ramp'].append((p.q - q0 - ask) * LEVER * 1e3)
            # creep
            g2 = np.random.default_rng(A.seed + i)
            p = Plant(g2)
            p.q = g2.normal(0, .005)
            q0 = p.q
            ch = fake_chain()
            ch.q_b = torch.tensor([[0., p.q, 0., 0., 0., 0.], [0.] * 6])
            cr = Creep(ch, torch.zeros(6), c, np.zeros(6))
            creep_step(p, q0 + ask, c, cr)
            res['creep'].append((p.q - q0 - ask) * LEVER * 1e3)
        out[f'{mm} mm'] = {k: dict(n=len(v), abs_p50=float(np.percentile(np.abs(v), 50)), abs_p90=float(np.percentile(np.abs(v), 90)), abs_max=float(np.abs(v).max())) for k, v in res.items() if v}
    print(json.dumps(out, indent=1))
    if A.json:
        Path(A.json).write_text(json.dumps(out, indent=1))
    worst = max(v['creep']['abs_p90'] for k, v in out.items() if k != '0.3 mm')       # 0.3 mm is the stage's move tolerance: it asks no such step alone
    print('creep p90 final error, steps of 0.5 mm and more (mm):', round(worst, 2), 'OK' if worst < 1.0 else 'FAILED: above 1 mm')
    return worst < 1.0


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
