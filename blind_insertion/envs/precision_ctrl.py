"""Precision joint controller for the YAM: integral torque with settle lock, friction feed-forward, vibration watch.

The law of the athena3 cell's controller (github.com/XinyueZhuXY/yam-control d592e8e, yam_controller.py, itself
real_yam's yam_twin/real_env.py of 2026-11-02), as a pure step function with no I/O, so the real arm
(scripts/real/precision_arm.py) and a sim can run the same code. The PD springs and the gravity model stay in i2rt;
this law only adds a feed-forward torque on top of them (i2rt: motor torque = PD + command torques + gravity +
its own Coulomb term).

Per call (`step`), per joint, e = target - measured (rad, encoder frame):

  1. Integral torque I += kp / i_tau * e * dt, only where deadband < |e| < freeze, capped at +-i_max. It pauses
     while the target moves faster than i_move_rad_s and resumes i_resume_s after it slows (no wind-up on tracking
     lag). It is kept from move to move.
  2. Settle lock: once every joint stays within latch_band for latch_s with the target still, the integral stops
     changing (its torque is kept). A target change or an error beyond unlatch_band unlocks it.
  3. Friction feed-forward: fric_gain * friction * clip(e / fric_eps, -1, 1), only after the target has held still
     for fric_still_s. Pushes toward the target with the static friction, fades out at the target.
  4. Vibration watch: with the target still for vib_still_s and a full window of vib_win_s, if a joint's velocity
     rms exceeds vib_rms with at least vib_swings sign changes, the law enters fallback: friction feed-forward off
     (the integral stays on), and the caller may set fallback gains. A target farther than recover_rad from the
     fallback target ends it, until max_fallbacks have happened.

Every feature has its own switch (`integral`, `settle_lock`, `friction_ff`, `vibration_watch`), all off by default.
The numbers in DEFAULTS are the athena3 cell's; the per-arm values (friction, gains) must come from this rig's own
system identification. `friction` defaults to zeros: no feed-forward until it is measured.
"""
from __future__ import annotations

from collections import deque

import numpy as np

DEFAULTS = dict(
    integral=False, settle_lock=False, friction_ff=False, vibration_watch=False,
    hz=200.0,
    kp=[80.0, 80.0, 80.0, 40.0, 10.0, 10.0],      # the PD springs the arm runs with (integral gain = kp / i_tau)
    i_tau=0.12,                                    # s
    i_max=[3.0, 3.0, 3.0, 1.2, 0.8, 0.8],          # N m cap per joint
    deadband=0.0001,                               # rad: no integration inside
    freeze=0.100,                                  # rad: no integration beyond (contact, fault)
    i_move_rad_s=0.05,                             # integral pauses while the target moves faster than this
    i_resume_s=0.2,                                # ...and resumes this long after it slows
    latch_band=0.0005, latch_s=0.3, unlatch_band=0.0015,
    friction=[0.0] * 6,                            # N m static friction per joint (from system ID)
    fric_gain=0.7, fric_eps=0.0005, fric_still_s=0.1,
    vib_rms=0.12, vib_swings=6, vib_win_s=0.5, vib_still_s=1.0, vib_min_vel=0.02,
    recover_rad=0.002, max_fallbacks=3,
    target_still_eps=1e-6,                         # rad: a smaller target change counts as still
)
ARRAYS = ('kp', 'i_max', 'friction')


def params(p=None, n=6):
    """DEFAULTS merged with p; per-joint entries as arrays of length n."""
    out = {**DEFAULTS, **(p or {})}
    unknown = set(p or {}) - set(DEFAULTS)
    if unknown:
        raise KeyError(f'unknown precision controller keys: {sorted(unknown)}')
    for k in ARRAYS:
        out[k] = np.broadcast_to(np.asarray(out[k], float), (n,)).copy()
    return out


class PrecisionLaw:
    """State of one arm's law. step() once per control cycle (nominally 1 / hz)."""

    def __init__(self, p=None, n=6, i0=None):
        self.n = n
        self.p = params(p, n)
        self.I = np.zeros(n) if i0 is None else np.clip(np.asarray(i0, float), -self.p['i_max'], self.p['i_max'])
        self.frozen = False             # set by the caller: hold the integral (e.g. a contact stage)
        self.latched = False
        self.band_since = None
        self.fast_until = -np.inf
        self.still_since = None
        self.last_target = None
        self.t_prev = None
        self.fallback = None            # None, or the reason string while in fallback
        self.fallback_target = None
        self.fallbacks = 0
        win = max(2, int(round(self.p['vib_win_s'] * self.p['hz'])))
        self.vel, self.tg = deque(maxlen=win), deque(maxlen=win)
        self.tau = np.zeros(n)

    def reset_integral(self, i0=None):
        self.I = np.zeros(self.n) if i0 is None else np.clip(np.asarray(i0, float), -self.p['i_max'], self.p['i_max'])
        self.latched, self.band_since = False, None

    def step(self, t, q, target, qd=None):
        """t (s, monotonic), q / target (n,) rad, qd (n,) rad/s or None. Returns dict(tau (n,), event or None)."""
        P = self.p
        q, target = np.asarray(q, float)[:self.n], np.asarray(target, float)[:self.n]
        e = target - q
        dt_nom = 1.0 / P['hz']
        dt = dt_nom if self.t_prev is None else float(np.clip(t - self.t_prev, 0.0, 4 * dt_nom))
        self.t_prev = t
        event = None
        moved = 0.0 if self.last_target is None else float(np.max(np.abs(target - self.last_target)))

        if (self.fallback is not None and self.fallback_target is not None and self.fallbacks < P['max_fallbacks']
                and np.max(np.abs(target - self.fallback_target)) > P['recover_rad']):
            self.fallback, self.fallback_target = None, None
            self.vel.clear()
            self.tg.clear()
            event = 'recover'

        if self.last_target is not None and moved / dt_nom > P['i_move_rad_s']:
            self.fast_until = t + P['i_resume_s']

        if t < self.fast_until or moved > P['target_still_eps']:
            self.latched, self.band_since = False, None
        elif P['settle_lock']:
            emax = float(np.max(np.abs(e)))
            if self.latched and emax > P['unlatch_band']:
                self.latched, self.band_since = False, None
            elif not self.latched:
                if emax < P['latch_band']:
                    self.band_since = t if self.band_since is None else self.band_since
                    self.latched = t - self.band_since >= P['latch_s']
                else:
                    self.band_since = None
        else:
            self.latched = False

        if P['integral'] and not self.frozen and t >= self.fast_until and not self.latched:
            live = (np.abs(e) > P['deadband']) & (np.abs(e) < P['freeze'])
            self.I = np.clip(self.I + np.where(live, P['kp'] / P['i_tau'] * e * dt, 0.0), -P['i_max'], P['i_max'])

        tau = self.I.copy() if P['integral'] else np.zeros(self.n)
        if self.last_target is None or moved > P['target_still_eps']:
            self.still_since = t
        self.last_target = target.copy()
        if P['friction_ff'] and self.fallback is None and t - self.still_since > P['fric_still_s']:
            tau += P['fric_gain'] * P['friction'] * np.clip(e / P['fric_eps'], -1.0, 1.0)

        if P['vibration_watch'] and qd is not None:
            self.vel.append(np.asarray(qd, float)[:self.n].copy())
            self.tg.append(target.copy())
            if (self.fallback is None and len(self.vel) == self.vel.maxlen
                    and np.max(np.abs(self.tg[-1] - self.tg[0])) < 1e-4 and t - self.still_since > P['vib_still_s']):
                V = np.asarray(self.vel)
                rms = np.sqrt(np.mean(V ** 2, axis=0))
                j = int(np.argmax(rms))
                vj = V[:, j][np.abs(V[:, j]) > P['vib_min_vel']]
                swings = int(np.sum(np.diff(np.sign(vj)) != 0)) if len(vj) > 1 else 0
                if rms.max() > P['vib_rms'] and swings >= P['vib_swings']:
                    self.fallback = f'vibration joint {j + 1}, rms {np.round(rms, 3).tolist()} rad/s, {swings} swings'
                    self.fallback_target = target.copy()
                    self.fallbacks += 1
                    event = 'fallback'
        self.tau = tau
        return dict(tau=tau, event=event)

    def status(self):
        return dict(I=self.I.copy(), tau=self.tau.copy(), latched=bool(self.latched), fallback=self.fallback,
                    fallbacks=self.fallbacks, frozen=bool(self.frozen))
