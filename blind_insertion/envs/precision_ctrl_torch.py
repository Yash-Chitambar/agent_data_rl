"""Batched torch version of blind_insertion/envs/precision_ctrl.PrecisionLaw for the sims (n worlds x 6 joints).

Same law and parameter names (precision_ctrl.DEFAULTS / params): integral (dead band, freeze, cap, pause while the target
moves), settle lock, friction feed-forward. Not here: the vibration watch / fallback (off in every real config so far).
Per-world time: each world keeps its own clock `t` (s), advanced by `dt` per call; reset(ids) restarts it.
Check against the NumPy law: scripts/eval/align_ctrl_parity.py (one world, random targets, max torque difference).
"""
from __future__ import annotations

import torch

from blind_insertion.envs.precision_ctrl import params


class PrecisionLawTorch:
    def __init__(self, n, p=None, device='cpu', dtype=torch.float32):
        self.p = params(p, 6)
        if self.p.get('vibration_watch'):
            raise ValueError('vibration_watch is not implemented in the batched law')
        P, dev = self.p, device
        f = lambda x: torch.as_tensor(x, dtype=dtype, device=dev)  # noqa: E731
        self.kp, self.i_max, self.fric = f(P['kp']), f(P['i_max']), f(P['friction'])
        self.n, self.dev = n, dev
        self.I = torch.zeros(n, 6, device=dev, dtype=dtype)
        self.latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.band_since = torch.full((n,), float('nan'), device=dev, dtype=torch.float64)
        self.fast_until = torch.full((n,), -1e9, device=dev, dtype=torch.float64)
        self.still_since = torch.zeros(n, device=dev, dtype=torch.float64)
        self.last_target = torch.zeros(n, 6, device=dev, dtype=dtype)
        self.has_last = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t = torch.zeros(n, device=dev, dtype=torch.float64)
        self.frozen = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tau = torch.zeros(n, 6, device=dev, dtype=dtype)

    def reset(self, ids):
        self.I[ids] = 0
        self.latched[ids] = False
        self.band_since[ids] = float('nan')
        self.fast_until[ids] = -1e9
        self.still_since[ids] = 0
        self.has_last[ids] = False
        self.t[ids] = 0
        self.tau[ids] = 0

    def step(self, q, target, dt, t_abs=None):
        """q, target (n, 6) rad, encoder frame; dt (s) since the last call (t_abs: the clock instead, for parity tests).
        Returns tau (n, 6) N m."""
        P = self.p
        self.t = self.t + dt if t_abs is None else torch.full_like(self.t, float(t_abs))
        t = self.t
        e = target - q
        dt_nom = 1.0 / P['hz']
        moved = torch.where(self.has_last, (target - self.last_target).abs().amax(1).double(), torch.zeros_like(t))
        fast = self.has_last & (moved / dt_nom > P['i_move_rad_s'])     # as the NumPy law (call it at hz = 1 / dt)
        self.fast_until = torch.where(fast, t + P['i_resume_s'], self.fast_until)
        unlock = (t < self.fast_until) | (moved > P['target_still_eps'])
        emax = e.abs().amax(1).double()
        if P['settle_lock']:
            lat = self.latched & ~unlock
            lat_off = lat & (emax > P['unlatch_band'])
            in_band = ~unlock & ~lat & (emax < P['latch_band'])
            bs = torch.where(in_band, torch.where(torch.isnan(self.band_since), t, self.band_since), torch.full_like(t, float('nan')))
            new_lat = in_band & (t - bs >= P['latch_s'])
            self.latched = (lat & ~lat_off) | new_lat
            self.band_since = torch.where(lat & ~lat_off, self.band_since, bs)
            self.band_since = torch.where(lat_off | unlock, torch.full_like(t, float('nan')), self.band_since)
        else:
            self.latched = torch.zeros_like(self.latched)
        if P['integral']:
            act = ~self.frozen & (t >= self.fast_until) & ~self.latched
            live = (e.abs() > P['deadband']) & (e.abs() < P['freeze']) & act[:, None]
            self.I = torch.maximum(torch.minimum(self.I + torch.where(live, self.kp / P['i_tau'] * e * dt, torch.zeros_like(e)), self.i_max), -self.i_max)
            tau = self.I.clone()
        else:
            tau = torch.zeros_like(e)
        newly = ~self.has_last | (moved > P['target_still_eps'])
        self.still_since = torch.where(newly, t, self.still_since)
        self.last_target = target.clone()
        self.has_last = torch.ones_like(self.has_last)
        if P['friction_ff']:
            on = (t - self.still_since > P['fric_still_s'])[:, None]
            tau = tau + torch.where(on, P['fric_gain'] * self.fric * (e / P['fric_eps']).clamp(-1, 1), torch.zeros_like(e))
        self.tau = tau
        return tau
