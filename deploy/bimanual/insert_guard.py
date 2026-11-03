"""Run-time checks of the insertion (config insert.seat2, insert.guard; docs/real/ALIGN_STAGE.md). Batched tensors, so the sim evals run the same code as the
robot (n = 1 there). All positions: plug tip from believed FK, port axes (x across, y up, z out of the port), relative to the tip at the hand-over.

SeatDetector2: seated = the plug went far enough past the MEASURED gap, the plug stays near the port in across and up, and it is at rest. A plug pressing on the
case (real runs 2026-11-02 A and A2: read as about 6 mm deep by the old detector) has advanced about the gap only.

InsertGuard: a bound around the hand-over pose in across and up.
  * measured tip at the bound: the command loses its outward lead (the target stops pulling further out; inside the bound nothing changes);
  * measured tip at bound + hard margin: stop;
  * press abort: no axial progress for stall_s while the target still pushes, and no seat: stop.
A stop lifts the plug straight out with closed jaws (deploy/bimanual/insert.py): the reader stays in the hand.
"""
from __future__ import annotations

import torch


def fk_tip(chain, q, s):
    """Tip position, TCP position, rotation of chain at q (n, 6); tip = p - R s."""
    p, R = chain.fk(q)
    return p - (R @ s.to(R.dtype)[..., None])[..., 0], p, R


class SeatDetector2:
    def __init__(self, n, device, gap0_mm, lat0_mm, seat_depth_mm=5.8, margin_mm=1.0, gate_mm=(3.0, 6.0), hold_s=.2, speed_mm_s=5.0, still_mm=1.0, dt=.02):
        self.dev = device
        self.gap0 = torch.as_tensor(gap0_mm, dtype=torch.float32, device=device).expand(n).clone()
        self.lat0 = torch.as_tensor(lat0_mm, dtype=torch.float32, device=device).expand(n, 2).clone()
        self.need, self.gate = float(seat_depth_mm) - float(margin_mm), torch.as_tensor(gate_mm, dtype=torch.float32, device=device)
        self.vmax, self.still, self.dt = float(speed_mm_s), float(still_mm), float(dt)
        self.n_hold = max(1, int(round(hold_s / dt)))
        self.run = torch.zeros(n, dtype=torch.long, device=device)
        self.seated = torch.zeros(n, dtype=torch.bool, device=device)
        self.win = []

    def reset(self, ids, gap0_mm=None, lat0_mm=None):
        self.run[ids], self.seated[ids] = 0, False
        if gap0_mm is not None:
            self.gap0[ids] = gap0_mm
        if lat0_mm is not None:
            self.lat0[ids] = lat0_mm
        if self.win:
            for w in self.win:
                w[ids] = float('nan')

    def update(self, rel_mm):
        """rel_mm (n, 3): tip against the hand-over. Returns dict depth, lat (n, 2), speed, lat_range, deep, gate_ok, seated."""
        depth = -rel_mm[:, 2] - self.gap0
        lat = self.lat0 + rel_mm[:, :2]
        self.win = (self.win + [torch.cat([depth[:, None], lat], 1)])[-max(self.n_hold + 1, 6):]
        W = torch.stack(self.win)                      # (w, n, 3); NaN just after a reset
        k = min(6, len(self.win))
        v = (W[-1, :, 0] - W[-k, :, 0]) / max((k - 1) * self.dt, self.dt)
        rng = (W[..., 1:].amax(0) - W[..., 1:].amin(0)).amax(1)
        deep = depth >= self.need
        gate_ok = (lat.abs() <= self.gate).all(1)
        self.deep_ok = deep & gate_ok
        ok = deep & gate_ok & (torch.nan_to_num(v, 0.).abs() <= self.vmax) & (torch.nan_to_num(rng, 0.) <= self.still)
        self.run = torch.where(ok, self.run + 1, torch.zeros_like(self.run))
        self.seated = self.seated | (self.run >= self.n_hold)
        return dict(depth=depth, lat=lat, speed=v, lat_range=rng, deep=deep, gate_ok=gate_ok, seated=self.seated)


class InsertGuard:
    def __init__(self, chain, tip_s, port_R, n, device, bound_mm=(5.0, 5.0), hard_mm=3.0, stall_s=3.0, stall_eps_mm=.3, lead_min_mm=3.0, t_min_s=1.0, dt=.02, damping=1e-4):
        self.ch, self.s, self.dev, self.dt = chain, tip_s.to(device), device, float(dt)
        self.eR = port_R.to(device).float().reshape(-1, 3, 3).expand(n, 3, 3).clone()
        self.B = torch.as_tensor(bound_mm, dtype=torch.float32, device=device)
        self.hard, self.eps, self.lead_min = float(hard_mm), float(stall_eps_mm), float(lead_min_mm)
        self.n_stall, self.n_min = int(round(stall_s / dt)), int(round(t_min_s / dt))
        self.damping = float(damping)
        self.tip0 = torch.zeros(n, 3, device=device)
        self.best = torch.zeros(n, device=device)
        self.t_best = torch.zeros(n, dtype=torch.long, device=device)
        self.t = torch.zeros(n, dtype=torch.long, device=device)
        self.clamped = torch.zeros(n, dtype=torch.long, device=device)
        self.stop = torch.zeros(n, dtype=torch.long, device=device)      # 0 none, 1 bound, 2 press

    def reset(self, ids, q_b, port_R=None):
        if port_R is not None:
            self.eR[ids] = port_R.to(self.dev).float()
        self.tip0[ids] = fk_tip(self.ch, q_b[ids], self.s)[0]
        self.best[ids], self.t_best[ids], self.t[ids], self.clamped[ids], self.stop[ids] = 0, 0, 0, 0, 0

    def rel(self, q):
        """Tip against the hand-over, port axes, mm (n, 3)."""
        tip = fk_tip(self.ch, q, self.s)[0]
        return ((self.eR.transpose(1, 2) @ (tip - self.tip0)[..., None])[..., 0]) * 1e3

    def limit(self, q_b, q_t, seated=None):
        """Measured q_b, proposed target q_t (n, 6) -> (limited target, info). Sets self.stop."""
        rm, rt = self.rel(q_b), self.rel(q_t)
        lead = rt - rm
        over = rm[:, :2].abs() >= self.B
        outward = (lead[:, :2] * torch.sign(rm[:, :2])) > 0
        cut = over & outward
        out = q_t
        if cut.any():
            d = torch.zeros_like(rm)
            d[:, :2] = torch.where(cut, -lead[:, :2], torch.zeros_like(lead[:, :2]))
            dw = (self.eR @ d[..., None])[..., 0] * 1e-3
            ids = cut.any(1).nonzero()[:, 0]
            _, _, J = self.ch.fk(q_t[ids], with_jac=True)
            tw = torch.cat([dw[ids], torch.zeros(len(ids), 3, device=self.dev)], 1)
            JJt = J @ J.transpose(1, 2) + self.damping * torch.eye(6, device=self.dev, dtype=J.dtype)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, tw[..., None].to(J.dtype)))[..., 0]
            out = q_t.clone()
            out[ids] = q_t[ids] + dq.to(q_t.dtype)
            self.clamped[ids] += 1
        self.t = self.t + 1
        hard = (rm[:, :2].abs() >= self.B + self.hard).any(1)
        adv = -rm[:, 2]
        better = adv > self.best + self.eps
        self.best = torch.where(better, adv, self.best)
        self.t_best = torch.where(better, self.t, self.t_best)
        stalled = (self.t - self.t_best >= self.n_stall) & (self.t >= self.n_min) & (-lead[:, 2] >= self.lead_min)
        if seated is not None:
            stalled = stalled & ~seated
            hard = hard & ~seated
        self.stop = torch.where(self.stop == 0, torch.where(hard, torch.ones_like(self.stop), torch.where(stalled, 2 * torch.ones_like(self.stop), self.stop)), self.stop)
        return out, dict(rel_m=rm, rel_t=rt, cut=cut, stop=self.stop)
