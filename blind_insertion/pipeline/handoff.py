"""Task H pick -> insert switching logic shared by the sim pipeline (pipeline/pick_insert.py) and the
real-arm runner (deploy/runner.py), so the two cannot drift: ACT temporal ensembling, the handoff
gate, and the release trigger (the real arm has no ground-truth success).

All inputs are batched tensors (N worlds; the real runner uses N = 1) in the model joint frame
(encoder reading + joint offsets) and the right arm's base frame. Only joint encoders, the finger
reading and forward kinematics are used, so everything runs on the real arm unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from blind_insertion.scenes import pc_bench as S


def nominal_preinsert(port_xy, body_z, gripper):
    """Nominal tcp position at the pre-insert pose (m) for the configured PC position: the tcp sits
    `body_z` + 6 mm in front of the port along -x, at port height + the grasp height."""
    return (port_xy[0] - body_z - 0.006, port_xy[1], S.PORT_HEIGHT + float(np.mean(S.GRASP_Y[gripper])))


class TemporalEnsemble:
    """ACT temporal ensembling: push the chunk predicted this step, get the action for now = the
    average of every stored chunk's entry for now, weights exp(-k * age)."""

    def __init__(self, k=0.01):
        self.k = k
        self.buf = None

    def reset(self):
        self.buf = None

    def __call__(self, c):
        N, C, D = c.shape
        if self.buf is None:
            self.buf = torch.zeros(N, C, C, D, device=c.device)    # [age, offset] predictions
            self.valid = torch.zeros(C, dtype=torch.bool, device=c.device)
        self.buf = torch.roll(self.buf, 1, dims=1)
        self.buf[:, 0] = c
        self.valid = torch.roll(self.valid, 1)
        self.valid[0] = True
        ages = torch.arange(C, device=c.device)
        preds = self.buf[:, ages, ages]                             # chunk of age i, entry i = now
        w = torch.exp(-self.k * ages.float()) * self.valid
        return (preds * w[None, :, None]).sum(1) / w.sum()


class HandoffGate:
    """When control passes from ACT to the insertion policy.

    done path: ACT's done output > 0.5 for 5 steps in a row.
    settle path (settle=True): the arm has rested `rest_s` (all |qd| < rest_qd rad/s) with the gripper
    closed on the reader (finger opening > min_finger), the tcp within `near` of the nominal pre-insert
    pose `p_nom` and within `near_z` of its height (the grasp spot on the table is within `near` of
    it: without the height check the gate fired at the grasp, FINDINGS 2026-10-27), for when ACT does
    not raise done.
    grasp gate (gate=True): no handoff with the gripper open or closed on nothing (fingers ~19 mm on
    the reader, ~-6 mm empty); ACT keeps control and can regrasp. The fingers pass through the band
    while closing, so the grasp must hold for `grasp_s`."""

    def __init__(self, p_nom, control_dt, settle=True, rest_qd=0.03, rest_s=0.3, near=0.06, near_z=0.010,
                 min_finger=0.005, gate=True, grasp_s=0.5):
        self.p_nom, self.dt = p_nom, control_dt
        self.settle, self.rest_qd, self.rest_s, self.near, self.near_z = settle, rest_qd, rest_s, near, near_z
        self.min_finger, self.gate, self.grasp_s = min_finger, gate, grasp_s

    def reset(self, n, device):
        self.p = torch.as_tensor(self.p_nom, dtype=torch.float32, device=device)
        self.done_run = torch.zeros(n, dtype=torch.long, device=device)
        self.rest_run = torch.zeros(n, dtype=torch.long, device=device)
        self.grasp_run = torch.zeros(n, dtype=torch.long, device=device)

    def __call__(self, done_out, closed, finger, qd, tcp):
        """done_out (N,) ACT done output, closed (N,) gripper commanded closed, finger (N,) mean finger
        opening (m), qd (N, 6) joint velocities, tcp (N, 3) tcp position. Returns (N,) bool."""
        self.done_run = torch.where(done_out > 0.5, self.done_run + 1, torch.zeros_like(self.done_run))
        grasped = closed & (finger > self.min_finger)
        self.grasp_run = torch.where(grasped, self.grasp_run + 1, torch.zeros_like(self.grasp_run))
        held = self.grasp_run >= int(self.grasp_s / self.dt)
        done = (self.done_run >= 5) & (held if self.gate else True)
        if self.settle:
            rest = ((qd.abs().max(1).values < self.rest_qd) & closed & (finger > self.min_finger)
                    & ((tcp - self.p).norm(dim=1) < self.near) & ((tcp[:, 2] - self.p[2]).abs() < self.near_z))
            self.rest_run = torch.where(rest, self.rest_run + 1, torch.zeros_like(self.rest_run))
            done |= (self.rest_run >= int(self.rest_s / self.dt)) & (held if self.gate else True)
        return done


class ReleaseTrigger:
    """When to release on the real arm, which has no ground-truth success: `after_s` after the
    handoff, or earlier once the tcp has travelled `max_travel` along the insertion axis (+x).

    No proprioceptive seat signal works for this policy (FINDINGS 2026-10-28): it never saw the
    states after success (training episodes end there) and keeps pushing, so the hand slides along
    the reader instead of stalling, before and after the plug seats (tcp travel at seating 11-23 mm).
    Seating takes 1.9 s median, 3.0 s p90; at 5 s as many plugs are seated as ever get seated, while
    the hand has slid 23 mm median (42 mm p90; at 10 s 55 / 189 mm, toward the PC)."""

    def __init__(self, control_dt, after_s=5.0, max_travel=0.06):
        self.dt, self.after_s, self.max_travel = control_dt, after_s, max_travel

    def reset(self, n, device):
        self.x0 = torch.zeros(n, device=device)
        self.t = torch.zeros(n, dtype=torch.long, device=device)

    def start(self, ids, tcp):
        self.x0[ids], self.t[ids] = tcp[ids, 0], 0

    def __call__(self, active, tcp):
        """active (N,) worlds inserting, tcp (N, 3), after each insertion step. Returns (N,) bool."""
        self.t += active.long()
        return active & ((self.t >= int(round(self.after_s / self.dt))) | (tcp[:, 0] - self.x0 >= self.max_travel))
