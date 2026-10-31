"""Bounded integral trim for the settle after a slow move_to; off unless move_to(settle_trim=True).

Why: the real joints stick (joint 3: ~1-2 N m static friction against kp 80, i.e. a 12-25 mrad dead
band), and the sim's PD action scheme has no integral, so a held target stays 20+ mrad away
(intermediate pose 2026-10-27: 20.2 mrad, constant over 1 s). The trim adds a slowly growing
offset to the held command until the joint breaks away and reaches its goal (stick-slip: 25.5 ->
14.8 -> 1.5 mrad on joint 3 within 5.3 s, data/task_h/intermediate_trim_001.json).

  * Rate-limited (`rate` rad/s per joint) and capped (`limit` rad): the command never leaves the
    goal by more than `limit`, and every command still goes through SafeArm's checks.
  * No integration inside `deadband` of the goal: once there, the offset is frozen (the first
    version wound up to its cap while the joint already sat 1.5 mrad from the goal).
  * Joint radians in the motor frame, never an encoder calibration. It lives only in the held
    command: the first ACT tick sends ACT's own target (last measured pose + capped step), so the
    policy starts as in the sim (command = pose, no integral).
"""
import numpy as np


class ApproachTrim:
    def __init__(self, n=6, ki=.5, limit=.05, rate=.01, deadband=.002):
        self.ki, self.limit, self.rate, self.deadband = ki, limit, rate, deadband
        self.bias = np.zeros(n)

    def step(self, error, dt, stopped=False):
        """error = goal - measured (rad, per joint) -> the offset to add to the held command."""
        error = np.asarray(error, float)
        if stopped or not np.all(np.isfinite(error)) or not 0 < dt <= .1:
            return self.bias.copy()
        error = np.where(np.abs(error) < self.deadband, 0.0, error)
        delta = np.clip(self.ki * error * dt, -self.rate * dt, self.rate * dt)
        self.bias = np.clip(self.bias + delta, -self.limit, self.limit)
        return self.bias.copy()


def settled(errors, tol):
    """True when every joint of the last len(errors) error vectors is within tol (rad)."""
    return len(errors) > 0 and bool(np.all(np.abs(np.asarray(errors)) < tol))
