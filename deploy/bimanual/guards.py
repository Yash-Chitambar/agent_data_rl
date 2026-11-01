"""Run-time guards for the bimanual runner, on top of scripts/real/safety.SafeArm (joint box, table clearance,
step cap, deviation limit, checked on every command and measured pose):

  * software stop: scripts/real/estop.py's flag file, or Enter typed in the runner's own terminal while a stage
    moves (StopKey: one stdin reader that also answers the stage prompts);
  * PD torque estimate per joint, tau = kp (target - q) + feed-forward (the force the controller asks for; gravity
    is compensated by the driver separately): above tau_peak for tau_peak_s -> stop; above tau_cont for
    tau_cont_s in total -> stop (left joint 2 at the 25 N press sits near its ~9 N m rating);
  * time under the left press (hold_max_s) -> stop (the runner then unloads the press);
  * joint speed (qd_max) and the TCP box of both arms (targets, right-base frame) -> stop;
  * --max-steps.

A stop raises GuardStop; the runner then holds both arms at their measured pose (zero PD error: no press force,
no push) and asks the operator what to do.
"""
from __future__ import annotations

import queue
import sys
import threading

import numpy as np


class GuardStop(RuntimeError):
    pass


class StopKey:
    """One stdin reader thread. While a prompt waits, a typed line answers it; otherwise any line (Enter) is a stop."""

    def __init__(self, enabled=True):
        self.stop = threading.Event()
        self.answers = queue.Queue()
        self.prompting = threading.Event()
        self.enabled = enabled and sys.stdin is not None and sys.stdin.isatty()
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for line in sys.stdin:
            if self.prompting.is_set():
                self.answers.put(line.strip())
            else:
                self.stop.set()
                print('\n*** STOP (Enter in the runner terminal): holding both arms ***', flush=True)

    def ask(self, text, auto=None):
        """Prompt; returns the typed line ('' = Enter). auto: answer without asking (dry runs, --yes)."""
        if auto is not None:
            print(f'{text} [auto: {auto!r}]', flush=True)
            return auto
        print(text, end=' ', flush=True)
        if not self.enabled:
            return input()
        self.prompting.set()
        try:
            return self.answers.get()
        finally:
            self.prompting.clear()

    def clear(self):
        self.stop.clear()


class Guards:
    def __init__(self, cfg, kp, dt, max_steps=None, estop_fn=None, stop_key=None):
        s = cfg['safety']
        self.kp = np.asarray(kp, float)                         # (2, 6) right, left
        self.dt = dt
        self.tau_peak, self.tau_cont = np.asarray(s['tau_peak_nm'], float), np.asarray(s['tau_cont_nm'], float)
        self.n_peak = max(1, int(round(s['tau_peak_s'] / dt)))
        self.n_cont = int(round(s['tau_cont_s'] / dt))
        self.n_hold = int(round(s['hold_max_s'] / dt))
        self.qd_max = float(s['qd_max_rad_s'])
        self.box = s['tcp_box']
        self.table_z = float(s['table_z'])
        self.max_steps = max_steps
        self.estop_fn, self.stop_key = estop_fn, stop_key
        self.reset_counts()

    def reset_counts(self):
        self.peak = np.zeros((2, 6), int)
        self.cont = np.zeros((2, 6), int)
        self.hold_ticks = 0
        self.ticks = 0

    def tau(self, q_target, q, ff=None):
        """(2, 6) PD torque estimate; q_target and q in the same (believed) frame."""
        t = self.kp * (np.asarray(q_target, float) - np.asarray(q, float))
        if ff is not None:
            t = t + np.asarray(ff, float)
        return t

    def check_target(self, tcp_r, tcp_l):
        """TCP positions (3,) of the commanded poses, right-base frame."""
        for name, p in (('right', tcp_r), ('left', tcp_l)):
            x, y, z = (float(v) for v in p)
            zt = z - self.table_z
            b = self.box
            if not (b['x'][0] <= x <= b['x'][1] and b['y'][0] <= y <= b['y'][1] and b['z_above_table'][0] <= zt <= b['z_above_table'][1]):
                raise GuardStop(f'{name} TCP target ({x:.3f}, {y:.3f}, {zt:.3f} above the table) outside the box {b}')

    def check_state(self, q_target, q, qd, ff=None, pressing=False):
        """After each tick: stop key, torques, speed, hold time, step count. Returns the torque estimate."""
        self.ticks += 1
        if self.stop_key is not None and self.stop_key.stop.is_set():
            raise GuardStop('software stop (Enter in the runner terminal)')
        if self.estop_fn is not None and self.estop_fn():
            raise GuardStop('software stop (estop.py)')
        if self.max_steps is not None and self.ticks > self.max_steps:
            raise GuardStop(f'--max-steps {self.max_steps} reached')
        tau = self.tau(q_target, q, ff)
        a = np.abs(tau)
        self.peak = np.where(a > self.tau_peak, self.peak + 1, 0)
        self.cont = self.cont + (a > self.tau_cont)
        if (self.peak >= self.n_peak).any():
            s, j = np.argwhere(self.peak >= self.n_peak)[0]
            raise GuardStop(f'{("right", "left")[s]} joint {j + 1} torque estimate {tau[s, j]:+.1f} N m above {self.tau_peak[j]} for {self.n_peak} ticks')
        if (self.cont >= self.n_cont).any():
            s, j = np.argwhere(self.cont >= self.n_cont)[0]
            raise GuardStop(f'{("right", "left")[s]} joint {j + 1} above its continuous torque {self.tau_cont[j]} N m for {self.n_cont * self.dt:.0f} s in total')
        if np.abs(qd).max() > self.qd_max:
            s, j = np.unravel_index(np.abs(qd).argmax(), np.shape(qd))
            raise GuardStop(f'{("right", "left")[s]} joint {j + 1} speed {qd[s][j]:+.2f} rad/s > {self.qd_max}')
        if pressing:
            self.hold_ticks += 1
            if self.hold_ticks > self.n_hold:
                raise GuardStop(f'left press held longer than hold_max_s ({self.n_hold * self.dt:.0f} s)')
        return tau
