"""Real-arm side of the precision controller (blind_insertion/envs/precision_ctrl.py) and the fitted gravity model.

ArmController runs the law in its own thread at `hz` for one i2rt arm: it reads the measured joints and velocities,
takes the target from a callable (SafeArm.last_safe: what the safety layer last sent), and writes the law's torque
into i2rt's command `torques` field under its command lock, the same path as the Coulomb feed-forward in
SafeArm.command. i2rt adds that field to PD + gravity + its own Coulomb term. command_joint_pos replaces the command
and zeroes the field: SafeArm re-applies the latest torque right after every send (SafeArm.ctrl), and this thread
rewrites it every cycle. No i2rt patch is needed.

If the loop raises, it zeroes its torque, stops and keeps the error (status()['error']); the arm stays on PD.
`stop()` zeroes the torque too: the integral's holding torque goes, so the arm may settle a little on PD alone.

Gravity: gravity_fit(path or dict) reads a fitted payload model, either this repo's format or the athena3 cell's
fit.json ({"deploy": {"ee_mass", "gravity_comp_factor", "ee_inertia" (ipos first)}}). safety.connect(...,
gravity=...) passes ee_mass and the factors to i2rt and sets the payload's centre of mass in i2rt's gravity model
(body "gripper"; i2rt's own ee_inertia argument writes an attribute MuJoCo rejects).

    .venv/bin/python -m pytest tests/test_precision_ctrl.py     # offline checks, no hardware
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def gravity_fit(src):
    """{'ee_mass': kg, 'gravity_comp_factor': (6,), 'ee_ipos': (3,) m or None, 'source': str} from a path or dict."""
    if src is None:
        return None
    d = src if isinstance(src, dict) else json.load(open(REPO / src if not Path(src).is_absolute() else src))
    name = str(src) if not isinstance(src, dict) else d.get('source', 'dict')
    if 'deploy' in d:                       # athena3 fit.json
        d = d['deploy']
        ipos = d.get('ee_inertia')
        ipos = None if ipos is None else list(ipos)[:3]
    else:
        ipos = d.get('ee_ipos')
    out = dict(ee_mass=float(d['ee_mass']), gravity_comp_factor=np.asarray(d['gravity_comp_factor'], float).reshape(6),
               ee_ipos=None if ipos is None else np.asarray(ipos, float).reshape(3), source=name)
    if not 0.0 < out['ee_mass'] < 3.0 or np.any(out['gravity_comp_factor'] < 0.3) or np.any(out['gravity_comp_factor'] > 2.5):
        raise ValueError(f'gravity fit {name} out of range: {out}')
    if out['ee_ipos'] is not None and np.max(np.abs(out['ee_ipos'])) > 0.15:
        raise ValueError(f'gravity fit {name}: centre of mass {out["ee_ipos"]} more than 15 cm from the hand frame')
    return out


def set_payload_com(robot, ipos, body='gripper'):
    """Write the payload's centre of mass into i2rt's gravity model. Returns the previous value."""
    import mujoco
    km = robot.kdl.model
    b = mujoco.mj_name2id(km, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0:
        raise KeyError(f'no body {body!r} in the i2rt gravity model')
    old = km.body_ipos[b].copy()
    km.body_ipos[b] = np.asarray(ipos, float)
    return old


class ArmController:
    """The precision law on one real arm, in its own thread."""

    def __init__(self, robot, target_fn, p=None, name='arm', fallback_gains=None, i0=None):
        """robot: i2rt robot (7 values, motor frame). target_fn() -> the 7-value target last sent (motor frame).
        p: precision_ctrl params. fallback_gains: (kp6, kd6) to set on a vibration fallback, or None (the law then
        only turns its friction feed-forward off, as on athena3 where the fallback gains equal the normal ones)."""
        from blind_insertion.envs.precision_ctrl import PrecisionLaw
        self.robot, self.target_fn, self.name = robot, target_fn, name
        self.law = PrecisionLaw(p, n=6, i0=i0)
        self.hz = float(self.law.p['hz'])
        self.fallback_gains = fallback_gains
        self._gains = (robot._kp.copy(), robot._kd.copy())
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = None
        self.error = None
        self.events = []                # (t, event, detail)
        self.n_cycles, self.late_max = 0, 0.0

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive() and self.error is None

    def tau(self):
        with self._lock:
            return self.law.tau.copy()

    def write(self, tau=None):
        """Put the torque in i2rt's current command (arm joints only; the gripper's entry is left alone)."""
        tau = self.tau() if tau is None else np.asarray(tau, float)
        r = self.robot
        with r._command_lock:
            r._commands.torques[:6] = tau
        return tau

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._loop, name=f'precision-{self.name}', daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self._lock:
            self.law.tau = np.zeros(6)
        try:
            self.write(np.zeros(6))
        except Exception:  # noqa: BLE001
            pass
        self._restore_gains()

    def freeze(self, on=True):
        with self._lock:
            self.law.frozen = bool(on)

    def status(self):
        with self._lock:
            s = self.law.status()
        s.update(error=self.error, cycles=self.n_cycles, late_max_ms=round(self.late_max * 1e3, 2))
        return s

    def _read(self):
        r = self.robot
        with r._state_lock:
            q, qd = r._joint_state.pos.copy(), r._joint_state.vel.copy()
        return np.asarray(q, float), np.asarray(qd, float)

    def _restore_gains(self):
        if self.fallback_gains is not None:
            try:
                self.robot.update_kp_kd(*self._gains)
            except Exception:  # noqa: BLE001
                pass

    def _loop(self):
        dt = 1.0 / self.hz
        t_next = time.monotonic()
        try:
            while not self._stop.is_set():
                t = time.monotonic()
                q, qd = self._read()
                target = np.asarray(self.target_fn(), float)
                if not (np.all(np.isfinite(q[:6])) and np.all(np.isfinite(target[:6]))):
                    raise RuntimeError(f'non-finite state or target: q {q[:6]}, target {target[:6]}')
                with self._lock:
                    out = self.law.step(t, q[:6], target[:6], qd[:6])
                    tau = out['tau'].copy()
                self.write(tau)
                if out['event'] is not None:
                    self.events.append((t, out['event'], self.law.fallback))
                    if self.fallback_gains is not None:
                        if out['event'] == 'fallback':
                            kp, kd = self._gains[0].copy(), self._gains[1].copy()
                            kp[:6], kd[:6] = self.fallback_gains
                            self.robot.update_kp_kd(kp, kd)
                        else:
                            self.robot.update_kp_kd(*self._gains)
                    print(f'[precision {self.name}] {out["event"]}: {self.law.fallback or ""}', flush=True)
                self.n_cycles += 1
                t_next += dt
                late = time.monotonic() - t_next
                self.late_max = max(self.late_max, late)
                if late > 5 * dt:
                    t_next = time.monotonic()
                else:
                    time.sleep(max(0.0, t_next - time.monotonic()))
        except Exception as exc:  # noqa: BLE001
            self.error = f'{type(exc).__name__}: {exc}'
            with self._lock:
                self.law.tau = np.zeros(6)
            try:
                self.write(np.zeros(6))
            except Exception:  # noqa: BLE001
                pass
            print(f'[precision {self.name}] loop stopped, torque zeroed: {self.error}', flush=True)
