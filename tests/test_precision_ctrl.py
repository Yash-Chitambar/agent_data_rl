"""Offline checks of the precision controller (blind_insertion/envs/precision_ctrl.py, scripts/real/precision_arm.py).

No hardware: the law runs on a one-joint plant with Coulomb stick-slip and a gravity-model error, and the real-arm
thread runs against a fake i2rt robot.
"""
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / 'scripts/real'))   # appended: its deploy.py must not shadow the deploy package

from blind_insertion.envs.precision_ctrl import DEFAULTS, PrecisionLaw, params  # noqa: E402

ALL_ON = dict(integral=True, settle_lock=True, friction_ff=True, vibration_watch=True)


def plant(law, target_fn, T=4.0, bias=-1.0, fs=1.2, kp=80.0, kd=5.0, J=0.02, q0=0.0, dt=1e-3, every=5):
    """One joint (joint 1 of law): J qdd = kp e - kd qd + bias + tau, Coulomb static friction fs (stick when the
    driving torque is below fs at rest). The law steps every `every` plant steps. Returns arrays t, q, target, I."""
    q, qd, tau = q0, 0.0, 0.0
    out = []
    for i in range(int(T / dt)):
        t = i * dt
        tgt = target_fn(t)
        if i % every == 0:
            r = law.step(t, np.r_[q, np.zeros(5)], np.r_[tgt, np.zeros(5)], np.r_[qd, np.zeros(5)])
            tau = r['tau'][0]
        drive = kp * (tgt - q) - kd * qd + bias + tau
        if abs(qd) < 1e-4 and abs(drive) <= fs:
            qd = 0.0
        else:
            f = drive - fs * np.sign(qd if abs(qd) >= 1e-4 else drive)
            qd_new = qd + f / J * dt
            qd = 0.0 if qd != 0 and np.sign(qd_new) != np.sign(qd) and abs(drive) <= fs else qd_new
        q += qd * dt
        out.append((t, q, tgt, law.I[0]))
    return np.array(out).T


def test_defaults_off_give_zero_torque():
    law = PrecisionLaw()
    for k in range(50):
        r = law.step(k * .005, np.full(6, .01), np.zeros(6), np.zeros(6))
    assert np.all(r['tau'] == 0) and r['event'] is None


def test_unknown_key_refused():
    with pytest.raises(KeyError):
        params({'integral_gain': 1.0})


def test_integral_removes_gravity_and_friction_error():
    t, q, tgt, _ = plant(PrecisionLaw(dict(integral=False)), lambda t: 0.0)
    pd_err = abs(q[-1])                           # PD alone: (bias +- friction) / kp, here >= (1.0 - 1.2)/80 band
    law = PrecisionLaw(dict(integral=True, settle_lock=True))
    t, q, tgt, I = plant(law, lambda t: 0.0)
    assert abs(q[-1]) < 1.5e-3, q[-1]
    assert abs(q[-1]) < pd_err or pd_err < 1.5e-3
    assert law.latched                            # settled and locked: the integral stopped changing


def test_settle_lock_freezes_integral_and_unlocks_on_new_target():
    law = PrecisionLaw(dict(integral=True, settle_lock=True))
    plant(law, lambda t: 0.0, T=3.0)
    assert law.latched
    I0 = law.I.copy()
    law.step(3.0, np.zeros(6), np.zeros(6), np.zeros(6))
    assert np.allclose(law.I, I0)
    law.step(3.005, np.zeros(6), np.r_[0.01, np.zeros(5)], np.zeros(6))     # new target
    assert not law.latched


def test_integral_pauses_while_target_moves_and_resumes():
    law = PrecisionLaw(dict(integral=True))
    dt = 1 / DEFAULTS['hz']
    law.step(0.0, np.r_[-.005, np.zeros(5)], np.zeros(6))   # first cycle: no earlier target, it integrates once
    I1 = law.I[0]
    for k in range(1, 40):                        # target ramps at 0.2 rad/s, arm lags 5 mrad
        tg = 0.2 * k * dt
        law.step(k * dt, np.r_[tg - .005, np.zeros(5)], np.r_[tg, np.zeros(5)])
    assert law.I[0] == I1
    tg_end = 0.2 * 39 * dt
    k0 = 40
    for k in range(k0, k0 + 60):                  # target still: integral waits i_resume_s, then integrates
        law.step(k * dt, np.r_[tg_end - .005, np.zeros(5)], np.r_[tg_end, np.zeros(5)])
        if (k - k0 + 1) * dt < DEFAULTS['i_resume_s'] - 2 * dt:
            assert law.I[0] == I1
    assert law.I[0] > I1


def test_freeze_band_cap_and_external_freeze():
    law = PrecisionLaw(dict(integral=True))
    for k in range(400):                          # 0.2 rad error: beyond `freeze`, no integration (contact, fault)
        law.step(k * .005, np.r_[0.2, np.zeros(5)], np.zeros(6))
    assert law.I[0] == 0.0
    for k in range(400, 4000):                    # 50 mrad error: integrates to the cap
        law.step(k * .005, np.r_[-0.05, np.zeros(5)], np.zeros(6))
    assert law.I[0] == pytest.approx(DEFAULTS['i_max'][0])
    law.frozen = True
    I0 = law.I.copy()
    law.step(20.1, np.r_[0.05, np.zeros(5)], np.zeros(6))
    assert np.allclose(law.I, I0)


def test_friction_ff_only_when_target_still_and_toward_target():
    fric = [0.1, 1.0, 1.0, 0.0, 0.1, 0.1]
    law = PrecisionLaw(dict(friction_ff=True, friction=fric))
    dt = 1 / DEFAULTS['hz']
    r = law.step(0.0, np.zeros(6), np.r_[0.0, 0.01, -0.01, 0, 0, 0])
    assert np.all(r['tau'] == 0)                  # target just changed
    for k in range(1, 40):
        r = law.step(k * dt, np.zeros(6), np.r_[0.0, 0.01, -0.01, 0, 0, 0])
    assert r['tau'][1] == pytest.approx(0.7 * 1.0) and r['tau'][2] == pytest.approx(-0.7)
    r = law.step(40 * dt, np.r_[0, 0.01 - 0.00025, -0.01, 0, 0, 0], np.r_[0.0, 0.01, -0.01, 0, 0, 0])
    assert r['tau'][1] == pytest.approx(0.7 * 0.5)  # faded within fric_eps of the target


def test_vibration_fallback_recover_and_limit():
    p = dict(vibration_watch=True, friction_ff=True, friction=[0.5] * 6, max_fallbacks=2)
    law = PrecisionLaw(p)
    dt = 1 / DEFAULTS['hz']
    tgt = np.zeros(6)
    events = []
    k = 0

    def shake(n, target):
        nonlocal k
        for _ in range(n):
            qd = np.r_[0.3 * np.sin(2 * np.pi * 8 * k * dt), np.zeros(5)]
            r = law.step(k * dt, np.zeros(6), target, qd)
            if r['event']:
                events.append(r['event'])
            k += 1
        return r
    r = shake(400, tgt)
    assert events == ['fallback'] and law.fallback is not None
    assert r['tau'][0] == 0.0                     # friction feed-forward off in fallback
    shake(5, tgt + 0.01)                          # a new target beyond recover_rad: back to normal
    assert events[-1] == 'recover' and law.fallback is None
    shake(600, tgt + 0.01)
    assert events.count('fallback') == 2 and law.fallbacks == 2
    shake(5, tgt + 0.02)                          # max_fallbacks reached: stays in fallback
    assert law.fallback is not None and events[-1] == 'fallback'


class FakeRobot:
    def __init__(self, q=None):
        self._state_lock, self._command_lock = threading.Lock(), threading.Lock()
        self._joint_state = types.SimpleNamespace(pos=np.zeros(7) if q is None else np.asarray(q, float), vel=np.zeros(7))
        self._commands = types.SimpleNamespace(torques=np.zeros(7))
        self._kp, self._kd = np.r_[80, 80, 80, 40, 10, 10, 2.], np.r_[5, 5, 5, 1.5, 1.5, 1.5, .5]
        self.gains = []

    def get_joint_pos(self):
        with self._state_lock:
            return self._joint_state.pos.copy()

    def command_joint_pos(self, q7):
        with self._command_lock:
            self._commands = types.SimpleNamespace(torques=np.zeros(7))

    def update_kp_kd(self, kp, kd):
        self.gains.append((np.asarray(kp).copy(), np.asarray(kd).copy()))


def test_arm_controller_writes_torque_and_stops_clean():
    from precision_arm import ArmController
    r = FakeRobot(q=np.r_[0.0, 0.01, 0, 0, 0, 0, 1])
    target = np.zeros(7)
    c = ArmController(r, lambda: target, dict(integral=True), name='test')
    c.start()
    time.sleep(0.15)
    tau = r._commands.torques.copy()
    assert c.active and tau[1] < 0 and tau[6] == 0.0      # pulls joint 2 back toward its target; gripper untouched
    r.command_joint_pos(target)                              # a send zeroes the field...
    time.sleep(0.02)
    assert r._commands.torques[1] < 0                        # ...the thread puts it back within a cycle
    c.stop()
    assert np.all(r._commands.torques == 0) and not c.active


def test_arm_controller_error_zeroes_torque():
    from precision_arm import ArmController
    r = FakeRobot(q=np.r_[0.0, 0.01, 0, 0, 0, 0, 1])
    target = [np.zeros(7)]
    c = ArmController(r, lambda: target[0], dict(integral=True), name='test')
    c.start()
    time.sleep(0.1)
    target[0] = np.full(7, np.nan)
    time.sleep(0.05)
    assert c.error is not None and not c.active and np.all(r._commands.torques == 0)


def test_safearm_reapplies_controller_torque_and_ignores_fric_comp():
    import safety
    from precision_arm import ArmController

    class OK:
        def violation(self, q6):
            return None
    r = FakeRobot(q=np.r_[0.0, 0.01, 0, 0, 0, 0, 1])
    safe = safety.SafeArm(r, OK())
    c = ArmController(r, lambda: safe.last_safe, dict(integral=True), name='test')
    safe.ctrl = c
    c.start()
    time.sleep(0.15)
    sent = safe.command(np.zeros(7), fric_comp=([5.0] * 6, 0.01))
    assert np.allclose(safe.last_safe, sent)
    assert np.allclose(safe.last_ff, r._commands.torques[:6]) and abs(r._commands.torques[1]) < 3.0
    c.stop()


def test_gravity_fit_formats(tmp_path):
    import json
    from precision_arm import gravity_fit
    athena = {'deploy': {'ee_mass': 0.8766, 'gravity_comp_factor': [1.0, 1.1, 1.05, 1.1, 1.58, 0.63],
                         'ee_inertia': [0.0136, 0.001, -0.0111, 1, 0, 0, 0, 1e-4, 1e-4, 1e-4]}}
    f = tmp_path / 'fit.json'
    f.write_text(json.dumps(athena))
    g = gravity_fit(str(f))
    assert g['ee_mass'] == pytest.approx(0.8766) and np.allclose(g['ee_ipos'], [0.0136, 0.001, -0.0111])
    ours = gravity_fit({'ee_mass': 0.6, 'gravity_comp_factor': [1] * 6, 'ee_ipos': [0, 0, -0.02]})
    assert ours['gravity_comp_factor'].shape == (6,)
    with pytest.raises(ValueError):
        gravity_fit({'ee_mass': 9.0, 'gravity_comp_factor': [1] * 6})
