"""The parallel schedule's real path (deploy/bimanual/parallel.move_real) on mock hardware: RealIO with fake i2rt robots
(first-order tracking), the REAL RealArm.move_to code (scripts/real/deploy.py, bound to the fake arms), the real SafeArm +
FlexChecker, a private software-stop file. No CAN, no cameras.

Checks:
  1. serial (RealIO.move_to) and parallel capture moves reach the same goals; the parallel one takes about the
     longer arm's time, not the sum;
  2. a SafetyStop of one arm in the middle of a combined move stops BOTH (the other arm short of its goal), both hold,
     the SafetyStop reaches the caller;
  3. the operator's software stop (estop file) during a combined move stops both;
  4. the arm-arm watch (watch_min_mm above the real distance) stops both;
  5. the lead: the second arm does not move during the lead.

    PYTHONPATH=. .venv/bin/python -m deploy.checks.parallel_realio_mock
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(tempfile.mkdtemp())
os.environ['YAM_ESTOP_FILE'] = str(TMP / 'estop_mock')          # never the live stop file
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))

from deploy.checks.bimanual_realio_mock import FakeCam, FakeRobot  # noqa: E402

FAILED = []


def check(name, ok, info=''):
    print(('ok     ' if ok else 'FAILED ') + name + (f'  {info}' if info else ''), flush=True)
    if not ok:
        FAILED.append(name)


def main():
    import cv2
    from deploy.backends import real_loop
    from deploy.bimanual import io_real, parallel as PS
    from deploy.bimanual.runlog import RunLogger
    from deploy.bimanual.scene import SceneModel
    from deploy.run_bimanual import load_cfg
    import deploy.cameras as DC
    D = real_loop()
    cfg = load_cfg(ROOT / 'deploy/config/bimanual_align11_par_home_capture.json')
    snap = ROOT / 'data/sim_ref/2026-10-25_scene'
    raws = {'top': cv2.imread(str(snap / 'pointed_zed_left.png')), 'wrist_right': cv2.imread(str(snap / 'pointed_wrist_right.png')),
            'wrist_left': cv2.imread(str(snap / 'pointed_wrist_left.png'))}
    jm = json.loads((snap / 'pointed.json').read_text())['joints_measured']

    class Arm:
        """RealArm's state on a FakeRobot; move_to IS scripts/real/deploy.RealArm.move_to."""
        move_to = D.RealArm.move_to
        hold = D.RealArm.hold

        def __init__(self, side, kp, kd, **kw):
            self.side, self.robot = side, FakeRobot(jm[side])
            self.safe, self.ctrl, self.motion_trace = None, None, []
            self.timing = type('T', (), {'label': None, 'enc': []})()
            self.kp, self.kd = kp, kd
            self.fail_after = None          # test hook: raise a SafetyStop from the n-th command on

        def read(self):
            self.robot.step()
            return self.robot.get_joint_pos(), np.zeros(7), time.monotonic(), True

        def info(self):
            return {'kp': list(self.kp), 'kd': list(np.minimum(self.kd, 5.0)), 'gains_before': None}

        def wait(self, t_next):
            while time.monotonic() < t_next:
                time.sleep(0.001)
            self.robot.step()

        def close(self):
            pass

    D.RealArm = Arm
    DC.ZedTop = lambda: FakeCam(raws['top'])
    DC.WristCam = lambda side: FakeCam(raws['wrist_' + side])
    sc = SceneModel('cpu')
    log = RunLogger(TMP / 'log', {'check': 'parallel_realio_mock'})
    io = io_real.RealIO(cfg, sc.kp.numpy(), sc.kd.numpy(), log=log)
    off = sc.encoder_offsets.numpy().astype(float)
    cap = json.loads((ROOT / 'blind_insertion/configs/zed_capture_pose.json').read_text())['model_frame_rad']
    goal = {s: np.asarray(cap[s], float) - off[i] for i, s in enumerate(('right', 'left'))}        # motor frame
    # start: the capture pose with base yaw and wrist roll turned 0.35 rad (moves of about 1.3 s at 0.4 rad/s)
    start = {s: goal[s] + np.array([0.35 * sg, 0., 0., 0., 0., 0.35]) for s, sg in (('right', -1.), ('left', 1.))}
    model = PS.ArmArmModel(sc)
    vmax = 0.4
    stopf = Path(os.environ['YAM_ESTOP_FILE'])

    def reset():
        stopf.unlink(missing_ok=True)
        for s in ('right', 'left'):
            r = io.arms[s].robot
            with r._state_lock:
                r._joint_state.pos[:6] = start[s]
            r.target[:6] = start[s]
            io.arms[s].safe._hold(r.get_joint_pos())
            io.arms[s].fail_after = None

    def err(s):
        return float(np.abs(io.arms[s].robot.get_joint_pos()[:6] - goal[s]).max())

    # 1. serial vs parallel
    reset()
    t0 = time.monotonic()
    io.move_to({'right': goal['right'], 'left': goal['left']}, max_vel=vmax)
    t_ser = time.monotonic() - t0
    e_ser = max(err('right'), err('left'))
    reset()
    t0 = time.monotonic()
    PS.move_real(io, goal, None, vmax, True, 0.0, 'right', model, off, 30.0, log=log)
    t_par = time.monotonic() - t0
    e_par = max(err('right'), err('left'))
    check('serial and parallel reach the goals', e_ser < 0.02 and e_par < 0.02, f'max error serial {e_ser:.4f}, parallel {e_par:.4f} rad')
    check('parallel time ~ longer arm, not the sum', t_par < 0.75 * t_ser, f'serial {t_ser:.2f} s, parallel {t_par:.2f} s')

    # 2. SafetyStop of one arm stops both
    reset()
    arm = io.arms['right']
    orig = arm.safe.command
    n = {'k': 0}

    def failing(q7, ff=None):
        n['k'] += 1
        if n['k'] > 30:
            raise io.safety.SafetyStop('mock: target unsafe (test)')
        return orig(q7, ff)
    arm.safe.command = failing
    try:
        PS.move_real(io, goal, None, vmax, True, 0.0, 'left', model, off, 30.0, log=log)
        got = 'no stop'
    except io.safety.SafetyStop as e:
        got = str(e)
    arm.safe.command = orig
    time.sleep(0.1)
    el, er = err('left'), err('right')
    check('a stop of the right arm stops the left arm too', 'mock: target unsafe' in got and el > 0.05 and er > 0.05,
          f'{got[:90]}; left {el:.3f} rad, right {er:.3f} rad from the goal')

    # 3. software stop file
    reset()
    threading.Timer(0.6, lambda: stopf.write_text('stop')).start()
    try:
        PS.move_real(io, goal, None, vmax, True, 0.0, 'right', model, off, 30.0, log=log)
        got = 'no stop'
    except io.safety.SafetyStop as e:
        got = str(e)
    stopf.unlink(missing_ok=True)
    el, er = err('left'), err('right')
    check('the software stop stops both', 'software stop' in got and el > 0.05 and er > 0.05, f'{got[:90]}; left {el:.3f}, right {er:.3f} rad')

    # 4. watch
    reset()
    try:
        PS.move_real(io, goal, None, vmax, True, 0.0, 'right', model, off, 5000.0, log=log)
        got = 'no stop'
    except io.safety.SafetyStop as e:
        got = str(e)
    el, er = err('left'), err('right')
    check('the arm-arm watch stops both', 'arm-arm watch' in got and el > 0.05 and er > 0.05, f'{got[:90]}')

    # 5. lead: the left does not move during the right's lead... here the left leads by 1.5 s: right still at start at 1.0 s
    reset()
    seen = {}
    threading.Timer(1.0, lambda: seen.update(r=float(np.abs(io.arms['right'].robot.get_joint_pos()[:6] - start['right']).max()),
                                             l=float(np.abs(io.arms['left'].robot.get_joint_pos()[:6] - start['left']).max()))).start()
    PS.move_real(io, goal, None, vmax, True, 1.5, 'left', model, off, 30.0, log=log)
    check('lead: the second arm holds during the lead, the first arm moves', seen.get('r', 1) < 1e-3 and seen.get('l', 0) > 1e-3, str(seen))
    check('both reach the goals after the lead', max(err('right'), err('left')) < 0.02)
    io.hold()
    log.close()
    print('FAILED: ' + (', '.join(FAILED) if FAILED else 'none'))
    raise SystemExit(1 if FAILED else 0)


if __name__ == '__main__':
    main()
