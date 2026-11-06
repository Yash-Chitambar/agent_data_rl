"""RealIO on mock hardware (no CAN, no cameras): the real SafeArm + FlexChecker + friction feed-forward on a fake i2rt
robot (first-order joint tracking), fake RealArm (scripts/real/deploy.RealArm interface) and fake camera grabbers
(fixed real frames through the real preprocessing). Exercises read / send (incl. SafeArm.command(q7, fric_comp)) /
move_to / frames / drain_can / hold and a short RealChain capture + hold with the run logger, i.e. the code the
dry run (SimIO) bypasses.

    PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_realio_mock
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))


class FakeRobot:
    def __init__(self, q7):
        self._joint_state = SimpleNamespace(pos=np.asarray(q7, float), vel=np.zeros(7), eff=np.zeros(7), timestamp=0.0)
        self._state_lock = threading.Lock()
        self._command_lock = threading.RLock()
        self._commands = SimpleNamespace(torques=np.zeros(7))
        self._kp = np.array([80, 80, 80, 40, 10, 10, 2.], float)
        self._kd = np.array([5, 5, 5, 1.5, 1.5, 1.5, .5], float)
        self.target = np.asarray(q7, float).copy()

    def update_kp_kd(self, kp, kd):
        self._kp, self._kd = np.asarray(kp, float).copy(), np.asarray(kd, float).copy()

    def get_joint_pos(self):
        with self._state_lock:
            return self._joint_state.pos.copy()

    def command_joint_pos(self, q7):
        self.target = np.asarray(q7, float).copy()
        self._commands.torques[:] = 0

    def step(self):
        with self._state_lock:
            p = self._joint_state.pos
            self._joint_state.pos = p + 0.5 * (self.target - p)
            self._joint_state.timestamp = time.monotonic()


class FakeArm:
    """scripts/real/deploy.RealArm interface on a FakeRobot."""

    def __init__(self, side, kp, kd, q7, gravity=None, precision=None):
        self.side, self.robot = side, FakeRobot(q7)
        self.safe = None                      # RealIO installs SafeArm with the FlexPoint checker
        self.timing = SimpleNamespace(label=None, enc=[])
        self.motion_trace = []
        self.kp, self.kd = kp, kd
        self.gravity, self.ctrl = gravity, None
        if precision is not None:             # as deploy.RealArm: the controller follows whichever SafeArm is attached
            from precision_arm import ArmController
            self.ctrl = ArmController(self.robot, lambda: self.safe.last_safe if self.safe is not None
                                      else self.robot.get_joint_pos(), dict(precision), name=side)
            self.ctrl.start()

    def read(self):
        self.robot.step()
        t = time.monotonic()
        self.timing.enc.append((t, len(self.timing.enc), self.robot.get_joint_pos()))
        return self.robot.get_joint_pos(), np.zeros(7), t, True

    def info(self):
        return {'kp': list(self.kp), 'kd': list(np.minimum(self.kd, 5.0)), 'gains_before': None}

    def move_to(self, q6, max_vel=0.3, settle_trim=False):
        q0 = self.robot.get_joint_pos()
        for s in np.linspace(0, 1, 20):
            tgt = q0.copy()
            tgt[:6] = (1 - s) * q0[:6] + s * np.asarray(q6)
            sent = self.safe.command(tgt)
            for _ in range(3):
                self.robot.step()
            self.motion_trace.append(dict(host_monotonic=time.monotonic(), target=sent.tolist(),
                                          measured=self.robot.get_joint_pos().tolist(), mode='move_to'))

    def wait(self, t):
        pass

    def hold(self):
        self.safe._hold(self.robot.get_joint_pos())

    def close(self):
        if self.ctrl is not None:
            self.ctrl.stop()


class FakeCam:
    def __init__(self, raw):
        self.raw, self.prep, self.k = raw, None, 0
        self._pol = None

    def latest_policy(self):
        self.k += 1
        if self._pol is None and self.prep is not None:
            self._pol = self.prep(self.raw)
        return self._pol, time.monotonic(), self.k

    def latest(self):
        return self.raw, time.monotonic(), self.k

    def close(self):
        pass


def main():
    import cv2
    import torch
    from deploy.bimanual import io_real
    from deploy.backends import real_loop
    from deploy.bimanual.chain import RealChain
    from deploy.bimanual.guards import Guards
    from deploy.bimanual.runlog import RunLogger, load_run
    from deploy.bimanual.scene import SceneModel
    from deploy.run_bimanual import gripper_maps, load_cfg
    import deploy.cameras as DC
    cfg = load_cfg(ROOT / 'deploy/config/bimanual.json')
    snap = ROOT / 'data/sim_ref/2026-10-25_scene'
    raws = {'top': cv2.imread(str(snap / 'pointed_zed_left.png')), 'wrist_right': cv2.imread(str(snap / 'pointed_wrist_right.png')),
            'wrist_left': cv2.imread(str(snap / 'pointed_wrist_left.png'))}
    jm = json.loads((snap / 'pointed.json').read_text())['joints_measured']
    D = real_loop()
    D.RealArm = lambda side, kp, kd, **kw: FakeArm(side, kp, kd, jm[side], **kw)
    DC.ZedTop = lambda: FakeCam(raws['top'])
    DC.WristCam = lambda side: FakeCam(raws['wrist_' + side])
    sc = SceneModel('cpu')
    tmp = Path(tempfile.mkdtemp())
    log = RunLogger(tmp, {'check': 'realio_mock'})
    io = io_real.RealIO(cfg, sc.kp.numpy(), sc.kd.numpy(), log=log)
    res = {}
    s = io.read()
    res['read_shapes'] = {k: list(np.shape(v)) for k, v in s.items()}
    sent, ff = io.send(s['q'] + 0.01, np.array([0.5, 1.0]), (cfg['fric_comp_right']['tau'], cfg['fric_comp_right']['ramp']))
    res['ff_right_nm'] = ff[0].round(4).tolist()
    res['ff_written_to_command'] = bool(np.allclose(io.arms['right'].robot._commands.torques[:6], ff[0]))
    r_ = io.arms['right'].robot
    g0 = r_._kp.copy(), r_._kd.copy()
    io.set_gains('right', [80, 80, 80, 40, 30, 30], [5, 5, 5, 1.5, 2, 9.])
    res['set_gains'] = bool(r_._kp[5] == 30 and r_._kd[5] == 5. and r_._kp[6] == g0[0][6] and r_._kd[6] == g0[1][6])   # kd capped; gripper untouched
    io.set_gains('right', g0[0][:6], g0[1][:6])
    fr = io.frames(('top', 'wrist_right', 'wrist_left'))
    res['frames'] = {k: [list(v[0].shape), str(v[0].dtype)] for k, v in fr.items()}
    res['can_rows'] = {k: len(v) for k, v in io.drain_can().items()}
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    g = Guards(cfg, sc.kp.numpy(), cfg['control_dt'], max_steps=400, estop_fn=lambda: False)
    ch = RealChain(cfg, io, sc, consts, gripper_maps(cfg), log=log, guards=g, device='cpu', net_device='cpu',
                   policy_path=ROOT / cfg['bundle_dir'] / 'models/insert_KJ.pt', auto='')
    lg = {}
    ep, eR, ok = ch.capture_pc(lg)
    res['capture'] = {k: lg['zed'][k] for k in ('port_xy', 'yaw_deg', 'ok')}
    ch.park_left()
    for _ in range(10):
        ch.hstep(ch.q_b[0], ch.q_b[1], '2 hold')
    try:
        for _ in range(50):
            ch.hstep(ch.q_b[0], ch.q_b[1] + torch.tensor([.3, 0, 0, 0, 0, 0.]), '2 hold')
        res['torque_guard'] = 'did NOT trip'
    except Exception as e:  # noqa: BLE001
        res['torque_guard'] = f'{type(e).__name__}: {str(e)[:120]}'
    io.hold()
    log.close()
    meta, st, ev, frs = load_run(tmp)
    res['log_streams'] = {k: len(next(iter(v.values()))) for k, v in st.items()}
    res['log_frames'] = len(frs)
    res['dropped'] = meta['dropped']
    ok_all = res['set_gains'] and res['ff_written_to_command'] and 'Stop' in res['torque_guard'] and 'ticks' in st and 'moves_right' in st
    res['precision'] = precision_check(cfg, sc, raws, jm)
    ok_all = ok_all and res['precision']['ok']
    res['ok'] = bool(ok_all)
    print(json.dumps(res, indent=1, default=str))
    raise SystemExit(0 if ok_all else 1)


def precision_check(cfg, sc, raws, jm):
    """RealIO with cfg['arm_control']: the right arm's controller thread runs, every send carries its torque (not the
    Coulomb feed-forward), ctrl_right rows are logged, the left arm has none, and close stops it with zero torque."""
    import copy
    from deploy.bimanual import io_real
    from deploy.bimanual.runlog import RunLogger, load_run
    cfg = copy.deepcopy(cfg)
    cfg['arm_control'] = {'right': {'precision': {'integral': True, 'settle_lock': True}}}
    tmp = Path(tempfile.mkdtemp())
    log = RunLogger(tmp, {'check': 'realio_mock_precision'})
    io = io_real.RealIO(cfg, sc.kp.numpy(), sc.kd.numpy(), log=log)
    out = {}
    r = io.arms['right']
    s = io.read()
    for _ in range(20):
        sent, ff = io.send(s['q'] + np.array([[0, .004, 0, 0, 0, 0], [0] * 6]), np.array([0.5, 1.0]),
                           (cfg['fric_comp_right']['tau'], cfg['fric_comp_right']['ramp']))
        time.sleep(0.01)
    out['ctrl_active'] = bool(r.ctrl.active)
    out['ff_is_ctrl_torque'] = bool(np.allclose(ff[0], r.safe.last_ff) and np.allclose(r.robot._commands.torques[:6], r.ctrl.tau(), atol=0.05))
    out['left_has_no_ctrl'] = io.arms['left'].ctrl is None and np.allclose(ff[1], 0)
    out['I_right'] = r.ctrl.status()['I'].round(3).tolist()
    io.hold()
    for a in io.arms.values():
        a.close()
    out['zero_after_close'] = bool(np.all(r.robot._commands.torques == 0))
    log.close()
    _, st, _, _ = load_run(tmp)
    out['ctrl_rows'] = len(next(iter(st['ctrl_right'].values()))) if 'ctrl_right' in st else 0
    out['ok'] = bool(out['ctrl_active'] and out['ff_is_ctrl_torque'] and out['left_has_no_ctrl'] and out['zero_after_close']
                     and out['ctrl_rows'] >= 20 and out['I_right'][1] != 0)
    return out


if __name__ == '__main__':
    main()
