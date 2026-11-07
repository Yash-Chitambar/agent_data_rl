"""Bumpless switch-off of the precision controller (deploy/bimanual/io_real.py, arm_control.<side>.bumpless_off).

On the real-I/O mock (deploy/checks/bimanual_realio_mock.py's fake arms), with a stub i2rt gravity model whose 'fit' and
'default' torques differ: build an integral in the controller, then leave its active stage. At a FIXED measured pose,
the motor torque the arm is commanded to produce is E = kp (target - q) + command torques + gravity model torque. Its
change across the switching send, divided by kp, is the jump of the commanded equilibrium (rad). Pass: < 0.5 mrad on
every joint with bumpless_off; without it the jump is reported for contrast.

    PYTHONPATH=. .venv/bin/python -m deploy.checks.handover_bumpless
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))

G_FIT = np.array([0.0, 4.1, 1.9, 0.30, 0.05, 0.0])        # stub gravity torques at the test pose (N m)
G_DEF = np.array([0.0, 4.9, 1.4, 0.25, 0.05, 0.0])        # i2rt default model: 0.8 / 0.5 N m off on joints 2 / 3 (as measured)
I_SET = np.array([0.19, -1.14, 0.75, 0.05, 0.12, -0.02])  # the real integral at the align_full8 hand-over (N m)


def stub_gravity(robot):
    robot.gravity_models = {'fit': {}, 'default': {}}
    robot.gravity_in_use = 'fit'
    robot.gravity_comp_factor = np.ones(7)
    robot.kdl = SimpleNamespace(compute_inverse_dynamics=lambda q, v, a: (G_FIT if robot.gravity_in_use == 'fit' else G_DEF).copy())


def motor_torque(robot, q7):
    kp = robot._kp[:6]
    g = G_FIT if robot.gravity_in_use == 'fit' else G_DEF
    return kp * (robot.target[:6] - q7[:6]) + robot._commands.torques[:6] + g


def run(bumpless):
    import precision_arm
    from deploy.bimanual import io_real
    from deploy.backends import real_loop
    from deploy.bimanual.scene import SceneModel
    from deploy.run_bimanual import load_cfg
    import deploy.checks.bimanual_realio_mock as M
    cfg = load_cfg(ROOT / 'deploy/config/bimanual.json')
    cfg['arm_control'] = copy.deepcopy(json.loads((ROOT / 'deploy/config/bimanual_align4_precision.json').read_text())['arm_control'])
    cfg['arm_control']['right'].pop('gravity', None)
    if bumpless:
        cfg['arm_control']['right']['bumpless_off'] = True
    snap = ROOT / 'data/sim_ref/2026-10-25_scene'
    jm = json.loads((snap / 'pointed.json').read_text())['joints_measured']
    D = real_loop()
    D.RealArm = lambda side, kp, kd, **kw: M.FakeArm(side, kp, kd, jm[side], **kw)
    import cv2
    import deploy.cameras as DC
    raws = {'top': cv2.imread(str(snap / 'pointed_zed_left.png')), 'wrist_right': cv2.imread(str(snap / 'pointed_wrist_right.png')),
            'wrist_left': cv2.imread(str(snap / 'pointed_wrist_left.png'))}
    DC.ZedTop = lambda: M.FakeCam(raws['top'])
    DC.WristCam = lambda side: M.FakeCam(raws['wrist_' + side])
    orig_use = precision_arm.use_gravity

    def use_gravity(robot, name):                    # the stub model: switch which torque it returns
        if getattr(robot, 'gravity_in_use', None) == name or getattr(robot, 'gravity_models', None) is None:
            return False
        robot.gravity_in_use = name
        return True
    precision_arm.use_gravity = use_gravity
    try:
        sc = SceneModel('cpu')
        io = io_real.RealIO(cfg, sc.kp.numpy(), sc.kd.numpy(), log=None)
        arm = io.arms['right']
        r = arm.robot
        stub_gravity(r)
        io.precision_active('right', True)
        s = io.read()
        q_cmd = s['q'].copy()
        for _ in range(5):
            io.send(q_cmd, np.array([0.5, 1.0]))
            time.sleep(0.01)
        arm.ctrl.freeze(True)
        with arm.ctrl._lock:
            arm.ctrl.law.I = I_SET.copy()
            arm.ctrl.law.p['friction_ff'] = False     # a constant controller torque for the comparison
            arm.ctrl.law.tau = I_SET.copy()
        time.sleep(0.05)
        io.send(q_cmd, np.array([0.5, 1.0]))          # torque re-applied after this command
        q7 = r.get_joint_pos()
        e0 = motor_torque(r, q7)
        io.precision_active('right', False)           # leaving the active stage (the chain's _precision_stage)
        io.send(q_cmd, np.array([0.5, 1.0]))          # the next tick of the chain (same target, same pose)
        e1 = motor_torque(r, q7)
        jump = (e1 - e0) / r._kp[:6]
        out = dict(bumpless=bumpless, ctrl_enabled_after=bool(arm.ctrl.enabled), gravity_after=r.gravity_in_use,
                   bias_mrad=(io.bias['right'] * 1e3).round(3).tolist(), jump_mrad=(jump * 1e3).round(3).tolist(),
                   max_jump_mrad=float(np.abs(jump).max() * 1e3))
        # a later slow move clears the bias (moves start from the measured pose)
        io.move_to({'right': q_cmd[0]}, settle_trim=False)
        out['bias_after_move_mrad'] = (io.bias['right'] * 1e3).round(3).tolist()
        for a in io.arms.values():
            a.close()
        return out
    finally:
        precision_arm.use_gravity = orig_use


def main():
    a, b = run(True), run(False)
    ok = (a['max_jump_mrad'] < 0.5 and not a['ctrl_enabled_after'] and a['gravity_after'] == 'default'
          and not any(a['bias_after_move_mrad']) and b['max_jump_mrad'] > 5)
    print(json.dumps(dict(bumpless=a, plain=b, ok=bool(ok)), indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
