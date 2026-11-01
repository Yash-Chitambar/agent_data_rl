"""Both real YAM arms and the three cameras behind one interface (the dry run's SimIO speaks the same one).

Joint frames: everything here is the MOTOR (encoder) frame i2rt reads and takes. The chain layer adds
rig.joint_offsets(side) to get the model ("believed") frame and subtracts them from its targets.
Gripper values are i2rt units (0 closed .. 1 open).

Arms: scripts/real/deploy.RealArm per side = scripts/real/safety.connect (PD hold from the first instant,
i2rt native gains = the bimanual sim's controller profile), BusTiming CAN hooks (every encoder reading stamped
at arrival), SafeArm with the FlexPoint FlexChecker (joint box, table clearance on targets and measured poses,
step cap, deviation limit). The right-arm Coulomb feed-forward goes through SafeArm.command(q7, fric_comp).
Cameras: deploy/cameras ZedTop (HD1080 rectified left -> real_top_policy_image) and WristCam per side
(1280x1024 MJPG -> real_policy_image), one capture thread each, policy image made in the thread.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / 'scripts/real'))

SIDES = ('right', 'left')


class RealIO:
    is_sim = False

    def __init__(self, cfg, kp, kd, cams=('top', 'wrist_right', 'wrist_left'), log=None):
        import estop
        import safety
        from deploy.backends import real_loop
        from deploy.safety_task import FlexChecker
        self.estop, self.safety, self.D = estop, safety, real_loop()
        self.cfg, self.log, self.dt = cfg, log, cfg['control_dt']
        from deploy import conventions as C
        self.off = np.stack([C.joint_offsets(s) for s in SIDES])
        self.cams = {}
        self._open_cameras(cams)          # blocking SDK init before the motors take commands
        sc = cfg['safety']
        self.arms = {}
        try:
            for i, side in enumerate(SIDES):
                arm = self.D.RealArm(side, np.asarray(kp[i], float), np.asarray(kd[i], float))
                chk = FlexChecker(self.off[i], clearance=sc['clearance'], tip_clearance=sc['tip_clearance'],
                                  body_clearance=sc.get('body_clearance'), table_z=sc['table_z'])
                arm.safe = safety.SafeArm(arm.robot, chk)
                arm.safe._hold(np.asarray(arm.robot.get_joint_pos(), float))
                self.arms[side] = arm
                info = arm.info()
                if log is not None:
                    log.event('arm_connected', side=side, **info)
                if not np.allclose(info['kp'], kp[i]) or not np.allclose(info['kd'], np.minimum(kd[i], self.D.DM_KD_MAX)):
                    raise RuntimeError(f'{side} arm gains {info["kp"]}/{info["kd"]} differ from the sim profile {kp[i]}/{kd[i]}')
        except Exception:
            self.close()
            raise
        r = self.arms['right'].robot
        self.ff_ok = hasattr(r, '_command_lock') and hasattr(getattr(r, '_commands', None), 'torques')
        if not self.ff_ok:
            print('WARNING: this i2rt has no _command_lock/_commands.torques: the right-arm friction feed-forward is OFF '
                  '(the approach then sags; the sim chain used it)', flush=True)
        if log is not None:
            log.event('ff_support', ok=self.ff_ok)
        self.can_idx = {s: 0 for s in SIDES}
        self.k = 0
        self.t_next = None

    def _open_cameras(self, names):
        from blind_insertion.assets.real_camera import load_spec, real_policy_image
        from blind_insertion.scenes.physical_rig import real_top_policy_image
        import rig
        from deploy.cameras import WristCam, ZedTop
        spec, res = load_spec(rig.SPEC), self.cfg['cameras']['res']
        for n in names:
            if n == 'top':
                c = ZedTop()
                c.prep = lambda f: real_top_policy_image(f, res)
            else:
                side = n.split('_')[1]
                c = WristCam(side)
                c.prep = (lambda s: (lambda f: real_policy_image(f, spec, res, s)))(side)
            self.cams[n] = c
        t0 = time.monotonic()
        while any(c.latest_policy()[0] is None for c in self.cams.values()):
            if time.monotonic() - t0 > 8.0:
                raise RuntimeError('no frames from ' + ', '.join(k for k, c in self.cams.items() if c.latest_policy()[0] is None))
            time.sleep(0.01)

    # ------------------------------------------------------------------ state and commands
    def read(self):
        """dict: q (2,6) motor frame, qd (2,6), grip (2,) i2rt reading, eff (2,6) driver effort, t_obs (2,) CAN
        arrival of the reading used, t (host monotonic)."""
        q, qd, g, eff, t_obs = np.zeros((2, 6)), np.zeros((2, 6)), np.zeros(2), np.full((2, 6), np.nan), np.zeros(2)
        for i, s in enumerate(SIDES):
            arm = self.arms[s]
            q7, qd7, t_obs[i], _ = arm.read()
            q[i], qd[i], g[i] = q7[:6], qd7[:6], q7[6]
            r = arm.robot
            with r._state_lock:
                e = getattr(r._joint_state, 'eff', None)
            if e is not None:
                eff[i] = np.asarray(e, float)[:6]
        return dict(q=q, qd=qd, grip=g, eff=eff, t_obs=t_obs, t=time.monotonic())

    def send(self, q_motor, grip, ff_right=None):
        """q_motor (2,6) targets, grip (2,) i2rt targets, ff_right = (tau, ramp) Coulomb feed-forward or None.
        Returns what SafeArm sent (2,7) and the feed-forward torques (2,6)."""
        sent, ff = np.zeros((2, 7)), np.zeros((2, 6))
        if not self.ff_ok:
            ff_right = None
        for i, s in enumerate(SIDES):
            arm = self.arms[s]
            arm.timing.label = (self.k, s)
            sent[i] = arm.safe.command(np.append(np.asarray(q_motor[i], float), float(grip[i])),
                                       ff_right if s == 'right' else None)
            lf = getattr(arm.safe, 'last_ff', None)
            if lf is not None and s == 'right' and ff_right is not None:
                ff[i] = lf
        self.k += 1
        return sent, ff

    def start_clock(self):
        self.t_next = time.monotonic()

    def wait_tick(self):
        if self.t_next is None:
            self.start_clock()
        self.t_next += self.dt
        late = time.monotonic() - self.t_next
        if late > 5 * self.dt:          # a long computation (IK, a net): do not try to catch up
            self.t_next = time.monotonic()
        self.arms['right'].wait(self.t_next)
        return late

    def move_to(self, q_motor, grip=None, settle_trim=True, max_vel=0.15):
        """Slow checked moves, one arm after the other: {side: q6 motor}; grip {side: i2rt} set first."""
        for s, q in q_motor.items():
            arm = self.arms[s]
            if grip is not None and s in grip:
                q7 = np.asarray(arm.robot.get_joint_pos(), float)
                q7[6] = grip[s]
                arm.safe.command(q7)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 1.0 and abs(arm.robot.get_joint_pos()[6] - grip[s]) > 0.03:
                    time.sleep(0.02)
            if self.log is not None:
                self.log.event('move_to', side=s, goal_motor=np.asarray(q, float).tolist())
            n0 = len(arm.motion_trace)
            try:
                arm.move_to(np.asarray(q, float), max_vel=max_vel, settle_trim=settle_trim)
            finally:
                if self.log is not None:     # slow moves at 50 Hz: target / measured (system-ID data for the replay)
                    for r in arm.motion_trace[n0:]:
                        self.log.row('moves_' + s, t=r['host_monotonic'], target=np.asarray(r['target'], float),
                                     measured=np.asarray(r['measured'], float), settle=float(r['mode'] == 'settle'))
        self.t_next = None

    def set_grip(self, side, value, wait_s=1.0):
        arm = self.arms[side]
        q7 = np.asarray(arm.robot.get_joint_pos(), float)
        q7[6] = value
        arm.safe.command(q7)
        time.sleep(wait_s)

    def hold(self):
        for arm in self.arms.values():
            try:
                arm.hold()
            except Exception as e:  # noqa: BLE001
                print(f'hold failed: {e}', flush=True)

    def stopped(self):
        return self.estop.stopped()

    def drain_can(self):
        """{side: [(t_arrival, q7), ...]} encoder readings since the last call (CAN rate)."""
        out = {}
        for s, arm in self.arms.items():
            enc = arm.timing.enc
            i = self.can_idx[s]
            out[s] = [(t, q) for (t, _, q) in enc[i:]]
            self.can_idx[s] = len(enc)
        return out

    # ------------------------------------------------------------------ cameras
    def frames(self, names=('top', 'wrist_right')):
        """{name: (policy RGB (res,res,3), raw BGR, t_arrival, frame count)} of the newest frames; raises if a
        frame is older than cameras.max_age_ms."""
        now, lim = time.monotonic(), self.cfg['cameras']['max_age_ms'] / 1e3
        out = {}
        for n in names:
            c = self.cams[n]
            pol, t, k = c.latest_policy()
            raw, _, _ = c.latest()
            if now - t > lim:
                raise RuntimeError(f'{n} camera frame is {(now - t) * 1e3:.0f} ms old')
            out[n] = (pol, raw, t, k)
        return out

    def fresh_frames(self, names=('top', 'wrist_right'), timeout_s=1.0):
        """Wait for a frame captured after this call (so it shows the arm at rest after a move)."""
        k0 = {n: self.cams[n].latest()[2] for n in names}
        t0 = time.monotonic()
        while any(self.cams[n].latest()[2] <= k0[n] + 1 for n in names):
            if time.monotonic() - t0 > timeout_s:
                break
            time.sleep(0.005)
        return self.frames(names)

    def truth(self):
        return None

    def close(self, read_line=input):
        """Cameras off; arms hold powered until the operator types SUPPORTED (deploy/recovery.py rule), then torque off."""
        for c in self.cams.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        if not getattr(self, 'arms', None):
            return
        self.hold()
        print('Holding powered. Secure both arms and any held object. Type SUPPORTED to disable torque.', flush=True)
        while True:
            try:
                ans = read_line('support confirmation > ')
            except (EOFError, KeyboardInterrupt):
                print('No support confirmation; maintaining power.', flush=True)
                time.sleep(.5)
                continue
            if ans.strip() == 'SUPPORTED':
                break
            print('Still holding; torque remains enabled.', flush=True)
        for s, arm in self.arms.items():
            try:
                arm.close()
            except Exception as e:  # noqa: BLE001
                print(f'{s} close: {e}', flush=True)
        self.arms = {}
