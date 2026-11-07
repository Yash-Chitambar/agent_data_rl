"""Both real YAM arms and the three cameras behind one interface (the dry run's SimIO speaks the same one).

Joint frames: everything here is the MOTOR (encoder) frame i2rt reads and takes. The chain layer adds
rig.joint_offsets(side) to get the model ("believed") frame and subtracts them from its targets.
Gripper values are i2rt units (0 closed .. 1 open).

Optional cfg['arm_control'][side] = {'gravity': fitted payload model (path or dict), 'precision': precision
controller params} (scripts/real/precision_arm.py; blind_insertion/envs/precision_ctrl.py): a 200 Hz thread per arm
adds the integral / settle lock / friction feed-forward torque to every command (logged as ctrl_<side> rows). Absent:
unchanged behaviour.

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
        # GIL convoy: i2rt's motor threads release/re-take the GIL on every CAN send/receive; while the control loop
        # runs NumPy-heavy code (FastChain IK, the hold plan) each re-take waits up to the switch interval (5 ms
        # default). Real run 2026-11-01 hold_160735: 250-270 ms without CAN readings during the hold plan. Offline
        # (14 I/O calls per motor cycle vs a NumPy loop): cycle p50 61 ms at 5 ms, 7 ms at 0.5 ms.
        import sys
        self.gil_switch_s = float(cfg.get('gil_switch_s', 0.0005))
        sys.setswitchinterval(self.gil_switch_s)
        self._open_cameras(cams)          # blocking SDK init before the motors take commands
        sc = cfg['safety']
        # per-arm gravity model and precision controller (scripts/real/precision_arm.py); absent = as before
        ac = cfg.get('arm_control') or {}
        self.arms = {}
        try:
            for i, side in enumerate(SIDES):
                a = {k: v for k, v in (ac.get(side) or {}).items() if k in ('gravity', 'precision') and v is not None}
                arm = self.D.RealArm(side, np.asarray(kp[i], float), np.asarray(kd[i], float), **a)
                chk = FlexChecker(self.off[i], clearance=sc['clearance'], tip_clearance=sc['tip_clearance'],
                                  body_clearance=sc.get('body_clearance'), table_z=sc['table_z'])
                arm.safe = safety.SafeArm(arm.robot, chk)
                arm.safe.ctrl = getattr(arm, 'ctrl', None)
                arm.safe._hold(np.asarray(arm.robot.get_joint_pos(), float))
                if 'active_stages' in (ac.get(side) or {}):
                    self._precision_active(arm, side, False)     # the old behaviour until the chain enters an active stage
                self.arms[side] = arm
                info = arm.info()
                if log is not None:
                    log.event('arm_connected', side=side, **info)
                if not np.allclose(info['kp'], kp[i]) or not np.allclose(info['kd'], np.minimum(kd[i], self.D.DM_KD_MAX)):
                    raise RuntimeError(f'{side} arm gains {info["kp"]}/{info["kd"]} differ from the sim profile {kp[i]}/{kd[i]}')
        except Exception:
            self.close()
            raise
        # bumpless switch-off (arm_control.<side>.bumpless_off, docs/real/K2_PROGRESS.md): leaving the controller's active
        # stages folds its torque (integral + feed-forward) and the fitted-minus-default gravity at the measured pose into a
        # held target bias (tau / kp), in the same send that switches it off, so the commanded equilibrium does not jump.
        self.bumpless = {s_: bool((ac.get(s_) or {}).get('bumpless_off', False)) for s_ in SIDES}
        self.bias = {s_: np.zeros(6) for s_ in SIDES}
        self._pending_off = {s_: False for s_ in SIDES}
        r = self.arms['right'].robot
        self.ff_ok = hasattr(r, '_command_lock') and hasattr(getattr(r, '_commands', None), 'torques')
        if not self.ff_ok:
            print('WARNING: this i2rt has no _command_lock/_commands.torques: the right-arm friction feed-forward is OFF '
                  '(the approach then sags; the sim chain used it)', flush=True)
        if log is not None:
            log.event('ff_support', ok=self.ff_ok)
            log.event('gil_switch_interval', seconds=self.gil_switch_s)
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
            ctrl = getattr(arm, 'ctrl', None)
            if self._pending_off[s]:
                self._bumpless_off(arm, s)
            qm = np.asarray(q_motor[i], float) + self.bias[s]
            sent[i] = arm.safe.command(np.append(qm, float(grip[i])),
                                       ff_right if s == 'right' else None)
            lf = getattr(arm.safe, 'last_ff', None)
            if lf is not None and (ctrl is not None or (s == 'right' and ff_right is not None)):
                ff[i] = lf          # the torque in the command: precision controller, else the Coulomb feed-forward
            if ctrl is not None and self.log is not None:
                st = ctrl.status()
                self.log.row('ctrl_' + s, t=time.monotonic(), k=self.k, I=st['I'], tau=st['tau'],
                             latched=float(st['latched']), fallback=float(st['fallback'] is not None),
                             frozen=float(st['frozen']), ok=float(st['error'] is None))
                if st['error'] is not None and not getattr(self, '_ctrl_err_logged', {}).get(s):
                    self.log.event('precision_error', side=s, error=st['error'], k=self.k)
                    self._ctrl_err_logged = {**getattr(self, '_ctrl_err_logged', {}), s: True}
        self.k += 1
        return sent, ff

    def _precision_active(self, arm, side, on):
        from precision_arm import use_gravity
        ctrl = getattr(arm, 'ctrl', None)
        if ctrl is not None:
            ctrl.set_enabled(on)
        switched = use_gravity(arm.robot, 'fit' if on else 'default')
        if self.log is not None:
            self.log.event('precision_active', side=side, on=bool(on), gravity=getattr(arm.robot, 'gravity_in_use', None),
                           gravity_switched=bool(switched), k=getattr(self, 'k', 0))

    def precision_active(self, side, on=True):
        """Controller torque and fitted gravity on (on) or the arm's old behaviour: zero controller torque, integral
        reset, i2rt's default gravity, the chain's Coulomb feed-forward (cfg arm_control.<side>.active_stages).
        With arm_control.<side>.bumpless_off, switching off waits for the next send, which folds the torque into a bias."""
        arm = self.arms[side]
        ctrl = getattr(arm, 'ctrl', None)
        if on:
            self._pending_off[side] = False
            if np.any(self.bias[side]) and self.log is not None:
                self.log.event('precision_bias_cleared', side=side, bias=self.bias[side].tolist(), k=getattr(self, 'k', 0))
            self.bias[side] = np.zeros(6)
        elif self.bumpless.get(side) and ctrl is not None and ctrl.enabled:
            self._pending_off[side] = True
            return
        self._precision_active(arm, side, on)

    @staticmethod
    def _gravity_now(robot, q6):
        kdl = getattr(robot, 'kdl', None)
        if kdl is None or getattr(robot, 'gravity_models', None) is None:
            return np.zeros(6)
        z = np.zeros(6)
        return np.asarray(kdl.compute_inverse_dynamics(np.asarray(q6, float), z, z), float)[:6] * np.asarray(robot.gravity_comp_factor, float)[:6]

    def _bumpless_off(self, arm, side):
        """Switch the controller off and hold its torque as a target bias (motor frame, rad): bias = (tau_ctrl + g_now -
        g_after) / kp, g = i2rt's gravity torque before / after the model switch, at the measured pose."""
        r = arm.robot
        q6 = np.asarray(r.get_joint_pos(), float)[:6]
        tau = np.asarray(arm.ctrl.tau(), float)[:6]
        g0 = self._gravity_now(r, q6)
        self._precision_active(arm, side, False)
        g1 = self._gravity_now(r, q6)
        kp = np.asarray(r._kp, float)[:6]
        self.bias[side] = self.bias[side] + (tau + g0 - g1) / kp
        self._pending_off[side] = False
        if self.log is not None:
            self.log.event('precision_bumpless_off', side=side, tau=tau.tolist(), dgrav=(g0 - g1).tolist(),
                           bias=self.bias[side].tolist(), k=getattr(self, 'k', 0))

    def precision_freeze(self, side, on=True):
        """Hold (on) or release the precision controller's integral on one arm, e.g. around a contact stage."""
        ctrl = getattr(self.arms[side], 'ctrl', None)
        if ctrl is not None:
            ctrl.freeze(on)
            if self.log is not None:
                self.log.event('precision_freeze', side=side, on=bool(on), k=self.k)

    def set_gains(self, side, kp, kd):
        """Arm joint gains (6,) from the next command on (i2rt update_kp_kd; gripper gains unchanged; kd capped at the
        DM MIT limit). Used by the guarded grasp for the wrist during the close."""
        r = self.arms[side].robot
        kp7, kd7 = r._kp.copy(), r._kd.copy()
        kp7[:6], kd7[:6] = np.asarray(kp, float), np.minimum(np.asarray(kd, float), self.D.DM_KD_MAX)
        r.update_kp_kd(kp7, kd7)

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
            if self._pending_off.get(s):        # a slow move: switch off plainly (moves start from the measured pose, own trim)
                self._pending_off[s] = False
                self._precision_active(arm, s, False)
            if np.any(self.bias.get(s, 0)):
                if self.log is not None:
                    self.log.event('precision_bias_cleared', side=s, bias=self.bias[s].tolist(), k=getattr(self, 'k', 0), at='move_to')
                self.bias[s] = np.zeros(6)
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
        """Arms hold powered until the operator types SUPPORTED (deploy/recovery.py rule), then torque off; cameras
        off LAST. pyzed's Camera.close holds the GIL ~0.2 s (open ~1.7 s, measured 2026-11-01): with the arms
        powered, i2rt's motor thread stops sending and the DM motors trip their CAN watchdog (0xD, torque off)."""
        try:
            self._close_arms(read_line)
        finally:
            for c in self.cams.values():
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            self.cams = {}

    def _close_arms(self, read_line):
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
