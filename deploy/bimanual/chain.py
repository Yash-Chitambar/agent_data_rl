"""The bimanual chain on encoders and cameras only: capture -> pick -> hold -> transport -> approach -> insert ->
release, on a RealIO (the arms) or a SimIO (dry run, the same code path).

Sim code is shared, not copied:
  * hold, transport and approach ARE scripts/eval/eval_chain.py's Chain.stage_hold / stage_transport /
    stage_approach (borrowed as methods of RealChain, which duck-types the attributes they use: a, h, n, dev, dt,
    hstep, render, rel_pose, inhand, rt.comp, wrist / fine / inhand nets, R_sr0 / p_sr0, env.hold.cfg.yaws_deg);
  * the left hold is envs/left_hold.LeftHold on RealHarness (CPU scene chains + PD gains, believed angles from
    the encoders);
  * the pick is blind_insertion/pipeline/pick_servo.ServoPick (Part 1 PickExpert + wrist servo) on RealPickTask
    (PickTask's accumulate controller and grasp geometry, encoder-only state);
  * the ZED PC and reader estimators are perception/zed_estimator.ZedEstimator and pick_stage.ZedReader;
  * the insertion observation is checked against InsertionTask._actor_obs (deploy/checks/bimanual_obs.py).
Privileged sim state (rel_pose, inhand, PC pose, contacts) only feeds logs: NaN on the real robot, the sim truth
in a dry run.

Frames: io.read() is the motor (encoder) frame; believed = motor + rig.joint_offsets (the sim's
_measured_model_angles); every target is believed - offsets on the way out.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from blind_insertion.envs import friction_comp as FC, kinematics as K
from blind_insertion.envs.bimanual_task import BimanualTask
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask
from blind_insertion.envs.left_hold import Harness, LeftHold
from deploy.bimanual.guards import GuardStop
from deploy.bimanual.io_sim import eval_chain

SIDES = ('right', 'left')
STAGES = ('0 capture', '1 pick', '2 hold', '3 transport', '4 approach', '5 insert', '6 release', '7 home', 'manual', 'align')
EC = eval_chain()


class Privileged(RuntimeError):
    """A sim-only quantity was requested on the encoder-only path."""


class _Rt:
    """The runtime attributes the shared stage code reads (BimanualRuntime subset, from the CPU scene)."""

    def __init__(self, scene):
        self.scene = scene
        self.mjm, self.lay, self.indices = scene.mjm, scene.lay, scene.indices
        self._reader0, self._pc0, self.initial_q = scene._reader0, scene._pc0, scene.initial_q
        self.encoder_offsets, self.device = scene.encoder_offsets, scene.device
        self.comp = {'right': None, 'left': None}
        self.q = scene.initial_q[None].clone()          # finger entries follow the gripper readings
        self.cfg = SimpleNamespace(device=str(scene.device), nworld=1)


class RealHarness:
    """envs/left_hold.Harness interface on encoders: chains, PD gains, believed angles."""
    jac_point = Harness.jac_point

    def __init__(self, owner):
        self.owner, sc = owner, owner.scene
        self.rt, self.n, self.dev = owner.rt, 1, sc.device
        self.chain, self.kp = sc.chain, sc.kp
        self.enc_error = torch.zeros(1, 2, 6, device=sc.device)

    def believed(self, side):
        return self.owner.q_b[SIDES.index(side)][None].clone()

    def pc_pose(self):
        tr = self.owner.truth()
        if tr is None:
            return torch.full((1, 3), float('nan'), device=self.dev), torch.full((1, 3, 3), float('nan'), device=self.dev)
        return torch.tensor(tr['pc_p'], device=self.dev)[None], torch.tensor(tr['pc_R'], device=self.dev)[None]

    def contacts(self):
        if getattr(self.owner.io, 'is_sim', False):
            c = self.owner.io.h.contacts()
            return {k: v.to(self.dev) for k, v in c.items()}
        z = torch.zeros(1, dtype=torch.bool, device=self.dev)
        return {'arm_arm': z, 'left_other': z, 'left_pc_force': torch.zeros(1, 3, device=self.dev), 'right_pc': z}


class RealPickTask:
    """PickTask's controller interface (normalized accumulate actions, cap 0.1 rad, joint limits, grasp geometry)
    on the encoders. reader_pose() is the ZED estimate (set before the expert plans), never the truth."""
    grasp_flip = PickTask.grasp_flip
    _update_joint_targets = BimanualTask._update_joint_targets

    def __init__(self, owner, episode_s):
        sc = owner.scene
        self.owner, self.device, self.dt = owner, sc.device, owner.dt
        self.cfg = SimpleNamespace(physics=SimpleNamespace(nworld=1), action_scale_rad=.04, action_mode='accumulate',
                                   target_cap_rad=.1)
        self.max_steps = math.ceil(episode_s / self.dt)
        self.runtime, self.lay, self.chain = owner.rt, sc.lay, sc.tcp_chain
        self.offsets, self.pad_contact_height = sc.offsets, sc.pad_contact_height
        self.finger_qadr = sc.finger_qadr
        self.enc_error = torch.zeros(1, 2, 6, device=sc.device)
        self.joint_target = self._measured_model_angles().clone()
        self.previous_action = torch.zeros(1, 14, device=sc.device)
        self.failed = torch.zeros(1, dtype=torch.bool, device=sc.device)
        self.est = None

    def _measured_model_angles(self):
        return self.owner.q_b[None].clone()

    def reader_pose(self):
        if self.est is None:
            raise Privileged('reader pose: no estimate set')
        return self.est[0].clone(), self.est[1].clone()

    def q_true(self):
        raise Privileged('true joint angles')

    def tip_height(self, q, finger=None):
        """Lowest fingertip above the table (m) at believed angles q (1, 6): FlexChecker FK (fingers at 0), the chain's floor measure."""
        h = self.owner.floor_chk.heights(q[0].cpu().numpy().astype(float))
        return torch.tensor([min(h['left_finger'], h['right_finger']) - self.owner.scene.table_top], dtype=torch.float32, device=q.device)

    def pick_metrics(self):
        """Privileged in the sim (lift, tilt, in-hand pose): NaN here; the runner judges the grasp by the gripper."""
        nan = lambda *s: torch.full((1, *s), float('nan'), device=self.device)   # noqa: E731
        return {'lift_m': nan(), 'tilt_deg': nan(), 'inhand_p': nan(3), 'inhand_R': nan(3, 3), 'pc_pos': nan(3),
                'finite': torch.ones(1, dtype=torch.bool, device=self.device)}

    def tcp_pose(self):
        raise Privileged('true TCP pose')

    def step(self, action):
        a = torch.nan_to_num(torch.as_tensor(action, device=self.device, dtype=torch.float32)).clamp(-1, 1)
        jt = self._update_joint_targets(a, self._measured_model_angles())
        self.owner.hstep(jt[:, 0], jt[:, 1], '1 pick', grip_r=float(a[0, 6]), grip_l=float(a[0, 13]))
        self.previous_action.copy_(a)


class ChainHooks:
    """pipeline/grasp_v3 hooks on the chain (one world), real and dry run: gains, gripper, feed-forward, logs."""

    def __init__(self, chain, gc):
        self.ch, self.gc = chain, gc
        self.kp0, self.kd0 = chain.scene.kp[0].cpu().numpy().astype(float), chain.scene.kd[0].cpu().numpy().astype(float)
        self.alpha = 0.
        self.tgt = None
        self.g = chain.gmaps['right']

    def grip_reading(self):
        return torch.tensor([float(self.ch.state['grip'][0])], dtype=torch.float32, device=self.ch.dev)

    def wrist(self, alpha):
        a = float(alpha[0])
        if a == self.alpha:
            return
        if self.alpha == 0. or a == 0. or a == 1.:
            self.ch.log_event('grasp_v3', event='wrist_gains', alpha=a, tick=self.ch.k)
        self.alpha = a
        self.apply(a)

    def apply(self, a):
        kp, kd = self.kp0.copy(), self.kd0.copy()
        kp[3:6] += a * (np.asarray(self.gc.wrist_kp, float) - kp[3:6])
        kd[3:6] += a * (np.asarray(self.gc.wrist_kd, float) - kd[3:6])
        self.ch.io.set_gains('right', kp, kd)
        if self.ch.guards is not None:
            self.ch.guards.kp[0] = kp             # the torque guard follows the gains in force

    def grip(self, mode, u):
        m, ch, dt = int(mode[0]), self.ch, self.ch.dt
        io = ch.io
        if m == 0:
            self.tgt, ch._grip_target_right = None, None
            if hasattr(io, 'set_grip_position'):
                io.set_grip_position(False)
            return
        rd = float(ch.state['grip'][0])
        if self.tgt is None:
            self.tgt = min(ch.grip_cmd('right', 1.), rd)
        if m == 1:      # gentle: run down at a low rate, never further than gentle_delta below the reading
            self.tgt = min(self.tgt, max(self.tgt - self.gc.gentle_rate * dt, rd - self.gc.gentle_delta))
            self.t_gentle = self.tgt
        else:           # squeeze: linear from where the gentle close ended to the final target
            self.tgt = self.t_gentle + (self.g.close - self.t_gentle) * float(u[0])
        tgt = max(self.tgt, self.g.close)
        ch._grip_target_right = float(tgt)
        if hasattr(io, 'set_grip_position'):
            io.set_grip_position(True)

    def friction_ff(self, mask):
        self.ch.rt.comp['right'] = self.ch.fric if bool(mask[0]) else None

    def tick(self, ex):
        lg = self.ch.logger
        if lg is not None and int(ex.g[0]) != 0:
            lg.row('grasp', k=self.ch.k, g=int(ex.g[0]), attempt=int(ex.attempt[0]), alpha=float(ex.alpha[0]), drop=float(ex.det.drop[0]),
                   z_cmd=float(ex.p_cmd[0, 2]), reading=float(self.ch.state['grip'][0]),
                   grip_target=float(getattr(self.ch, '_grip_target_right', None) or np.nan))

    def restore(self):
        """Nominal gains, gripper command and feed-forward, whatever ended the pick."""
        if self.alpha != 0.:
            self.alpha = 0.
            self.apply(0.)
            self.ch.log_event('grasp_v3', event='wrist_gains', alpha=0., tick=self.ch.k, restored=True)
        self.ch._grip_target_right = None
        self.ch.rt.comp['right'] = None
        if hasattr(self.ch.io, 'set_grip_position'):
            self.ch.io.set_grip_position(False)


class _TrackedHold(LeftHold):
    """LeftHold that registers itself with the chain (force estimate and latch state for the logs and guards)."""
    owner = None

    def __init__(self, h, cfg=None):
        super().__init__(h, cfg)
        if _TrackedHold.owner is not None:
            _TrackedHold.owner.hold_obj = self


class _LogNet(torch.nn.Module):
    """Pass-through wrapper: runs a perception net on its own device (input moved there, output moved back to the
    input's device) and logs its outputs (x its scale) with the frame tag (owner None: no log)."""

    def __init__(self, net, scale, name, owner=None):
        super().__init__()
        self.net, self.name, self.owner = net, name, owner
        self.ndev = next(net.parameters()).device
        self.scale = scale.to(self.ndev)

    def forward(self, x):
        with torch.no_grad():
            y = self.net(x.to(self.ndev))
        if self.owner is not None:
            self.owner.log_event('net', net=self.name, frame_tag=self.owner.last_frame_tag, out=(y * self.scale).tolist())
        return y.to(x.device)


class _StopDeadband(torch.nn.Module):
    """approach_nojump.stop_deadband_mm (off by default): wraps the approach's wrist net (after _LogNet, so the log keeps
    the raw output). In '4 approach', from the first iteration at the stop (eval_chain's schedule: max(iters - 1, 1)
    coarse iterations, then final_iters at the stop), each port axis (across, up, gap) of the estimate within its
    deadband_mm (scalar or 3 values) of the stop target reads as the target: the servo asks no step on that axis (the
    rotation is already zero there: eval_chain final_pos_only). Elsewhere (align's use of ch.fine) it passes through.
    Why (docs/real/ROBUSTNESS_PLAN_20261113.md, part C): at the 18 mm stop the servo integrated 0.3-6 mm net corrections
    on its command (eval_chain plans from FK of the command) while joints 2/3 stuck in Coulomb friction; the command
    built up a lead (joint 2 error swing up to ~45 mrad) and the joint broke free: a 3.6-5.0 mm hand jump."""

    def __init__(self, net, scale, chain, deadband_mm, tgt_mm, n_coarse):
        super().__init__()
        db = np.asarray(deadband_mm, float).reshape(-1)
        self.net, self.ch, self.db = net, chain, torch.as_tensor(np.broadcast_to(db, (3,)).copy(), dtype=torch.float32)
        self.scale = scale
        self.tgt = torch.as_tensor(tgt_mm, dtype=torch.float32)
        self.n_coarse = int(n_coarse)

    def forward(self, x):
        y = self.net(x)
        ch = self.ch
        if ch.label != '4 approach':
            return y
        key = getattr(ch, '_stage_k0', None)          # one iteration count per approach, shared by the wrist and fine wrappers
        if getattr(ch, '_nj_key', None) != key:
            ch._nj_key, ch._nj_calls = key, 0
        it = ch._nj_calls
        ch._nj_calls += 1
        if it < self.n_coarse:
            return y
        sc = self.scale.to(y.device)
        est = y * sc
        tgt = self.tgt.to(y.device)
        err = est[0, :3] - tgt
        skip = err.abs() < self.db.to(y.device)          # per port axis (across, up, gap)
        ch.log_event('approach_deadband', iteration=it, est=est[0].tolist(), err_mm=err.tolist(), deadband_mm=self.db.tolist(),
                     skip=skip.tolist(), tick=ch.k)
        if not bool(skip.any()):
            return y
        out = y.clone()
        out[0, :3] = torch.where(skip, tgt / sc[:3], y[0, :3])   # an axis inside its band reads as the target: no step on it
        return out


class RealChain:
    # eval_chain stage code, shared verbatim
    site_target = EC.Chain.site_target
    _stage_hold = EC.Chain.stage_hold
    stage_transport = EC.Chain.stage_transport
    stage_approach = EC.Chain.stage_approach

    def __init__(self, cfg, io, scene, consts, gmaps, log=None, keys=None, guards=None, speed=1.0,
                 hold_force=None, device='cpu', policy_path=None, auto=None, net_device=None):
        from deploy.bimanual.io_sim import chain_argv
        self.cfg, self.io, self.scene, self.consts, self.gmaps = cfg, io, scene, consts, gmaps
        self.logger, self.keys, self.guards, self.auto = log, keys, guards, auto
        self.dev = torch.device(device)                                   # kinematics, controllers, policy
        self.ndev = torch.device(net_device or device)                    # perception nets
        self.n, self.dt, self.seed = 1, cfg['control_dt'], 0
        extra = []
        if hold_force is not None:
            extra = ['--hold-force', str(hold_force)]
        a = EC.parse(chain_argv(cfg, policy_path, 1, extra))
        self.speed = float(speed)
        if self.speed != 1.0:   # scripted motions slower (the sim used 1.0); policies run at their own rate
            a.transport_s = tuple(t / self.speed for t in a.transport_s)
            a.ramp_s, a.settle_s, a.grasp_settle_s = a.ramp_s / self.speed, a.settle_s / self.speed, a.grasp_settle_s / self.speed
        # approach_nojump (off unless the config has the block): chain_args.iters / final_iters were never passed to
        # eval_chain (chain_argv), so every real approach ran 6 iterations at the stop (eval_chain defaults) instead
        # of the 2 the configs ask for; this block applies them, plus a deadband at the stop (_StopDeadband)
        nj = cfg.get('approach_nojump') or {}
        self.nojump = nj if nj.get('enabled', False) else None
        if self.nojump is not None:
            a.iters = int(nj.get('iters', a.iters))
            a.final_iters = int(nj.get('final_iters', a.final_iters))
            if a.final_pos_only > a.final_iters:
                a.final_pos_only = a.final_iters
            self.log_event('approach_nojump', iters=a.iters, final_iters=a.final_iters, final_pos_only=a.final_pos_only,
                           stop_deadband_mm=nj.get('stop_deadband_mm'))
        self.a = a
        self.rt = _Rt(scene)
        self.h = RealHarness(self)
        self.env = SimpleNamespace(hold=SimpleNamespace(cfg=SimpleNamespace(yaws_deg=tuple(consts['hold_yaws_deg']))))
        self.off = np.asarray(scene.encoder_offsets.cpu().numpy(), float)
        if not np.allclose(self.off, io.off, atol=1e-9):
            raise RuntimeError(f'IO joint offsets {io.off} differ from the scene calibration {self.off}')
        # nets (eval_chain.wrist_loader reads runs/perception/<name>.pt = links into the verified bundle)
        from blind_insertion.envs.inhand import measured_mean
        from blind_insertion.eval_insertion import load_insertion_policy
        from deploy.bimanual import netcache as NC     # loaded before the arms connect (Runner.open_io)
        self.zed = NC.zed(self.ndev, a.zed_gate)
        # ZED case-edge fit in the capture stage (config block zed_edge_refine; off when absent or enabled is false):
        # deploy/bimanual/zed_edgefit.py, docs/real/ZED_EDGEFIT_20261115.md
        from deploy.bimanual import zed_edgefit as ZE
        self.edge = ZE.parse(cfg)
        self.zed_start = None
        if self.edge is not None:
            if self.edge['start_gate']:
                self.zed_start = NC.zed(self.ndev, self.edge['start_gate'])
            self.log_event('zed_edge_refine', **{k: v for k, v in self.edge.items()})
        cap = json.loads((EC.ROOT / 'blind_insertion/configs/zed_capture_pose.json').read_text())['model_frame_rad']
        self.zed_q = {s: np.asarray(cap[s], float) for s in SIDES}
        w, f, ih = (NC.wrist(x, self.ndev) for x in (a.wrist_net, a.fine_net, a.inhand_net))
        self.wrist = (_LogNet(w[0], w[1], a.wrist_net, self), w[1].to(self.dev), w[2])
        self.fine = (_LogNet(f[0], f[1], a.fine_net, self), f[1].to(self.dev), f[2])
        self.inhand_net = (_LogNet(ih[0], ih[1], a.inhand_net, self), ih[1].to(self.dev), ih[2])
        if self.nojump is not None and self.nojump.get('stop_deadband_mm'):
            tgt = (a.stop_offset_mm[0], a.stop_offset_mm[1], a.stop + a.stop_offset_mm[2])   # eval_chain's tgt_p at the stop (mm)
            n_coarse = max(a.iters - 1, 1)
            self.wrist = (_StopDeadband(self.wrist[0], self.wrist[1], self, self.nojump['stop_deadband_mm'], tgt, n_coarse),) + self.wrist[1:]
            self.fine = (_StopDeadband(self.fine[0], self.fine[1], self, self.nojump['stop_deadband_mm'], tgt, n_coarse),) + self.fine[1:]
        self.R_sr0, self.p_sr0 = measured_mean(device=self.dev)
        self.pol = load_insertion_policy(str(policy_path), str(self.dev))
        self.fric = FC.parse(cfg['fric_comp_right'], side='right')
        self.kp = scene.kp.cpu().numpy().astype(float)
        self.hold_obj, self.pressing, self.k = None, False, 0
        self.label, self.last_frame_tag = 'manual', None
        self.stream_every = max(1, int(round(1.0 / (cfg['cameras']['stream_hz'] * self.dt))))
        # right fingertip floor (deploy/runner.py, try1 2026-10-28): a target pressing the tips below SafeArm's limit
        # + 2 mm is lifted to it (the sim fingers just press on the table; SafeArm would stop the run)
        from deploy.safety_task import FlexChecker
        sc_ = cfg['safety']
        self.floor_chk = FlexChecker(np.zeros(6), sc_['clearance'], sc_['tip_clearance'], sc_.get('body_clearance'), sc_['table_z'])
        self.floor = sc_['table_z'] + sc_['tip_clearance'] + 0.002
        self.floor_lifts = 0
        self.read_state()
        self.q_start = self.state['q'].copy()

    # ------------------------------------------------------------------ io and logs
    def log_event(self, kind, **kw):
        if self.logger is not None:
            self.logger.event(kind, **kw)
        live = getattr(self, 'live', None)
        if live is not None:
            try:
                live.event(kind, kw)
            except Exception:  # noqa: BLE001  (the viewer state never stops a run)
                pass

    def truth(self):
        return self.io.truth() if getattr(self.io, 'is_sim', False) else None

    def read_state(self):
        s = self.io.read()
        self.state = s
        self.q_b = torch.as_tensor(s['q'] + self.off, dtype=torch.float32, device=self.dev)       # (2, 6) believed
        fm = [self.gmaps[sd].finger_m(s['grip'][i]) for i, sd in enumerate(SIDES)]
        self.finger = np.asarray(fm, float)
        self.rt.q[0, self.scene.finger_qadr] = float(fm[0])
        return s

    def _close_press_relief(self, label, tb, grip_r):
        """pick.close_press_relief (real runs 2026-11-01): the servo pick closes while its right target still presses
        ~16-30 mm into the table (accumulate PD). On the real arm the pressed fingertips rotate the hand ~3.5 deg away
        from the planned orientation (sim ~0.5 deg; pick4-9), so the pads are not parallel to the reader's sides and
        ride over it. Without any press (relief v2, pick9) the squeezing pads push the hand UP and still ride over.
        From the open -> close switch of the right gripper in '1 pick':
          hold  = measured TCP position + lift_mm (negative: a light press), PLANNED orientation;
          jaws  = open for settle_s, then ramp to the close target over close_ramp_s, then dwell_s;
          after = blend from the hold to the plan's target over blend_s, then follow the plan.
        Config: true or {"lift_mm", "settle_s", "close_ramp_s", "dwell_s", "blend_s", "orient": "plan"|"measured"}.
        Returns (targets, grip_r); a ramped close sets self._grip_override_right (an i2rt reading)."""
        rc = self.cfg['pick'].get('close_press_relief')
        if not rc or label != '1 pick':
            self._relief = None
            self._prev_grip_r = None
            self._grip_override_right = None
            return tb, grip_r
        rc = rc if isinstance(rc, dict) else {}
        prev, self._prev_grip_r = getattr(self, '_prev_grip_r', None), grip_r
        ch_ = self.scene.chain['right']
        dev = self.scene.device
        fk = lambda q: ch_.fk(torch.as_tensor(q, dtype=torch.float32)[None].to(dev))  # noqa: E731
        n = lambda key, dflt: int(round(float(rc.get(key, dflt)) / self.dt))  # noqa: E731
        r = getattr(self, '_relief', None)
        if r is None:
            if not (prev is not None and prev > 0 and grip_r < 0):
                return tb, grip_r
            qm = self.q_b[0].cpu().numpy().astype(float).copy()
            pm, Rm = fk(qm)
            pt, Rt = fk(tb[0])
            R = Rt if rc.get('orient', 'plan') == 'plan' else Rm
            p = pm + torch.tensor([[0., 0., float(rc.get('lift_mm', 1.0)) * 1e-3]], device=dev)
            q, _, _ = ch_.ik(p, R, torch.as_tensor(qm, dtype=torch.float32)[None].to(dev), iters=20)
            q = q[0].cpu().numpy().astype(float)
            from scipy.spatial.transform import Rotation as _Rot
            dth = float(np.degrees(np.linalg.norm(_Rot.from_matrix((Rm[0].T @ Rt[0]).cpu().numpy()).as_rotvec())))
            k0 = self.k
            t_close = k0 + n('settle_s', 0.3)
            t_closed = t_close + n('close_ramp_s', 0.0)
            t_blend = t_closed + n('dwell_s', 0.0)
            g_open = self.grip_cmd('right', 1.)
            r = self._relief = dict(q=q, close=t_close, closed=t_closed, blend=t_blend, nb=max(1, n('blend_s', 0.0)),
                                    g_open=g_open, phase='hold')
            self.log_event('close_press_relief', tick=self.k, state='on', press_mm=float(pm[0, 2] - pt[0, 2]) * 1e3,
                           orient_fix_deg=dth, **{k: rc[k] for k in rc if k != 'orient'})
        k = self.k
        if r['phase'] == 'hold':
            plan_q = tb[0].copy()             # the expert's target this tick (its lift starts when it rises above the hold)
            tb = tb.copy()
            tb[0] = r['q']
            if k < r['close']:
                grip_r = 1.
                self._grip_override_right = None
            elif k < r['closed']:
                u = (k - r['close'] + 1) / max(1, r['closed'] - r['close'])
                self._grip_override_right = float(r['g_open'] + (self.gmaps['right'].close - r['g_open']) * u)
            else:
                self._grip_override_right = None      # sim close target from here
            if k >= r['blend'] and float(fk(plan_q)[0][0, 2]) >= float(fk(r['q'])[0][0, 2]):
                r['phase'], r['b0'] = 'blend', k
                self.log_event('close_press_relief', tick=k, state='blend')
        elif r['phase'] == 'blend':
            u = min(1., (k - r['b0'] + 1) / r['nb'])
            plan = tb[0].copy()
            tb = tb.copy()
            tb[0] = r['q'] + (plan - r['q']) * u
            grip_r = -1.
            self._grip_override_right = None
            if u >= 1.:
                self.log_event('close_press_relief', tick=k, state='off')
                r['phase'] = 'done'
        else:
            grip_r = -1. if grip_r < 0 else grip_r
        return tb, grip_r

    def grip_cmd(self, side, sign):
        """Sim gripper sign -> i2rt target: open = the sim's finger limit (right: the 24 mm approach opening),
        close = the squeeze target."""
        g = self.gmaps[side]
        if side == 'right' and getattr(self, '_grip_target_right', None) is not None:
            return float(np.clip(self._grip_target_right, g.close, g.open))   # guarded grasp: gentle close, squeeze ramp
        if side == 'right' and getattr(self, '_grip_override_right', None) is not None:
            return float(np.clip(self._grip_override_right, g.close, g.open))   # _close_press_relief ramp
        if sign > 0:
            w = self.cfg['gripper']['right_approach_open_m'] if side == 'right' else self.cfg['gripper']['full_open_m']
            if side == 'right' and getattr(self, 'right_open_override_m', None) is not None:
                w = self.right_open_override_m        # retreat_right: insert.release_open_m
            return float(np.clip(g.reading(w), g.close, g.open))
        return float(g.close)

    def _precision_stage(self, label):
        """cfg arm_control.<side>.active_stages: the controller and the fitted gravity only in these stages; every other
        stage runs as before the controller existed (2026-11-06: on in the pick, the hand went low onto the table).
        freeze_stages: hold the integral (torque kept, no integration). Real I/O only."""
        f = getattr(self.io, 'precision_freeze', None)
        act = getattr(self.io, 'precision_active', None)
        if f is None:
            return
        for side in ('right', 'left'):                    # the block also holds notes (_about): read the arms only
            a = (self.cfg.get('arm_control') or {}).get(side)
            if isinstance(a, dict) and a.get('precision') is not None:
                if 'active_stages' in a and act is not None:
                    act(side, label in a['active_stages'])
                f(side, label in (a.get('freeze_stages') or []))

    def hstep(self, qr, ql, label, grip_r=-1., grip_l=-1.):
        """One 20 ms control tick with absolute believed-frame targets for both arms (Harness.step on the robot)."""
        if label != self.label:
            self.log_event('stage', stage=label, prev=self.label, tick=self.k)
            self.label = label
            self._stage_k0 = self.k
            self._precision_stage(label)
        tgt = torch.stack([torch.as_tensor(qr, device=self.dev).reshape(6), torch.as_tensor(ql, device=self.dev).reshape(6)]).float()
        if not torch.isfinite(tgt).all():
            raise GuardStop(f'non-finite target at {label}')
        if self.guards is not None:
            pr, _ = self.scene.chain['right'].fk(tgt[0:1].to(self.scene.device))
            pl, _ = self.scene.chain['left'].fk(tgt[1:2].to(self.scene.device))
            self.guards.check_target(pr[0].cpu().numpy(), pl[0].cpu().numpy())
        tb = tgt.cpu().numpy().astype(float)
        tb, grip_r = self._close_press_relief(label, tb, grip_r)
        low = np.inf
        if self.guards is None or pr[0, 2] < self.scene.table_top + 0.08:     # fingertip heights only near the table
            h = self.floor_chk.heights(tb[0])
            low = min(h['left_finger'], h['right_finger'])
        if low < self.floor:
            ch_ = self.scene.chain['right']
            p_, R_ = ch_.fk(torch.as_tensor(tb[0], dtype=torch.float32)[None].to(self.scene.device))
            q_, _, _ = ch_.ik(p_ + torch.tensor([[0., 0., self.floor - low + 1e-4]], device=p_.device), R_,
                              torch.as_tensor(tb[0], dtype=torch.float32)[None].to(self.scene.device), iters=8)
            self.log_event('floor_lift', tick=self.k, stage=label, lift_mm=(self.floor - low) * 1e3)
            tb = tb.copy()
            tb[0] = q_[0].cpu().numpy()
            self.floor_lifts += 1
        motor = tb - self.off
        grip = np.array([self.grip_cmd('right', grip_r), self.grip_cmd('left', grip_l)])
        ff_cfg = self.rt.comp['right']
        t_send = time.perf_counter()
        compute = t_send - getattr(self, '_t_read', t_send)
        sent, ff = self.io.send(motor, grip, ff_cfg)
        late = self.io.wait_tick()
        s = self.read_state()
        self._t_read = time.perf_counter()
        self.compute_ms = compute * 1e3
        tau = None
        if self.guards is not None:
            tau = self.guards.check_state(tb, s['q'] + self.off, s['qd'], ff, pressing=self.pressing)
        self._log_tick(label, tb, motor, grip, sent, ff, tau, late)
        self.k += 1

    def _log_tick(self, label, tb, motor, grip, sent, ff, tau, late):
        if self.logger is None:
            return
        s = self.state
        self.logger.row('ticks', k=self.k, t=s['t'], t_obs=s['t_obs'], stage=STAGES.index(label) if label in STAGES else -1,
                        q=s['q'], qd=s['qd'], eff=s['eff'], grip=s['grip'], finger_m=self.finger,
                        target_b=tb, target_motor=motor, sent=sent, grip_cmd=grip, ff=ff,
                        tau_est=tau if tau is not None else np.full((2, 6), np.nan), late_s=late,
                        compute_ms=self.compute_ms)
        for side, rows in self.io.drain_can().items():
            for t, q7 in rows:
                self.logger.row('can_' + side, t=t, q7=np.asarray(q7, float))
        if self.hold_obj is not None and self.hold_obj.plan is not None:
            hb = self.hold_obj
            F = hb.force_estimate(torch.as_tensor(tb[1], dtype=torch.float32, device=self.dev)[None])
            self.logger.row('hold', k=self.k, F_est=F[0], Fn_est=float((F * hb.plan['normal']).sum(1)[0]),
                            latched=bool(hb.latched[0]), pressing=self.pressing)
        if not getattr(self.io, 'is_sim', False) and self.k % self.stream_every == 0:
            try:
                fr = self.io.frames(tuple(self.io.cams))
                for n, (pol, raw, t, cnt) in fr.items():
                    self.logger.frame(n, raw, 'stream', t_capture=t, t_use=time.monotonic(), kind='raw', bgr=True, count=cnt)
            except RuntimeError:
                pass

    def tick(self, label):
        pass

    def render(self, names=('top', 'wrist_right'), tag=None, fresh=True):
        """Newest frames as the sim renders them: {'top', 'wrist_cam'}: (1, 320, 320, 3) uint8 on dev. Every frame
        used is logged (policy image PNG = the exact net input; raw JPEG)."""
        tag = tag or self.label.replace(' ', '_')
        fr = (self.io.fresh_frames if fresh else self.io.frames)(names)
        self.last_raw = {n: v[1] for n, v in fr.items()}      # raw BGR frames (None in the dry run): the capture edge fit
        out = {}
        for n, (pol, raw, t, cnt) in fr.items():
            key = {'top': 'top', 'wrist_right': 'wrist_cam', 'wrist_left': 'left_wrist_cam'}[n]
            out[key] = torch.as_tensor(np.ascontiguousarray(pol), device=self.dev)[None]
            if self.logger is not None:
                self.logger.frame(n, pol, tag, t_capture=t, t_use=self.state['t'], kind='policy', count=cnt, tick=self.k)
                if raw is not None:
                    self.logger.frame(n, raw, tag, t_capture=t, t_use=self.state['t'], kind='raw', bgr=True, count=cnt, tick=self.k)
        self.last_frame_tag = f'{tag}@{self.k}'
        tr = self.truth()
        if tr is not None:      # dry run: sim truth at the frame (checks the exporter's FK labels)
            self.log_event('frame_truth', frame_tag=self.last_frame_tag, plug_in_port_mm=tr['plug_in_port_mm'],
                           plug_rot_deg=tr['plug_rot_deg'], inhand_p=tr['inhand_p'], inhand_R=tr['inhand_R'])
        return out

    def rel_pose(self):
        tr = self.truth()
        if tr is None:
            nan = torch.full((1, 3), float('nan'), device=self.dev)
            return nan, nan.clone()
        return (torch.tensor(tr['plug_in_port_mm'], device=self.dev)[None] * 1e-3,
                torch.deg2rad(torch.tensor(tr['plug_rot_deg'], device=self.dev))[None])

    def inhand(self):
        tr = self.truth()
        if tr is None:
            return torch.full((1, 3), float('nan'), device=self.dev), torch.full((1, 3, 3), float('nan'), device=self.dev)
        return torch.tensor(tr['inhand_p'], device=self.dev)[None], torch.tensor(tr['inhand_R'], device=self.dev)[None]

    def ask(self, text, default='', confirm=False):
        """confirm: a prompt where a person must act (recovery after a stop, holding the reader): asked even with --yes/--demo."""
        if self.keys is None:
            return default if self.auto is None else self.auto
        return self.keys.ask(text, auto=None if confirm else self.auto)

    # ------------------------------------------------------------------ poses
    def motor(self, side, q_believed):
        return np.asarray(q_believed, float) - self.off[SIDES.index(side)]

    def move(self, goals_b, grip=None, label='manual'):
        """Slow checked moves to believed-frame goals {side: q6}; grip {side: i2rt}."""
        if label != self.label:
            self.log_event('stage', stage=label, prev=self.label, tick=self.k)
            self.label = label
            self._precision_stage(label)      # home and capture move here, not through hstep (2026-11-06 align_full7)
        # peak joint speed of these smoothstep moves (capture, home, park); 0.15 rad/s unless the config sets
        # safety.move_max_vel_rad_s (2026-11-11: about 95 s of each run was these moves at 0.15)
        vmax = float(self.cfg['safety'].get('move_max_vel_rad_s', 0.15))
        self.io.move_to({s: self.motor(s, q) for s, q in goals_b.items()}, grip, max_vel=vmax)
        self.read_state()
        self.log_event('moved', goals_b={s: np.asarray(q).tolist() for s, q in goals_b.items()},
                       reached_b=(self.state['q'] + self.off).tolist())

    def arms_believed(self):
        """(q_left, q_right, right finger opening per finger m) for the estimators' arm-pose gates."""
        return (self.q_b[1][None].clone(), self.q_b[0][None].clone(),
                torch.tensor([float(self.finger[0])], device=self.dev))

    # ------------------------------------------------------------------ stage 0: captures
    def capture_pc(self, log):
        """Both arms to the ZED capture pose (configs/zed_capture_pose.json), grippers fully open; ZED frame ->
        port pose (ZedEstimator + presence/OOD + arm-pose gate)."""
        full = {s: self.gmaps[s].reading(self.cfg['gripper']['full_open_m']) for s in SIDES}
        full = {s: float(np.clip(v, self.gmaps[s].close, self.gmaps[s].open)) for s, v in full.items()}
        self.move({'right': self.zed_q['right'], 'left': self.zed_q['left']}, grip=full, label='0 capture')
        cams = ('top',) if getattr(self.io, 'is_sim', False) else tuple(n for n in ('top', 'wrist_right', 'wrist_left') if n in self.io.cams)
        top = self.render(cams, tag='zed_pc')['top']
        arms = self.arms_believed()
        xy, yaw, ok, info = self.zed(top.to(self.ndev), arms=tuple(x.to(self.ndev) for x in arms))
        xy, yaw, ok = xy.to(self.dev), yaw.to(self.dev), ok.to(self.dev)
        ep, eR = self.scene.port_frame(xy.cpu(), yaw.cpu())
        ep, eR = ep.to(self.dev), eR.to(self.dev)
        rec = dict(port_xy=xy[0].tolist(), yaw_deg=math.degrees(float(yaw[0])), ok=bool(ok[0]),
                   **{k: float(v[0]) for k, v in info.items()}, arms_b=[x[0].tolist() for x in arms])
        tr = self.truth()
        if tr is not None:
            yt = math.atan2(-tr['pc_R'][1, 2], -tr['pc_R'][0, 2])
            rec.update(true_port_xy=tr['pc_p'][:2].tolist(), true_yaw_deg=math.degrees(yt),
                       err_xy_mm=float(np.linalg.norm(np.asarray(rec['port_xy']) - tr['pc_p'][:2]) * 1e3),
                       err_z_mm=float((float(ep[0, 2]) - tr['pc_p'][2]) * 1e3))
        if self.edge is not None:
            xy, yaw, ok, ep, eR = self._edge_refine(top, arms, xy, yaw, ok, ep, eR, rec, tr)
        self.log_event('estimate', est='zed_pc', frame_tag=self.last_frame_tag, **rec)
        log.update(zed=rec)
        self.port = (ep, eR)
        return ep, eR, bool(ok[0])

    def _edge_refine(self, top, arms, xy, yaw, ok, ep, eR, rec, tr):
        """Capture edge fit (deploy/bimanual/zed_edgefit.py). Adds rec['net'] (the gate net, as today), rec['start'] and
        rec['edge_fit']. Mode 'use': port_xy / yaw_deg / ok in rec and the returned port pose are the refined ones
        (ok = net gate AND fit gate); mode 'shadow': unchanged."""
        from deploy.bimanual import zed_edgefit as ZE
        c = self.edge
        net = {k: rec[k] for k in ('port_xy', 'yaw_deg', 'ok')}
        if self.zed_start is not None:
            sxy, syaw, sok, sinfo = self.zed_start(top.to(self.ndev), arms=tuple(x.to(self.ndev) for x in arms))
            start = dict(port_xy=sxy[0].tolist(), yaw_deg=math.degrees(float(syaw[0])), ok=bool(sok[0]),
                         **{k: float(v[0]) for k, v in sinfo.items()})
        else:
            start = dict(net)
        model = None
        if getattr(self.io, 'is_sim', False):
            img, src, geom = ZE.canvas_from_policy(top[0].cpu().numpy()), 'sim_policy_upsampled', c['geom_sim']
            if c['dry_run_fit'] == 'truth_model' and tr is not None:     # dry run only (zed_edgefit.DEFAULTS)
                rng = np.random.default_rng(int(self.seed) + 15)
                model = (*(tr['pc_p'][:2] + rng.normal(0., c['dry_run_fit_sd_mm'] * 1e-3, 2)),
                         rec['true_yaw_deg'] + rng.normal(0., c['dry_run_fit_sd_deg']))
        else:
            img, src, geom = (getattr(self, 'last_raw', {}) or {}).get('top'), 'raw_hd1080', c['geom']
        fxy, fyaw, fok, frec = ZE.refine(img, src, start['port_xy'], start['yaw_deg'], net['port_xy'], net['yaw_deg'], c, geom,
                                         model_pose=model)
        print(ZE.summary_line(frec), flush=True)
        if tr is not None:
            for k_, v_ in (('net', net), ('start', start)):
                v_['err_xy_mm'] = float(np.linalg.norm(np.asarray(v_['port_xy']) - tr['pc_p'][:2]) * 1e3)
            if 'fit_xy' in frec:
                frec['err_xy_mm'] = float(np.linalg.norm(np.asarray(frec['fit_xy']) - tr['pc_p'][:2]) * 1e3)
                frec['err_mm'] = ((np.asarray(frec['fit_xy']) - tr['pc_p'][:2]) * 1e3).tolist()
                frec['err_yaw_deg'] = frec['fit_yaw_deg'] - rec['true_yaw_deg']
        rec.update(net=net, start=start, edge_fit=frec, edge_mode=c['mode'])
        if c['mode'] != 'use':
            rec['est_source'] = 'net (edge fit shadow)'
            return xy, yaw, ok, ep, eR
        ok_all = bool(ok[0]) and fok
        xy = torch.tensor([[float(fxy[0]), float(fxy[1])]], dtype=xy.dtype, device=self.dev)
        yaw = torch.tensor([math.radians(fyaw)], dtype=yaw.dtype, device=self.dev)
        ok = torch.tensor([ok_all], device=self.dev)
        ep, eR = self.scene.port_frame(xy.cpu(), yaw.cpu())
        ep, eR = ep.to(self.dev), eR.to(self.dev)
        rec.update(port_xy=xy[0].tolist(), yaw_deg=float(fyaw), ok=ok_all, est_source=f"edge fit x, y + {c['yaw_source']} yaw")
        if tr is not None:
            rec['err_xy_mm'] = float(np.linalg.norm(np.asarray(rec['port_xy']) - tr['pc_p'][:2]) * 1e3)
        return xy, yaw, ok, ep, eR

    def park_left(self):
        """Part 1 park (PickTaskCfg.left_park_q, joint 1 = 0) and one extra ZED frame (the sim chain's 'parked' frame,
        kept so the dry run's camera-noise draws line up with eval_chain)."""
        from blind_insertion.envs.bimanual_pick import PickTaskCfg
        self.move({'left': np.asarray(PickTaskCfg().left_park_q, float)}, label='0 capture')
        self.render(('top',), tag='zed_parked')

    def reader_frame(self):
        """pick_stage.ZedReader's frame: left parked, right at the capture pose, right gripper fully open."""
        full = float(np.clip(self.gmaps['right'].reading(self.cfg['gripper']['full_open_m']), 0, 1))
        if getattr(self.io, 'is_sim', False):
            self.io.move_to({}, {'right': full})          # eval_chain.reader_frame: fingers fully open for the frame
        elif abs(self.state['grip'][0] - full) > 0.05:
            self.io.set_grip('right', full)
        self.read_state()
        top = self.render(('top',), tag='zed_reader')['top']
        return top, self.arms_believed()

    # ------------------------------------------------------------------ stage 1: pick
    def make_pick(self):
        from blind_insertion.pipeline.pick_servo import ServoPick
        from deploy.bimanual import netcache as NC
        pc = self.cfg['pick']
        zr = NC.zed_reader(pc['reader_estimator'], self.dev, pc['reader_retries'])
        chain = self

        def estimator(ch):
            p, R, ok = zr(ch)
            ch.pick.est = (p, R)
            info = getattr(ch, 'reader_est_info', {})
            rec = dict(p=p[0].tolist(), yaw_deg=math.degrees(float(torch.atan2(-R[0, 1, 2], -R[0, 0, 2]))), ok=bool(ok[0]),
                       tries=float(info.get('tries', torch.ones(1))[0]))
            tr = chain.truth()
            if tr is not None:
                rp = torch.tensor(tr_reader(chain)[:3])
                rec['err_mm'] = float((p[0].cpu() - rp).norm() * 1e3)
            chain.log_event('estimate', est='zed_reader', frame_tag=chain.last_frame_tag, **rec)
            return p, R, ok

        def on_frame(kind, fr, delta, ex):
            chain.log_event('servo_frame', frame_kind=kind, frame_tag=chain.last_frame_tag, delta=delta[0].tolist(),
                            phase=int(ex.phase[0]), servo_k=int(ex.servo_k[0]), hover_iters=int(ex.hover_iters[0]))
        def dn(name):       # servo net on the net device, called with chain-device frames
            net, sc, k = NC.servo(name, self.ndev)
            return _LogNet(net, sc, name), sc.to(self.dev), k
        hv, nr = dn(pc['hover_net']), dn(pc['near_net'])
        rt_ = dn(pc['rot_net']) if pc.get('rot_net') else None
        make_expert = None
        self.grasp_hooks = None
        if pc.get('grasp_v3'):
            from blind_insertion.pipeline.grasp_v3 import GraspV3Cfg, GraspV3Expert
            if pc.get('close_press_relief'):
                raise ValueError('pick.grasp_v3 replaces pick.close_press_relief: use one')
            gd = dict(pc['grasp_v3'])
            if getattr(self, 'pick_test', None):
                gd['test_mode'] = self.pick_test
            gc = GraspV3Cfg.from_dict(gd)
            self.grasp_hooks = ChainHooks(self, gc)
            self.log_event('grasp_v3', event='config', tick=self.k, **gc.to_dict())

            def ev(kind, world, **kw):
                tr = self.truth()
                if tr is not None:      # dry run only: true roll of the closing axis (the encoder-offset DR hides it from FK)
                    Rt = self.scene.chain['right'].fk(torch.as_tensor(tr['q_true'][0], dtype=torch.float32)[None].to(self.scene.device))[1]
                    kw['roll_true_deg'] = math.degrees(math.asin(float(Rt[0, 2, 1])))
                    h_ = self.floor_chk.heights(np.asarray(tr['q_true'][0], float))
                    kw['tip_true_mm'] = (min(h_['left_finger'], h_['right_finger']) - self.scene.table_top) * 1e3
                self.log_event('grasp_v3', event=kind, **kw)
                print(f'grasp: {kind} ' + ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in kw.items()), flush=True)

            def make_expert(task, **kw):
                return GraspV3Expert(task, cfg=gc, hooks=self.grasp_hooks, on_event=ev, **kw)
        return ServoPick(estimator, hv, nr, rt_, fracs=pc['fracs'], rehover_mm=pc['rehover_mm'], rehover_max=pc['rehover_max'],
                         settle_max_s=pc['settle_max_s'], hold_s=pc['hold_s'], v_max=.10 * self.speed,
                         joint_speed=1.2 * self.speed, on_frame=on_frame, make_expert=make_expert)

    def stage_pick(self, log, mode='servo'):
        """Servo-descent pick from the capture pose. mode 'expert': pick_stage.ExpertPick with the ZED reader
        estimate and no servo (eval_chain's default pick; used for the dry-run equivalence check)."""
        self.pick = RealPickTask(self, self.cfg['pick']['episode_s'] if mode == 'servo' else 13.)
        if getattr(self.io, 'is_sim', False):   # dry run: the grasp model's servo error reaches the expert (None on the robot)
            self.pick.runtime.fidelity = self.io.rt.fidelity
        if mode == 'servo':
            stage = self.make_pick()
        else:
            from blind_insertion.pipeline.pick_stage import ExpertPick
            from deploy.bimanual import netcache as NC
            zr = NC.zed_reader(self.cfg['pick']['reader_estimator'], self.dev, self.cfg['pick']['reader_retries'])

            def est(ch):
                p, R, ok = zr(ch)
                ch.pick.est = (p, R)
                return p, R, ok
            stage = ExpertPick(est)
            self.pick.max_steps = math.ceil(13. / self.dt)
        self.label = '1 pick'
        try:
            res = stage.run(self)
        finally:
            if getattr(self, 'grasp_hooks', None) is not None:
                self.grasp_hooks.restore()
        ok = bool(res['done'][0]) and not bool(res['failed'][0])
        present = self.reader_in_hand()
        if bool(getattr(getattr(stage, 'expert', None), 'gave_up', torch.zeros(1, dtype=torch.bool))[0]):
            present = False       # open jaws read inside the wide band
        ex_ = getattr(stage, 'expert', None)
        if hasattr(ex_, 'outcome'):
            out_ = ex_.outcome()
            log.update(pick_outcome=out_, pick_attempts=int(ex_.attempt[0]))
            self.log_event('pick_outcome', outcome=out_, attempts=int(ex_.attempt[0]), batch=self.cfg['pick'].get('batch'),
                           open_m=self.cfg['gripper']['right_approach_open_m'], grasp_v3=self.cfg['pick'].get('grasp_v3'))
            print(f'pick outcome: {out_} ({int(ex_.attempt[0])} attempt(s))', flush=True)
        log.update(pick_done=bool(res['done'][0]), pick_failed=bool(res['failed'][0]), pick_reader_present=present,
                   pick_est_ok=bool(res.get('est_ok', torch.ones(1))[0]))
        self.log_event('pick_result', ok=ok, present=present, **{k: (v[0].tolist() if torch.is_tensor(v) else v) for k, v in res.items()})
        return self.pick.joint_target[:, 0].clone(), ok and present is not False

    def reader_in_hand(self):
        """Gripper reading between the empty-close and reader bands (deploy/grasp.reader_present), if calibrated."""
        g = self.cfg['gripper']['right']
        if g.get('g_empty') is None or g.get('g_reader') is None:
            return None
        from deploy.grasp import reader_present
        return bool(reader_present(float(self.state['grip'][0]), g['g_empty'], g['g_reader']))

    # ------------------------------------------------------------------ stage 2: hold (eval_chain.Chain.stage_hold)
    def stage_hold(self, qr, ep, eR, log):
        _TrackedHold.owner = self
        spd, HC = self.speed, EC.HoldCfg

        def hold_cfg(**kw):
            # the park -> stand-off swing (HoldCfg.move_s 1.5 s at speed 1) is scripted motion: scale it by --speed
            # like pick/transport/approach (real run 2026-11-01 hold_160051: 1.5 s gave 2.1 rad/s on left joint 4).
            # The guarded descent (1 cm/s) and the press are unchanged.
            c = HC(**kw)
            c.move_s = c.move_s / spd
            return c
        try:
            with _patch(EC, 'LeftHold', _TrackedHold), _patch(EC, 'HoldCfg', hold_cfg):
                self.pressing = True
                q_hold = self._stage_hold(qr, ep, eR, log)
        finally:
            _TrackedHold.owner = None
        hb = self.hold_obj
        F = hb.force_estimate(q_hold)
        self.log_event('hold_result', latched=bool(hb.latched[0]), plan_ok=bool(hb.plan['ok'][0]),
                       yaw_deg=float(hb.plan['yaw_deg'][0]), dq_force=hb.plan['dq_force'][0].tolist(),
                       F_est=F[0].tolist(), q_latch=hb.q_latch[0].tolist(), t_latch=float(hb.t_latch[0]),
                       point=hb.plan['point'][0].tolist(), normal=hb.plan['normal'][0].tolist(),
                       q_contact=hb.plan['q_contact'][0].tolist(), force_cmd=hb.plan['force_cmd'][0].tolist())
        return q_hold

    def unload_left(self, steps=25):
        """Remove the press force: command the measured left pose (zero PD error), then lift the hand."""
        ql = self.q_b[1].clone()
        qr = self.q_b[0].clone()
        for _ in range(steps):
            self.hstep(qr, ql, '6 release')
        self.pressing = False
        lift = self.cfg['insert']['left_lift_m']
        ch = self.scene.chain['left']
        p0, R0 = ch.fk(ql[None])
        q = ql[None]
        n = max(1, int(round(lift / 0.02 / self.dt / self.speed)))
        for k in range(n):
            pt = p0 + torch.tensor([[0., 0., lift * (k + 1) / n]], device=self.dev)
            q, _, _ = ch.ik(pt, R0, q, iters=10)
            self.hstep(qr, q[0], '6 release')

    def preinsert_lateral_mm(self, cmd):
        """Plug tip (encoder FK of the approach command + nominal in-hand pose, as eval_chain's servo pivot) across the
        ZED-estimated port axis (mm). The wrist servo aligns the plug with whichever port it sees; a value near the
        port pitch (18.6 mm) means it locked onto a neighbouring port."""
        ep, eR = self.port
        sp, sR = self.scene.chain['right'].fk(torch.as_tensor(cmd, dtype=torch.float32).reshape(1, 6).to(self.scene.device))
        tip = sp - (sR @ (self.R_sr0.T @ self.p_sr0).cpu()[:, None])[..., 0]
        rel = (eR.cpu().transpose(1, 2) @ (tip - ep.cpu())[..., None])[0, :, 0]
        return float(rel[:2].norm()) * 1e3

    # ------------------------------------------------------------------ stage 5: insertion and release
    def stage_insert(self, cmd, q_hold, log):
        from deploy.bimanual.insert import run_insertion
        return run_insertion(self, cmd, q_hold, log)

    def retreat_right(self, port_R):
        """Open the gripper, then back the TCP out along the port axis (port +z) and hold."""
        ic = self.cfg['insert']
        ql = self.q_b[1].clone()
        qr = self.q_b[0].clone()
        # release opening: the sim's right finger limit (24 mm) by default. Real run 2026-11-01 ai4: the jaws opened
        # 0.31 -> 0.38 only and dragged the seated reader during the retreat; insert.release_open_m (e.g. full open)
        # opens wider before backing out.
        self.right_open_override_m = ic.get('release_open_m')
        if self.right_open_override_m is not None:
            self.log_event('release_open', open_m=float(self.right_open_override_m))
        for _ in range(int(round(ic['release_open_s'] / self.dt))):
            self.hstep(qr, ql, '6 release', grip_r=1., grip_l=-1.)
        ch = self.scene.chain['right']
        p0, R0 = ch.fk(qr[None])
        z = port_R[:, :, 2].to(self.dev)
        n = max(1, int(round(ic['retreat_m'] / (ic['retreat_v_m_s'] * self.speed) / self.dt)))
        q = qr[None]
        for k in range(n):
            u = (k + 1) / n
            u = u * u * (3 - 2 * u)
            q, _, _ = ch.ik(p0 + z * ic['retreat_m'] * u, R0, q, iters=10)
            self.hstep(q[0], ql, '6 release', grip_r=1., grip_l=-1.)
        self.right_open_override_m = None

    def home(self):
        """Back to the pose at connect, slow and checked (left first: it is the one near the PC)."""
        q0b = self.q_start + self.off
        self.move({'left': q0b[1]}, label='7 home')
        self.move({'right': q0b[0]}, label='7 home')


def tr_reader(chain):
    """Sim reader pose (x, y, z, quat) for the dry-run logs."""
    return chain.io.ch.pick.reader_pose_q()[0].cpu().numpy()


@contextlib.contextmanager
def _patch(mod, name, value):
    old = getattr(mod, name)
    setattr(mod, name, value)
    try:
        yield
    finally:
        setattr(mod, name, old)
