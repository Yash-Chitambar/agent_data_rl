"""Align stage between the approach and the insertion policy (config "align", docs/real/ALIGN_STAGE.md): bring the plug to the hand-over gap, lined up with the port,
in small verified steps; hand over to K-J only if every gate holds. No sim state on the robot: the only inputs are a measurement source and encoder FK.

One cycle: arm at rest -> measure (several frames) -> gates -> one capped step -> FK check of the step -> at rest -> measure.
FK is used only for small relative moves and for checking the measurement; the absolute error comes from the measurement.

Real arm (results/real_align/tracking.json, collect run 2026-11-02): joint 2 friction band about 1.3 N m at kp 80. A small target step does nothing, then the joint breaks
free and jumps (operator jogs: asked 0.5-1 mm, moved 0, then 4-11 mm). So this stage does not ramp the target and wait: Creep raises the joint's target lead slowly, and cuts it the
moment the joint moves. Joints with low friction (1, 4, 5, 6) keep a plain position target at the goal plus a slow integral (their load).

On any failure: hold, then (on_fail "back_off") straight out along the port axis with closed jaws. A failed stage never hands to RL.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from blind_insertion.envs import friction_comp as FC

AXES = ('across', 'up', 'gap')
DEFAULTS = dict(
    enabled=False, mode='full',
    target_mm=[0.0, 0.0, 6.0], tol_mm=[1.0, 1.0, 1.0], confirm_n=2,
    step_mm=[1.0, 1.0, 1.0], gain=1.0, funnel_lat_mm=2.0, gap_hold_mm=9.0, min_clear_mm=3.0,
    frames=5, min_good=3, frame_spread_mm=1.5, min_conf=0.5, max_sigma_mm=1.5, max_no_measure=5,
    max_cycles=40, max_time_s=90.0, max_lat_mm=20.0, max_advance_mm=16.0,
    drift_mm=1.0, drift_n=2, pred_mm=2.5, pred_n=2, grow_margin_mm=.5, grow_n=4, grow_abs_mm=9.0, grip_change=.02,
    probe_mm=[2.0, 2.0, 2.0], slope_min=.6, slope_max=1.6, slope_early=[.25, 2.0], slope_min_motion_mm=1.5,
    inhand_refresh=True, settle_s=.3, settle_max_s=1.5, still_rad_s=.02, move_tol_mm=.3, step_timeout_s=3.0, stuck_n=3,
    lead_rate_rad_s=[.02, .04, .04, .02, .05, .05], lead_max_rad=[.02, .045, .045, .012, .06, .06], eps_rad=[.001, .001, .001, .002, .003, .003],
    heavy=[False, True, True, False, False, False], ki_per_s=1.0, move_eps_rad=.0005, release=.3,
    handover_lead_rad=[[-.0004, .0004], [-.0051, .0156], [-.011, .0129], [-.0024, .0013], [.0144, .0279], [-.0058, .0019]],
    source=dict(kind='stub'), sim_source=dict(kind='sim_truth'), secondary=None, secondary_sim=None, disagree_mm=[2.0, 2.0, 3.0], disagree_mode='gate',
    on_fail='back_off', back_off_mm=15.0, back_off_v_mm_s=10.0, press_release=True, repress_s=1.0, post_press_tol_factor=1.5,
)


def align_cfg(cfg):
    """Config dict + defaults; None when the stage is off (every older config)."""
    c = cfg.get('align')
    if not c or not c.get('enabled'):
        return None
    d = dict(DEFAULTS)
    d.update({k: v for k, v in c.items() if not k.startswith('_')})
    return d


@dataclass
class PortMeasure:
    """port_vision.PortMeasure's fields: mm, plug tip relative to port 2 (x across, y up, z gap)."""
    ok: bool
    reason: str = ''
    conf: float = 0.
    across_mm: float | None = None
    up_mm: float | None = None
    gap_mm: float | None = None
    sigma_mm: float | None = None
    extra: dict = field(default_factory=dict)

    def vec(self):
        return np.array([self.across_mm, self.up_mm, self.gap_mm], float)


class StubSource:
    """No measurement: the stage stops at its first gate, nothing moves. Default until the port vision lands."""

    def __init__(self, ch, spec):
        self.why = spec.get('why', 'stub source')

    def measure(self, ch, tag):
        return PortMeasure(False, self.why)


class PortVisionSource:
    """deploy/bimanual/port_vision.measure on the newest raw right wrist frame (1280x1024). One shell depth per grasp: taken still at the stop (prepare), held for the stage.
    docs/real/PORT_VISION.md: without it, 3 of 12 detections at gap 0-5 mm were false and the up residual was 1.49 mm. About 82 ms per call."""

    def __init__(self, ch, spec):
        self.kw = {k: v for k, v in spec.items() if k not in ('kind', 'depth_frames')}
        self.n_depth = int(spec.get('depth_frames', 9))
        self.plug_z = None
        try:
            from deploy.bimanual import port_vision
            self.fn, self.depth = port_vision.measure, port_vision.plug_depth
        except ImportError as e:
            self.fn, self.err = None, str(e)

    def raw(self, ch, tag):
        pol, raw, t, cnt = ch.io.fresh_frames(('wrist_right',))['wrist_right']
        if ch.logger is not None:
            ch.logger.frame('wrist_right', pol, tag, t_capture=t, t_use=ch.state['t'], kind='policy', count=cnt, tick=ch.k)
            if raw is not None:
                ch.logger.frame('wrist_right', raw, tag, t_capture=t, t_use=ch.state['t'], kind='raw', bgr=True, count=cnt, tick=ch.k)
        ch.last_frame_tag = f'{tag}@{ch.k}'
        return raw

    def prepare(self, ch, between):
        """Still at the stop: the shell depth of this grasp. No depth: every measure fails, so the stage stops without moving."""
        if self.fn is None:
            return dict(plug_z_mm=None, why=self.err)
        fr = []
        for i in range(self.n_depth):
            f = self.raw(ch, f'align_depth_{i}')
            if f is not None:
                fr.append(f)
            between()
        self.plug_z = self.depth(fr) if len(fr) >= 3 else None
        return dict(plug_z_mm=self.plug_z, frames=len(fr))

    def measure(self, ch, tag):
        if self.fn is None:
            return PortMeasure(False, f'port_vision not available: {self.err}')
        if self.plug_z is None:
            return PortMeasure(False, 'no shell depth for this grasp')
        raw = self.raw(ch, tag)
        if raw is None:
            return PortMeasure(False, 'no raw frame')
        m = self.fn(raw, side='right', plug_z_mm=self.plug_z, **self.kw)
        sig = None if m.sigma_mm is None else float(np.max(m.sigma_mm))
        return PortMeasure(bool(m.ok), str(m.reason), float(m.conf), m.across_mm, m.up_mm, m.gap_mm, sig)


class WristNetSource:
    """The fine wrist net (eval_chain's crop): real slope about 0.34, 0.17, 0.35 per mm (docs/real/SESSION_20261102_duchamp3.md), so the slope gate rejects it until retrained."""

    def __init__(self, ch, spec):
        self.spec = spec
        self.sigma = float(spec.get('sigma_mm', 1.0))
        self.conf = float(spec.get('conf', 1.0))
        name = spec.get('net')
        if name:
            from deploy.bimanual import netcache as NC
            w = NC.wrist(name, ch.ndev)
            self.net, self.scale, self.mode = w[0], w[1].to(ch.ndev), w[2]
        else:
            self.net, self.scale, self.mode = ch.fine[0], ch.fine[1].to(ch.ndev), ch.fine[2]

    def predict(self, fr, ndev):
        import torch.nn.functional as F
        if self.mode == 'crop':
            img = fr[:, 40:40 + 192, 64:64 + 192].contiguous()
        else:
            img = F.interpolate(fr.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
        with torch.no_grad():
            e = self.net(img.to(ndev)).to(ndev) * self.scale
        return e[0, :3].cpu().numpy().astype(float)

    def measure(self, ch, tag):
        fr = ch.render(('wrist_right',), tag=tag)['wrist_cam']
        v = self.predict(fr, ch.ndev)
        return PortMeasure(True, 'net', self.conf, float(v[0]), float(v[1]), float(v[2]), self.sigma)


class SimTruthSource:
    """Dry run only: sim truth with a real-like error model: slope, constant bias, per-frame noise, per-run bias draw, frame drop-outs."""

    def __init__(self, ch, spec):
        if not getattr(ch.io, 'is_sim', False):
            raise RuntimeError('sim_truth source on the real robot')
        self.slope = np.asarray(spec.get('slope', [1., 1., 1.]), float)
        self.bias = np.asarray(spec.get('bias_mm', [0., 0., 0.]), float)
        self.sigma = np.asarray(spec.get('sigma_mm', [.3, .3, .5]), float)
        self.run_bias = np.asarray(spec.get('run_bias_sigma_mm', [0., 0., 0.]), float)
        self.pose_sigma = np.asarray(spec.get('pose_sigma_mm', [0., 0., 0.]), float)       # same at one pose, new at the next (port vision: frame noise < 0.12 mm, residual over poses larger)
        self.pose_err, self.pose_at = np.zeros(3), None
        self.drop = float(spec.get('drop_p', 0.))
        self.conf = float(spec.get('conf', 1.0))
        self.g = np.random.default_rng(int(spec.get('seed', ch.seed or 0)) + 12345)
        self.rb = self.g.normal(0, 1, 3) * self.run_bias

    def measure(self, ch, tag):
        tr = ch.truth()
        if self.g.random() < self.drop:
            return PortMeasure(False, 'sim drop-out')
        p = np.asarray(tr['plug_in_port_mm'], float)
        if self.pose_at is None or np.linalg.norm(p - self.pose_at) > .3:
            self.pose_err, self.pose_at = self.g.normal(0, 1, 3) * self.pose_sigma, p
        v = self.slope * p + self.bias + self.rb + self.pose_err + self.g.normal(0, 1, 3) * self.sigma
        return PortMeasure(True, 'sim', self.conf, float(v[0]), float(v[1]), float(v[2]), float(max(self.sigma.max(), self.pose_sigma.max())), extra=dict(truth=p.tolist()))


SOURCES = dict(stub=StubSource, port_vision=PortVisionSource, wrist_net=WristNetSource, sim_truth=SimTruthSource)


def make_source(ch, spec):
    return SOURCES[spec.get('kind', 'stub')](ch, spec)


class AlignLogic:
    """The decisions, without an arm: feed it measurements and FK displacements; deploy/checks/align_gates.py tests each gate here.

    update(m, m2, tip, moved, grip) -> dict(kind='step' | 'retry' | 'done' | 'fail', delta (3 mm, port axes: +x across, +y up, +z out), reason, ...).
    m, m2: PortMeasure (primary, secondary or None). tip: FK tip against the stage's start, port axes, mm."""

    def __init__(self, c):
        self.c = c
        self.t = np.asarray(c['target_mm'], float)
        self.tol = np.asarray(c['tol_mm'], float)
        self.step_cap = np.asarray(c['step_mm'], float)
        self.cycle = self.no_meas = self.pred_bad = self.grow = self.drift_bad = self.stuck = self.confirm = 0
        self.last_m, self.last_tip, self.last_e = None, None, None
        self.best_e = float('inf')
        self.moved_since = False
        self.pending_probe = []
        for ax, mm in enumerate(c['probe_mm']):
            if mm:
                d = np.zeros(3)
                d[ax] = mm
                self.pending_probe += [('probe', d), ('probe', -d)]
        self.probing = False
        self.sxx, self.sxy = np.zeros(3), np.zeros(3)
        self.grip0 = None

    def slope(self):
        out = []
        for i in range(3):
            out.append(float(self.sxy[i] / self.sxx[i]) if self.sxx[i] > 0 else None)
        return out

    def slope_check(self):
        """(ok, reason): the measurement must follow FK motion, per axis."""
        c = self.c
        need = c['slope_min_motion_mm'] ** 2
        for i, s in enumerate(self.slope()):
            if self.sxx[i] < need:
                return False, f'{AXES[i]}: not enough motion to check the measurement scale ({math.sqrt(self.sxx[i]):.1f} mm)'
            if not (c['slope_min'] <= s <= c['slope_max']):
                return False, f'{AXES[i]}: measurement moves {s:.2f} mm per mm of FK motion (allowed {c["slope_min"]}-{c["slope_max"]})'
        return True, ''

    def fail(self, reason, **kw):
        return dict(kind='fail', reason=reason, delta=np.zeros(3), **kw)

    def update(self, m, m2, tip, grip=None, stuck=False):
        c = self.c
        self.cycle += 1
        tip = np.asarray(tip, float)
        if self.cycle > c['max_cycles']:
            return self.fail('step limit', cycles=self.cycle)
        if np.abs(tip[:2]).max() > c['max_lat_mm'] or -tip[2] > c['max_advance_mm']:
            return self.fail(f'moved too far from the stop ({tip.round(1).tolist()} mm)')
        if grip is not None:
            self.grip0 = grip if self.grip0 is None else self.grip0
            if abs(grip - self.grip0) > c['grip_change']:
                return self.fail(f'gripper reading changed {self.grip0:.3f} -> {grip:.3f}')
        if stuck:
            self.stuck += 1
            if self.stuck >= c['stuck_n']:
                return self.fail('step did not move the hand (contact or friction beyond the lead limit)')
        else:
            self.stuck = 0
        bad = None
        if m is None or not m.ok:
            bad = f'no measurement ({m.reason if m else "none"})'
        elif m.conf < c['min_conf']:
            bad = f'low confidence {m.conf:.2f}'
        elif m.sigma_mm is not None and m.sigma_mm > c['max_sigma_mm']:
            bad = f'sigma {m.sigma_mm:.2f} mm'
        elif getattr(m, 'spread', 0.) > c['frame_spread_mm']:
            bad = f'frames disagree by {m.spread:.2f} mm'
        if bad:
            self.no_meas += 1
            if self.no_meas >= c['max_no_measure']:
                return self.fail(bad + f', {self.no_meas} times')
            return dict(kind='retry', reason=bad, delta=np.zeros(3))
        self.no_meas = 0
        v = m.vec()
        if m2 is not None and m2.ok:
            dis = np.abs(v - m2.vec())
            if (dis > np.asarray(c['disagree_mm'], float)).any() and c['disagree_mode'] == 'gate':
                return self.fail(f'sources disagree {dis.round(2).tolist()} mm')
        if self.last_m is not None:
            dtip = tip - self.last_tip
            dm = v - self.last_m
            self.sxx += dtip ** 2
            self.sxy += dtip * dm
            moved = np.linalg.norm(dtip)
            perr = np.abs(dm - dtip)
            if self.moved_since is False and moved > c['drift_mm']:
                self.drift_bad += 1
                if self.drift_bad >= c['drift_n']:
                    return self.fail(f'hand drifted {moved:.2f} mm with no command')
            else:
                self.drift_bad = 0
            lim = c['pred_mm'] + 2 * float(m.sigma_mm or 0.)
            if (perr > lim).any():
                self.pred_bad += 1
                if self.pred_bad >= c['pred_n']:
                    return self.fail(f'measurement against FK: changed {dm.round(2).tolist()} mm, hand moved {dtip.round(2).tolist()} mm')
            else:
                self.pred_bad = 0
        self.last_m, self.last_tip = v, tip
        err = v - self.t
        e = float(np.linalg.norm(err))
        if v[2] < c['min_clear_mm']:
            return self.fail(f'gap {v[2]:.1f} mm below the {c["min_clear_mm"]} mm limit')
        if self.pending_probe:
            kind, d = self.pending_probe.pop(0)
            self.moved_since = True
            self.probing = True
            return dict(kind='step', delta=d, reason=kind, err=err, e=e, is_probe=True)
        if self.probing:          # the first cycle after the probes starts the divergence count
            self.probing = False
            self.last_e, self.best_e = e, e
            for i, sl in enumerate(self.slope()):      # a camera that does not follow the hand stops here, before any correction (loose: one probe, noisy; the hand-over gate is tighter)
                if self.sxx[i] >= c['slope_min_motion_mm'] ** 2 and not (c['slope_early'][0] <= sl <= c['slope_early'][1]):
                    return self.fail(f'scale gate after the probes: {AXES[i]} measurement moves {sl:.2f} mm per mm of FK motion', err=err)
        if self.last_e is not None:
            if e > self.last_e + c['grow_margin_mm']:
                self.grow += 1
            elif e <= self.last_e:
                self.grow = 0
            if self.grow >= c['grow_n'] or e > self.best_e + c['grow_abs_mm']:
                return self.fail(f'correction grows: {e:.1f} mm, best {self.best_e:.1f} mm, {self.grow} cycles', err=err)
        self.last_e, self.best_e = e, min(self.best_e, e)
        if (np.abs(err) <= self.tol).all():
            self.confirm += 1
            self.moved_since = False
            if self.confirm >= c['confirm_n']:
                ok, why = self.slope_check()
                if not ok:
                    return self.fail('scale gate: ' + why, err=err)
                return dict(kind='done', delta=np.zeros(3), err=err, e=e, reason='within tolerance')
            return dict(kind='retry', reason='confirm', delta=np.zeros(3), err=err, e=e)
        self.confirm = 0
        d = -c['gain'] * err
        lat = float(np.hypot(*err[:2]))
        lat_ok = lat <= c['funnel_lat_mm']
        if not lat_ok and v[2] < c['gap_hold_mm']:       # not lined up and close: back out while correcting
            d[2] = c['gap_hold_mm'] - v[2]
        elif not lat_ok:
            d[2] = min(0., c['gap_hold_mm'] - v[2])
        d = np.clip(d, -self.step_cap, self.step_cap)
        d[2] = max(d[2], c['min_clear_mm'] - v[2])       # never plan below the clearance
        self.moved_since = True
        return dict(kind='step', delta=d, err=err, e=e, reason='correct')


class Creep:
    """Slow joint-level positioning against stiction: target = measured + lead; the lead ramps while a joint is off its goal and is cut as soon as a high-friction joint moves."""

    def __init__(self, ch, q_left, c, lead0):
        self.ch, self.ql, self.c = ch, q_left, c
        self.lead = np.asarray(lead0, float).copy()
        self.lim = np.asarray(c['lead_max_rad'], float)
        self.rate = np.asarray(c['lead_rate_rad_s'], float)
        self.eps = np.asarray(c['eps_rad'], float)
        self.heavy = np.asarray(c['heavy'], bool)
        self.goal = ch.q_b[0].cpu().numpy().astype(float).copy()
        self.prev = self.goal.copy()

    def tick(self):
        ch, c = self.ch, self.c
        q = ch.q_b[0].cpu().numpy().astype(float)
        e = self.goal - q
        moved = (q - self.prev) * np.sign(e)
        tgt = q.copy()
        for j in range(6):
            if self.heavy[j]:      # target = measured + lead: torque ramps until the joint breaks free, then drops
                if abs(e[j]) > self.eps[j]:
                    if moved[j] >= c['move_eps_rad']:
                        self.lead[j] *= c['release']
                    elif self.lead[j] * e[j] < 0:
                        self.lead[j] = 0.
                    else:
                        self.lead[j] += math.copysign(self.rate[j] * ch.dt, e[j])
                else:
                    self.lead[j] *= .9
                self.lead[j] = np.clip(self.lead[j], -self.lim[j], self.lim[j])
                tgt[j] = q[j] + self.lead[j]
            else:                  # low friction: plain position target at the goal + a slow integral (carries the joint's load, e.g. joint 5)
                self.lead[j] = np.clip(self.lead[j] + c['ki_per_s'] * e[j] * ch.dt, -self.lim[j], self.lim[j])
                tgt[j] = self.goal[j] + self.lead[j]
        self.prev = q
        ch.hstep(torch.as_tensor(tgt, dtype=torch.float32, device=ch.dev), self.ql, 'align')


class AlignStage:
    def __init__(self, ch, cmd, q_hold, eR, c):
        self.ch, self.c, self.q_hold = ch, c, q_hold.reshape(6).to(ch.dev).float()
        self.eR = eR.reshape(3, 3).cpu().numpy().astype(float)
        self.chain = ch.scene.chain['right']
        self.s = (ch.R_sr0.T @ ch.p_sr0).cpu().float()
        spec = c['sim_source'] if getattr(ch.io, 'is_sim', False) else c['source']
        self.src = make_source(ch, spec)
        spec2 = c['secondary_sim'] if getattr(ch.io, 'is_sim', False) else c['secondary']
        self.src2 = make_source(ch, spec2) if spec2 else None
        self.logic = AlignLogic(c)
        self.cmd0 = cmd.reshape(6).to(ch.dev).float()
        self.ql = ch.q_b[1].clone()
        self.pressed = ch.pressing

    # ------------------------------------------------------------------ kinematics
    def fk(self, q):
        p, R = self.chain.fk(torch.as_tensor(q, dtype=torch.float32, device=self.ch.scene.device).reshape(1, 6))
        return p[0].cpu(), R[0].cpu()

    def tip_rel(self):
        p, R = self.fk(self.ch.q_b[0])
        tip = (p - R @ self.s).numpy().astype(float)
        return self.eR.T @ (tip - self.tip0) * 1e3

    def goal_of(self, delta_mm):
        """Joint goal: measured tip + delta (port axes), orientation of the stage's start."""
        ch = self.ch
        pm, Rm = self.fk(ch.q_b[0])
        dw = torch.as_tensor(self.eR @ np.asarray(delta_mm, float) * 1e-3, dtype=torch.float32)
        pg = pm + (self.R0 @ self.s - Rm @ self.s) + dw        # tip moves by delta, the hand rotates back to R0 (tip = p - R s)
        dev = ch.scene.device
        q, pe, re = self.chain.ik(pg.to(dev)[None], self.R0.to(dev)[None], ch.q_b[0:1].to(dev).float(), iters=30)
        if float(pe[0]) > 5e-4 or float(re[0]) > 5e-3:
            return None
        return q[0].cpu().numpy().astype(float)

    # ------------------------------------------------------------------ measure
    def still(self):
        """Hold until the arm is at rest (settle_s at least)."""
        c, ch = self.c, self.ch
        n0, nmax = int(round(c['settle_s'] / ch.dt)), int(round(c['settle_max_s'] / ch.dt))
        for k in range(nmax):
            self.creep.tick()
            if k >= n0 and float(np.abs(ch.state['qd'][0]).max()) < c['still_rad_s']:
                return True
        return False

    def one(self, src, tag):
        t0 = time.perf_counter()
        try:
            m = src.measure(self.ch, tag)
        except Exception as e:  # noqa: BLE001  (a vision failure is a failed measurement, not a crash in front of the port)
            m = PortMeasure(False, f'{type(e).__name__}: {e}')
        m.extra['ms'] = (time.perf_counter() - t0) * 1e3
        return m

    def measure(self, tag):
        c, ch = self.c, self.ch
        ms = []
        for i in range(int(c['frames'])):      # one control tick between frames: the arm keeps its 50 Hz commands as far as the camera allows
            ms.append(self.one(self.src, f'{tag}_{i}'))
            self.creep.tick()
        good = [m for m in ms if m.ok and m.gap_mm is not None and m.across_mm is not None and m.up_mm is not None]
        if not good:
            out = ms[-1]
            out.spread = 0.
            out.ok = False
            return out, ms
        V = np.stack([m.vec() for m in good])
        med = np.median(V, 0)
        out = PortMeasure(True, f'{len(good)}/{len(ms)} frames', float(np.median([m.conf for m in good])), float(med[0]), float(med[1]), float(med[2]),
                          None if good[0].sigma_mm is None else float(np.median([m.sigma_mm for m in good])))
        out.spread = float((V.max(0) - V.min(0)).max())
        if len(good) < min(int(c['min_good']), len(ms)):
            out.ok, out.reason = False, f'{len(good)} of {len(ms)} frames measured'
        return out, ms

    # ------------------------------------------------------------------ run
    def log_m(self, m, ms, m2, tip, dec, t):
        rec = dict(cycle=self.logic.cycle, tick=self.ch.k, t_stage=t, ok=m.ok, reason=m.reason, conf=m.conf, across=m.across_mm, up=m.up_mm, gap=m.gap_mm, sigma=m.sigma_mm, spread=getattr(m, 'spread', None),
                   frames=[dict(ok=x.ok, reason=x.reason, conf=x.conf, v=[x.across_mm, x.up_mm, x.gap_mm], ms=x.extra.get('ms'), truth=x.extra.get('truth')) for x in ms],
                   tip_rel=tip.tolist(), q_b=self.ch.q_b[0].tolist(), lead=self.creep.lead.tolist(), grip=float(self.ch.state['grip'][0]), decision=dec['kind'], decision_reason=dec.get('reason'), delta=np.asarray(dec.get('delta', np.zeros(3))).tolist(),
                   slope=self.logic.slope(), second=None if m2 is None else dict(ok=m2.ok, v=[m2.across_mm, m2.up_mm, m2.gap_mm]))
        tr = self.ch.truth()
        if tr is not None:
            rec['truth_mm'] = np.asarray(tr['plug_in_port_mm'], float).tolist()
            rec['truth_rot_deg'] = np.asarray(tr['plug_rot_deg'], float).tolist()
        self.ch.log_event('align_measure', **rec)
        print(f'align {self.logic.cycle:2d}: m {m.across_mm} {m.up_mm} {m.gap_mm} -> {dec["kind"]} {np.round(dec.get("delta", np.zeros(3)), 2).tolist()} {dec.get("reason", "")}', flush=True)

    def step(self, delta):
        """One capped move, verified on FK. Returns dict: moved (3), fraction of the asked motion, timed out."""
        c, ch, cr = self.c, self.ch, self.creep
        start = self.tip_rel()
        goal = self.goal_of(delta)
        if goal is None:
            return dict(moved=np.zeros(3), frac=0., timeout=False, ik=False)
        cr.goal = goal
        want = start + np.asarray(delta, float)
        n = int(round(c['step_timeout_s'] / ch.dt))
        reached = False
        k = 0
        for k in range(n):
            cr.tick()
            if np.linalg.norm(self.tip_rel() - want) <= c['move_tol_mm']:
                reached = True
                break
        moved = self.tip_rel() - start
        asked = float(np.linalg.norm(delta))
        frac = float(moved @ np.asarray(delta, float) / max(asked ** 2, 1e-9))
        rec = dict(tick=ch.k, delta=np.asarray(delta, float).tolist(), moved=moved.tolist(), frac=frac, reached=reached, ticks=k + 1, lead=cr.lead.tolist())
        ch.log_event('align_step', **rec)
        return dict(moved=moved, frac=frac, timeout=not reached, ik=True)

    def run(self, log):
        c, ch = self.c, self.ch
        ch.log_event('align_start', cfg={k: v for k, v in c.items() if not k.startswith('_')}, source=c['source'] if not getattr(ch.io, 'is_sim', False) else c['sim_source'])
        t_start = ch.k * ch.dt
        # press off: no force and no hold-time count during the slow steps (as the collect stage); the hand stays on the PC
        ql = ch.q_b[1].clone() if c['press_release'] else self.q_hold
        qr0 = self.cmd0
        ff_on = ch.rt.comp['right'] is not None
        lead0 = (qr0 - ch.q_b[0]).cpu().numpy().astype(float)
        if ff_on and ch.fric is not None:         # the feed-forward torque goes into the target, then it is off
            tau_ff = FC.feedforward(qr0 - ch.q_b[0], *ch.fric)
            lead0 = lead0 + (tau_ff / ch.scene.kp[0].to(ch.dev)).cpu().numpy().astype(float)
        ch.rt.comp['right'] = None
        lead0 = np.clip(lead0, -np.asarray(c['lead_max_rad'], float), np.asarray(c['lead_max_rad'], float))
        self.ql = ql
        if c['press_release']:
            ch.pressing = False
            ch.log_event('align_press_off', q_left=ql.tolist(), tick=ch.k)
        self.creep = Creep(ch, ql, c, lead0)
        ch.log_event('align_windup_at_entry', lead=lead0.tolist(), cmd=qr0.tolist(), q_b=ch.q_b[0].tolist())
        p0, R0 = self.fk(ch.q_b[0])
        self.R0, self.tip0 = R0, (p0 - R0 @ self.s).numpy().astype(float)
        self.creep.goal = ch.q_b[0].cpu().numpy().astype(float).copy()
        out = dict(ok=False, reason='', cycles=0, gap_mm=None, lat_mm=None)
        last = None
        self.still()
        if hasattr(self.src, 'prepare'):
            pr = self.src.prepare(ch, self.creep.tick)
            ch.log_event('align_prepare', **pr)
            print(f'align: shell depth for this grasp {pr}', flush=True)
        off = c.get('sim_entry_offset_mm')
        if off and getattr(ch.io, 'is_sim', False):       # dry run only: start from a real-like stop error (the sim's approach ends much closer than the real one)
            g = np.random.default_rng(int(ch.seed or 0) + 777)
            off = np.asarray(off, float)
            off = off[0] + (off[1] - off[0]) * g.random(3)
            for _ in range(30):
                rem = off - self.tip_rel()
                if np.linalg.norm(rem) < .5:
                    break
                self.step(np.clip(rem, -2., 2.))
            ch.log_event('align_sim_entry_offset', asked=off.tolist(), got=self.tip_rel().tolist())
            p0, R0 = self.fk(ch.q_b[0])
            self.R0, self.tip0 = R0, (p0 - R0 @ self.s).numpy().astype(float)
            self.creep.goal = ch.q_b[0].cpu().numpy().astype(float).copy()
        while True:
            self.still()
            m, ms = self.measure(f'align_{self.logic.cycle:03d}')
            m2 = self.one(self.src2, f'align2_{self.logic.cycle:03d}') if self.src2 is not None else None
            tip = self.tip_rel()
            stuck = bool(last and last.get('timeout') and last.get('frac', 1.) < .25)
            if ch.k * ch.dt - t_start > c['max_time_s']:
                dec = self.logic.fail('time limit')
            elif c['mode'] == 'measure':
                dec = self.logic.update(m, m2, tip, grip=float(ch.state['grip'][0]))
                self.logic.pending_probe = []
                if dec['kind'] == 'step' or dec['kind'] == 'done' or self.logic.cycle >= int(c.get('measure_n', 10)):
                    dec = dict(kind='retry', reason='measure mode: no motion', delta=np.zeros(3), planned=np.asarray(dec.get('delta', np.zeros(3))).tolist())
                    self.logic.moved_since = False
                    if self.logic.cycle >= int(c.get('measure_n', 10)):
                        dec = dict(kind='fail', reason='measure mode finished', delta=np.zeros(3))
            else:
                dec = self.logic.update(m, m2, tip, grip=float(ch.state['grip'][0]), stuck=stuck)
            self.log_m(m, ms, m2, tip, dec, ch.k * ch.dt - t_start)
            if dec['kind'] == 'fail':
                out['reason'] = dec['reason']
                break
            if dec['kind'] == 'done':
                out.update(ok=True, reason=dec['reason'])
                break
            if dec['kind'] == 'step':
                last = self.step(dec['delta'])
            else:
                last = None
        out.update(cycles=self.logic.cycle, slope=self.logic.slope(), seconds=ch.k * ch.dt - t_start)
        out['last'] = None if self.logic.last_m is None else self.logic.last_m.tolist()
        return out

    def handover(self, log):
        """Stage succeeded: target lead into the range K-J saw, press back on, last measurement. Returns (cmd for run_insertion or None, dict)."""
        c, ch = self.c, self.ch
        lo = np.asarray(c['handover_lead_rad'], float)
        self.creep.lead = np.clip(self.creep.lead, lo[:, 0], lo[:, 1])
        n = int(round(c['repress_s'] / ch.dt))
        if c['press_release']:
            ql0, ql1 = self.ql.clone(), self.q_hold
            ch.pressing = True
            for k in range(n):
                u = (k + 1) / n
                u = u * u * (3 - 2 * u)
                self.creep.ql = ql0 + (ql1 - ql0) * u
                self.creep.tick()
            ch.log_event('align_press_on', tick=ch.k)
        self.creep.ql = self.q_hold
        self.still()
        m, ms = self.measure('align_final')
        tip = self.tip_rel()
        tol = np.asarray(c['tol_mm'], float) * c['post_press_tol_factor']
        err = (m.vec() - np.asarray(c['target_mm'], float)) if m.ok else None
        ok = bool(m.ok and (np.abs(err) <= tol).all())
        self.log_m(m, ms, None, tip, dict(kind='done' if ok else 'fail', reason='final check after the press' if ok else 'moved out of tolerance after the press'), ch.k * ch.dt)
        if c.get('inhand_refresh', True) and ok:        # the in-hand net trained at the 6 mm stop: read it again at the hand-over
            import torch.nn.functional as F
            net, scale, _ = ch.inhand_net
            fr = ch.render(('wrist_right',), tag='align_inhand')['wrist_cam']
            img = F.interpolate(fr.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
            with torch.no_grad():
                new = net(img) * scale
            ch.log_event('align_inhand', before=ch.inhand_est6.reshape(-1).tolist(), after=new.reshape(-1).tolist())
            ch.inhand_est6 = new
        q = ch.q_b[0].cpu().numpy().astype(float)
        cmd = torch.as_tensor(q + self.creep.lead, dtype=torch.float32, device=ch.dev)
        res = dict(ok=ok, gap_mm=float(m.gap_mm) if m.ok else None, lat_mm=[float(m.across_mm), float(m.up_mm)] if m.ok else None, sigma_mm=m.sigma_mm, lead=self.creep.lead.tolist())
        ch.log_event('align_handover', **res)
        return (cmd[None] if ok else None), res

    def back_off(self):
        return back_off_closed(self.ch, self.eR, self.c['back_off_mm'], self.c['back_off_v_mm_s'], self.ql if not self.ch.pressing else self.q_hold)


def back_off_closed(ch, eR, mm, v_mm_s, q_left):
    """Hold the measured pose (no push), then straight out along the port axis, jaws closed. The reader stays in the hand."""
    eR = np.asarray(eR, float).reshape(3, 3)
    for _ in range(int(round(.2 / ch.dt))):
        ch.hstep(ch.q_b[0].clone(), q_left, ch.label)
    chain = ch.scene.chain['right']
    dev = ch.scene.device
    q0 = ch.q_b[0:1].to(dev).float()
    p0, R0 = chain.fk(q0)
    z = torch.as_tensor(eR[:, 2], dtype=torch.float32, device=p0.device)
    n = max(1, int(round(mm / (v_mm_s * max(ch.speed, 1e-3)) / ch.dt)))
    q = q0
    for k in range(n):
        u = (k + 1) / n
        u = u * u * (3 - 2 * u)
        q, _, _ = chain.ik(p0 + z[None] * mm * 1e-3 * u, R0, q, iters=10)
        ch.hstep(q[0].to(ch.dev), q_left, ch.label)
    ch.log_event('back_off', mm=mm, tick=ch.k, tip_q=ch.q_b[0].tolist())


def run_align(ch, cmd, q_hold, ep, eR, log, c):
    """Stage entry. Returns dict(ok, cmd (1, 6) or None, ...); on failure the plug is backed off with closed jaws and q_left in res['q_left']."""
    st = AlignStage(ch, cmd, q_hold, eR, c)
    out = st.run(log)
    out['q_left'] = st.ql
    out['cmd'] = None
    if out['ok'] and c['mode'] == 'full':
        out['cmd'], h = st.handover(log)
        out['handover'] = h
        if out['cmd'] is None:
            out.update(ok=False, reason='final check after the press failed')
        else:
            ch.align_result = h
    if out['ok'] and c['mode'] != 'full':
        out['reason'] = 'aligned; mode ' + c['mode'] + ': no insertion'
    if not out['ok'] or c['mode'] != 'full':
        if c['on_fail'] == 'back_off' or c['mode'] != 'full':
            st.back_off()
    ch.log_event('align_result', **{k: v for k, v in out.items() if k not in ('cmd', 'q_left')}, tick=ch.k)
    print(f'align: {"OK" if out["ok"] else "STOP"} {out["reason"]} ({out["cycles"]} cycles, {out.get("seconds", 0):.1f} s, slope {out.get("slope")})', flush=True)
    return out
