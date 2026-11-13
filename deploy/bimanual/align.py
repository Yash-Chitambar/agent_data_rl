"""Align stage between the approach and the insertion policy (config "align", docs/real/ALIGN_STAGE.md): bring the plug to the hand-over gap, lined up with the port,
in small verified steps; hand over to K-J only if every gate holds. No sim state on the robot: the only inputs are a measurement source and encoder FK.

One cycle: arm at rest -> measure (several frames) -> gates -> one capped step -> FK check of the step -> at rest -> measure.
FK is used only for small relative moves and for checking the measurement; the absolute error comes from the measurement.

Real arm (results/real_align/tracking.json, collect run 2026-11-02): joint 2 friction band about 1.3 N m at kp 80. A small target step does nothing, then the joint breaks
free and jumps (operator jogs: asked 0.5-1 mm, moved 0, then 4-11 mm). So this stage does not ramp the target and wait: Creep raises a joint's target lead slowly, and cuts it the
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
    step_mm=[1.0, 1.0, 1.0], angle=None, gain=1.0, min_clear_fail_mm=None, fuse_n=6, funnel_lat_mm=2.0, gap_hold_mm=9.0, min_clear_mm=3.0,
    frames=5, min_good=3, frame_spread_mm=1.5, min_conf=0.5, max_sigma_mm=1.5, max_no_measure=5,
    max_cycles=60, max_time_s=120.0, post_press_cycles=6, max_lat_mm=20.0, max_advance_mm=16.0,
    drift_mm=1.0, drift_n=2, pred_mm=2.5, pred_n=2, grow_margin_mm=.5, grow_n=4, grow_abs_mm=9.0, grip_change=.02,
    probe_mm=[2.0, 3.0, 2.0], extra_probe_mm=[2.0, 3.0, 2.0], extra_probes=4, slope_min=.6, slope_max=1.6, slope_se_max=.2, slope_noise_floor_mm=[.1, .1, .1],
    inhand_refresh=True, settle_s=.3, settle_max_s=1.5, still_rad_s=.02, move_tol_mm=.3, step_timeout_s=3.0, stuck_n=3,
    lead_rate_rad_s=[.02, .04, .04, .02, .05, .05], lead_max_rad=[.02, .045, .045, .012, .06, .06], eps_rad=[.001, .001, .001, .002, .003, .003],
    heavy=[False, True, True, False, False, False], ki_per_s=1.0, relax=.1, move_eps_rad=.0005, release=.3,
    handover_lead_rad=[[-.0004, .0004], [-.0051, .0156], [-.011, .0129], [-.0024, .0013], [.0144, .0279], [-.0058, .0019]],
    source=dict(kind='stub'), sim_source=dict(kind='sim_truth'), secondary=None, secondary_sim=None, disagree_mm=[2.0, 2.0, 3.0], disagree_mode='gate',
    on_fail='back_off', back_off_mm=15.0, back_off_v_mm_s=10.0, press_release=True, repress_s=1.0, post_press_tol_factor=1.5,
    servo=None,       # dict: continuous vision servo instead of the steps (ServoStage; docs/real/K2_PROGRESS.md "Servo deploy"); None = the step stage
    # fast step stage (docs/real/ALIGN_FAST_20261113.md); every default keeps the old behaviour
    reach_mode='strict',      # 'strict': step done when the FK tip is within move_tol_mm of the asked motion; 'loose': reach_ok (below)
    reach_rel=.5, reach_side_mm=.6, reach_hold_ticks=3, reach_frac_min=.6, reach_stall_s=.5, reach_stall_mm=.15, reach_freeze=False,
    slope_prior=None,         # dict(known_axes=[2]): no probe motion; the scale gate fails only on clear evidence (2 standard errors out of range); axes in known_axes must be known
    gap_step_ratio=None, gap_step_max_mm=3.,   # gap step cap = clip(ratio x (gap - target gap), step_mm[2], gap_step_max_mm) while lined up; None: step_mm[2]
    fresh_frames='all',       # 'first': only the first frame of a measurement waits for two new camera frames; the others wait for one new frame
    secondary_near_mm=None,   # second source only when the primary is within tol + this (it is used only at the hand-over); None: every cycle
)


def reach_ok(c, hist, delta, dt=.02):
    """Step completion test. hist: FK tip motion since the step start (port axes, mm), one per control tick, newest last; delta: asked motion (mm).
    strict: |moved - delta| <= move_tol_mm (real 2026-11-10/11: 123 of 229 steps ran to the 3 s timeout, median moved / asked 1.09 there).
    loose, per axis: every asked axis (|delta_i| > move_tol_mm) within move_tol_mm + reach_rel x |delta_i|, every other axis within reach_side_mm, for reach_hold_ticks ticks in a row; or every axis asked more than
    move_tol_mm has done at least reach_frac_min of its part and the tip moved under reach_stall_mm in the last reach_stall_s (stick-slip: the
    joints sit in their friction band). Per axis, so a step that closes the gap but leaves "up" behind does not count (dry run seed 7041 with a
    norm test: up moved 0.08 of 1 mm, the step counted). The next measurement corrects the rest."""
    d = np.asarray(delta, float)
    h = np.asarray(hist[-1], float)
    if c.get('reach_mode', 'strict') != 'loose':
        return bool(np.linalg.norm(h - d) <= c['move_tol_mm'])
    tol = np.where(np.abs(d) > c['move_tol_mm'], c['move_tol_mm'] + c['reach_rel'] * np.abs(d), c['reach_side_mm'])     # asked axes: relative; the others: reach_side_mm
    n = int(c['reach_hold_ticks'])
    if len(hist) >= n and all((np.abs(np.asarray(x, float) - d) <= tol).all() for x in hist[-n:]):
        return True
    w = max(1, int(round(c['reach_stall_s'] / dt)))
    big = np.abs(d) > c['move_tol_mm']
    if big.any() and len(hist) > w:
        done = (h * np.sign(d) >= c['reach_frac_min'] * np.abs(d)) | ~big
        if done.all() and np.linalg.norm(h - np.asarray(hist[-1 - w], float)) <= c['reach_stall_mm']:
            return True
    return False


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
    rv_deg: object = None         # plug against the port (rx pitch about across, ry yaw about up, rz roll), or None

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
        self.kw = {k: v for k, v in spec.items() if k not in ('kind', 'depth_frames', 'plug_z_range_mm')}
        self.n_depth = int(spec.get('depth_frames', 9))
        self.z_range = spec.get('plug_z_range_mm', [113.5, 123.5])
        self.R_cs_T_R_rs = None
        self.plug_z = None
        try:
            from deploy.bimanual import port_vision
            self.fn, self.depth = port_vision.measure, port_vision.plug_depth
        except ImportError as e:
            self.fn, self.err = None, str(e)

    def raw(self, ch, tag, fresh=True):
        """fresh: wait for two new camera frames (a frame captured after this call). Else wait for one frame newer than the last one used (the arm is
        already at rest: the later frames of one measurement; real 2026-11-10/11: the two-frame wait was about 60 of the 87 ms per frame)."""
        last = getattr(self, 'last_cnt', None)
        if fresh or last is None or not hasattr(ch.io, 'frames'):
            pol, raw, t, cnt = ch.io.fresh_frames(('wrist_right',))['wrist_right']
        else:
            t0 = time.monotonic()
            while True:
                pol, raw, t, cnt = ch.io.frames(('wrist_right',))['wrist_right']
                if cnt > last or time.monotonic() - t0 > .5:
                    break
                time.sleep(.002)
        self.last_cnt = cnt
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
        z = self.depth(fr) if len(fr) >= 3 else None
        lo, hi = self.z_range           # hand-placed grasps of 2026-11-01 read 133 and 142 mm, wrong for this model; picked grasps 117.7-121.5
        self.plug_z = z if z is not None and lo <= z <= hi else None
        return dict(plug_z_mm=z, accepted=self.plug_z is not None, range=[lo, hi], frames=len(fr))

    def measure(self, ch, tag):
        if self.fn is None:
            return PortMeasure(False, f'port_vision not available: {self.err}')
        if self.plug_z is None:
            return PortMeasure(False, 'no shell depth for this grasp')
        raw = self.raw(ch, tag, fresh=getattr(self, 'fresh_next', True))
        if raw is None:
            return PortMeasure(False, 'no raw frame')
        return self.measure_raw(raw)

    def measure_raw(self, raw):
        """The measurement of one raw frame (no camera, no logger: the servo's vision thread calls this)."""
        if self.fn is None:
            return PortMeasure(False, f'port_vision not available: {self.err}')
        if self.plug_z is None:
            return PortMeasure(False, 'no shell depth for this grasp')
        return self.convert(self.fn(raw, side='right', plug_z_mm=self.plug_z, **self.kw))

    def convert(self, m):
        """port_vision.PortMeasure -> this stage's PortMeasure (panel and plug angles)."""
        sig = None if m.sigma_mm is None else float(np.max(m.sigma_mm))
        out = PortMeasure(bool(m.ok), str(m.reason), float(m.conf), m.across_mm, m.up_mm, m.gap_mm, sig)
        R = (m.debug or {}).get('R')          # panel in the camera
        if R is not None:
            from scipy.spatial.transform import Rotation
            out.extra['panel_rv_deg'] = np.degrees(Rotation.from_matrix(np.asarray(R, float)).as_rotvec()).round(3).tolist()
            out.rv_deg = Rotation.from_matrix(np.asarray(R, float).T @ self.plug_in_camera()).as_rotvec(degrees=True)
        return out

    def plug_in_camera(self):
        """Reader in the camera: calibrated mount, mean of the measured real grasps (preset v2). Per-grasp pitch spread SD 0.6 deg stays as an error."""
        if self.R_cs_T_R_rs is None:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parents[2]
            sys.path.append(str(root / 'scripts/real2sim'))
            import inhand_measure as IM
            from blind_insertion.envs import inhand_real as IR
            hand = IM.Hand()
            d = np.mean(np.asarray(IR.load('v2')['inhand']['rows'], float), 0)
            Rrs = IR.reader_in_site(torch.tensor(d[None], dtype=torch.float32), 'cpu')[0][0].numpy().astype(float)
            self.R_cs_T_R_rs = hand.R_cs.T @ Rrs
        return self.R_cs_T_R_rs


class WristNetSource:
    """The fine wrist net (eval_chain's crop): real slope about 0.34, 0.17, 0.35 per mm (docs/real/SESSION_20261102_duchamp3.md), so the slope gate rejects it until retrained."""

    def __init__(self, ch, spec):
        self.spec = spec          # secondary source of bimanual_align1: the new nets are fast and cannot refuse a frame, and across sits 1.4-1.8 mm from vision (docs/real/WRIST_NETS_REAL.md)
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
        self.rot_sigma = np.asarray(spec.get('rot_sigma_deg', [.8, .3, .6]), float)    # pose residual of the panel rotation, real sweep near the port
        self.rot_run = self.g_rot_run = np.asarray(spec.get('rot_run_sigma_deg', [.6, .4, .6]), float)   # per-grasp pitch spread and the mount
        self.drop = float(spec.get('drop_p', 0.))
        self.conf = float(spec.get('conf', 1.0))
        self.g = np.random.default_rng(int(spec.get('seed', ch.seed or 0)) + 12345)
        self.rb = self.g.normal(0, 1, 3) * self.run_bias
        self.rrb = self.g.normal(0, 1, 3) * self.rot_run
        self.rot_err = np.zeros(3)

    def measure(self, ch, tag):
        tr = ch.truth()
        if self.g.random() < self.drop:
            return PortMeasure(False, 'sim drop-out')
        p = np.asarray(tr['plug_in_port_mm'], float)
        if self.pose_at is None or np.linalg.norm(p - self.pose_at) > .3:
            self.pose_err, self.pose_at = self.g.normal(0, 1, 3) * self.pose_sigma, p
            self.rot_err = self.g.normal(0, 1, 3) * self.rot_sigma
        v = self.slope * p + self.bias + self.rb + self.pose_err + self.g.normal(0, 1, 3) * self.sigma
        return PortMeasure(True, 'sim', self.conf, float(v[0]), float(v[1]), float(v[2]), float(max(self.sigma.max(), self.pose_sigma.max())), extra=dict(truth=p.tolist()),
                           rv_deg=np.asarray(tr['plug_rot_deg'], float) + self.rrb + self.rot_err)


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
        self.levels = []
        self.extra_probes = int(c['extra_probes'])
        self.scale_ok = False
        self.grip0 = None
        self.fused = []
        self.last_fused = None
        self.a = c.get('angle') or None      # {axes: [0, 1], tol_deg, step_deg, max_deg, required}
        self.fused_rv = []
        self.rot_applied = np.zeros(3)
        self.last_rv = None

    def slope(self):
        """Measurement against FK motion, per axis: regression on the levels of all cycles at their poses (differences would double the pose noise)."""
        return self.slope_se()[0]

    def slope_se(self):
        if len(self.levels) < 3:
            return [None] * 3, [float('inf')] * 3
        A = np.asarray(self.levels, float)      # (n, 6): tip, measurement
        sl, se = [], []
        for i in range(3):
            x, y = A[:, i] - A[:, i].mean(), A[:, i + 3] - A[:, i + 3].mean()
            sxx = float(x @ x)
            if sxx < 1e-6:
                sl.append(None)
                se.append(float('inf'))
                continue
            b = float(x @ y) / sxx
            res = y - b * x
            sd = max(float(np.sqrt(res @ res / max(len(x) - 2, 1))), self.c['slope_noise_floor_mm'][i])
            sl.append(b)
            se.append(sd / np.sqrt(sxx))
        return sl, se

    def slope_check(self):
        """(ok, axis needing more motion or None, reason): the measurement must follow FK motion, per axis, known within slope_se_max."""
        c = self.c
        sl, se = self.slope_se()
        need = None
        for i in range(3):
            if sl[i] is None or se[i] > c['slope_se_max']:
                need = i if need is None else need
                continue
            if not (c['slope_min'] <= sl[i] <= c['slope_max']):
                return False, None, f'{AXES[i]}: measurement moves {sl[i]:.2f} mm per mm of FK motion (allowed {c["slope_min"]}-{c["slope_max"]})'
        if need is not None:
            return False, need, f'{AXES[need]}: motion too small to check the measurement scale (se {se[need]:.2f})'
        return True, None, ''

    def slope_check_prior(self):
        """Fixed slope prior (config slope_prior; real 2026-11-10/11, n = 14 runs: across 0.94-1.10, up 0.76-0.98, gap 1.02-1.12): an axis that moved enough
        must lie in [slope_min, slope_max]; an axis that moved little fails only when it is out of the range by more than 2 standard errors. The axes in
        known_axes (default gap, which moves about 10 mm on the way in) must be known within slope_se_max. Returns (ok, axis needing motion or None, reason)."""
        c = self.c
        known = list((c.get('slope_prior') or {}).get('known_axes', [2]))
        sl, se = self.slope_se()
        for i in range(3):
            if sl[i] is None:
                if i in known:
                    return False, i, f'{AXES[i]}: no motion to check the measurement scale'
                continue
            if se[i] <= c['slope_se_max']:
                if not (c['slope_min'] <= sl[i] <= c['slope_max']):
                    return False, None, f'{AXES[i]}: measurement moves {sl[i]:.2f} mm per mm of FK motion (allowed {c["slope_min"]}-{c["slope_max"]})'
            elif i in known:
                return False, i, f'{AXES[i]}: motion too small to check the measurement scale (se {se[i]:.2f})'
            elif sl[i] + 2 * se[i] < c['slope_min'] or sl[i] - 2 * se[i] > c['slope_max']:
                return False, None, f'{AXES[i]}: measurement moves {sl[i]:.2f} mm per mm of FK motion (se {se[i]:.2f}, prior range {c["slope_min"]}-{c["slope_max"]})'
        return True, None, ''

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
        dis = np.abs(v - m2.vec()) if m2 is not None and m2.ok else None
        if self.last_m is not None:
            dtip = tip - self.last_tip
            dm = v - self.last_m
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
        if not self.scale_ok:
            self.levels.append(np.r_[tip, v])
        # port in the FK frame: measurement - FK tip, median of the last cycles. FK is good for small relative moves, the vision residual changes from pose to pose (up 0.74 mm).
        self.fused.append(v - tip)
        vf = np.median(self.fused[-int(c['fuse_n']):], 0) + tip
        self.last_fused = vf
        rot_err = None
        if self.a:
            if getattr(m, 'rv_deg', None) is not None:
                self.fused_rv.append(np.asarray(m.rv_deg, float) - self.rot_applied)
            if self.fused_rv:
                self.last_rv = np.median(self.fused_rv[-int(c['fuse_n']):], 0) + self.rot_applied
                rot_err = self.last_rv[self.a['axes']]
        err = vf - self.t
        e = float(np.linalg.norm(err))
        g_ = min(v[2], vf[2])
        floor_ = c['min_clear_fail_mm'] if c.get('min_clear_fail_mm') is not None else c['min_clear_mm']
        if g_ < floor_:
            return self.fail(f'gap {g_:.1f} mm below the {floor_} mm limit')
        if g_ < c['min_clear_mm']:         # too close but free: out first
            self.moved_since = True
            return dict(kind='step', delta=np.array([0., 0., float(self.step_cap[2])]), reason='too close: back out', err=err, e=e)
        if self.pending_probe:
            kind, d = self.pending_probe.pop(0)
            self.moved_since = True
            self.probing = True
            return dict(kind='step', delta=d, reason=kind, err=err, e=e, is_probe=True)
        if self.probing:          # the first cycle after the probes starts the divergence count
            self.probing = False
            self.last_e, self.best_e = e, e
            sl_, se_ = self.slope_se()
            for i in range(3):      # a camera that does not follow the hand stops here, before any correction: clearly out, by more than 2 standard errors
                if sl_[i] is not None and (sl_[i] + 2 * se_[i] < c['slope_min'] or sl_[i] - 2 * se_[i] > c['slope_max']):
                    return self.fail(f'scale gate after the probes: {AXES[i]} measurement moves {sl_[i]:.2f} mm per mm of FK motion (se {se_[i]:.2f})', err=err)
        if self.last_e is not None:
            if e > self.last_e + c['grow_margin_mm']:
                self.grow += 1
            elif e <= self.last_e:
                self.grow = 0
            if self.grow >= c['grow_n'] or e > self.best_e + c['grow_abs_mm']:
                return self.fail(f'correction grows: {e:.1f} mm, best {self.best_e:.1f} mm, {self.grow} cycles', err=err)
        self.last_e, self.best_e = e, min(self.best_e, e)
        rot_off = None
        if self.a:
            if rot_err is None:
                if self.a.get('required', True):
                    return self.fail('no angle measurement')
            elif np.abs(self.rot_applied).max() > self.a['max_deg']:
                return self.fail(f'angle correction above {self.a["max_deg"]} deg: {self.rot_applied.round(1).tolist()}')
            elif (np.abs(rot_err) > self.a['tol_deg']).any():
                rot_off = rot_err
        if (np.abs(err) <= self.tol).all() and rot_off is None:
            self.confirm += 1
            self.moved_since = False
            if self.confirm >= c['confirm_n']:
                if c.get('secondary_required') and dis is None:
                    return self.fail('second source has no reading at the hand-over', err=err)
                if dis is not None and c['disagree_mode'] == 'gate' and (dis > np.asarray(c['disagree_mm'], float)).any():
                    return self.fail(f'sources disagree at the hand-over {dis.round(2).tolist()} mm', err=err)
                if not self.scale_ok:
                    ok, ax, why = self.slope_check_prior() if c.get('slope_prior') else self.slope_check()
                    if not ok and ax is not None and self.extra_probes > 0:      # not known well enough yet: another probe on that axis, and back
                        self.extra_probes -= 1
                        d = np.zeros(3)
                        d[ax] = self.c['extra_probe_mm'][ax]
                        self.pending_probe += [('probe', -d)]
                        self.confirm = 0
                        self.probing = False
                        self.moved_since = True
                        return dict(kind='step', delta=d, reason='extra probe', err=err, e=e, is_probe=True)
                    if not ok:
                        return self.fail('scale gate: ' + why, err=err)
                    self.scale_ok = True
                self.final = vf.copy()
                return dict(kind='done', delta=np.zeros(3), err=err, e=e, reason='within tolerance')
            return dict(kind='retry', reason='confirm', delta=np.zeros(3), err=err, e=e)
        self.confirm = 0
        d = -c['gain'] * err
        lat = float(np.hypot(*err[:2]))
        lat_ok = lat <= c['funnel_lat_mm']
        if not lat_ok and vf[2] < c['gap_hold_mm']:       # not lined up and close: back out while correcting
            d[2] = c['gap_hold_mm'] - vf[2]
        elif not lat_ok:
            d[2] = min(0., c['gap_hold_mm'] - vf[2])
        cap = self.step_cap.copy()
        if c.get('gap_step_ratio') and lat_ok:            # lined up and far: larger gap steps (the last millimetres keep step_mm)
            cap[2] = float(np.clip(c['gap_step_ratio'] * (vf[2] - self.t[2]), self.step_cap[2], max(c['gap_step_max_mm'], self.step_cap[2])))
        d = np.clip(d, -cap, cap)
        d[2] = max(d[2], c['min_clear_mm'] - vf[2])       # never plan below the clearance
        self.moved_since = True
        dec = dict(kind='step', delta=d, err=err, e=e, reason='correct')
        if rot_off is not None:
            r = np.zeros(3)
            r[self.a['axes']] = np.clip(-rot_off, -self.a['step_deg'], self.a['step_deg'])
            self.rot_applied = self.rot_applied + r
            dec['rot_deg'] = r
        return dec


def creep_state(lead):
    lead = np.asarray(lead, float)
    return dict(lead=lead.copy(), brk_pos=np.full(lead.shape, np.nan), brk_neg=np.full(lead.shape, np.nan), rest=np.zeros(lead.shape, bool))


def creep_law(q, prev, goal, st, c, dt):
    """Joint targets of one control tick; arrays (..., 6), so the batched sim eval runs the same law. st: creep_state, updated in place. Returns the target.
    Target = goal + lead on every joint: the position error always pulls toward the goal, so the hand cannot run away (a target that follows the measured joint let it climb 20 mm in the batched eval).
    High-friction joints: the lead ramps while the joint is off its goal and is cut as soon as the joint moves, so that little stored torque is left when it breaks free. Breaks from rest in both directions give the
    middle of the friction band (the constant load: real joint 2 about -0.87 N m); that is the holding lead. At the goal the lead relaxes to it: a lead kept at the edge of the band slips later (dry run: 2.5 mm per cycle), a lead of zero
    springs back against the load (dry run: 0.7 mm per 1 mm step).
    Low friction: a slow integral (joint 5 carries about 0.2 N m)."""
    heavy = np.asarray(c['heavy'], bool)
    eps, rate, lim = np.asarray(c['eps_rad'], float), np.asarray(c['lead_rate_rad_s'], float), np.asarray(c['lead_max_rad'], float)
    lead = st['lead']
    e = goal - q
    moved = (q - prev) * np.sign(e)
    off = np.abs(e) > eps
    broke = off & (moved >= c['move_eps_rad'])
    fresh = broke & st['rest']
    st['brk_pos'] = np.where(fresh & (e > 0), lead + e, st['brk_pos'])
    st['brk_neg'] = np.where(fresh & (e < 0), lead + e, st['brk_neg'])
    st['rest'] = np.abs(q - prev) < c['move_eps_rad'] * .5
    hold = .5 * (st['brk_pos'] + st['brk_neg'])
    known = np.isfinite(hold) & (st['brk_pos'] > st['brk_neg'])
    base = np.where(known, hold, 0.)
    cut = np.where(known, hold, lead * c['release'])
    ramp = np.where((lead - base) * e < 0, base, lead + np.sign(e) * rate * dt)      # goal now on the other side: start again from the holding lead
    h = np.where(broke, cut, ramp)
    h = np.where(off, h, lead + (base - lead) * c['relax'])
    lo = lead + c['ki_per_s'] * e * dt
    st['lead'] = np.clip(np.where(heavy, h, lo), -lim, lim)
    return goal + st['lead']


class Creep:
    """Slow joint-level positioning against stiction: target = measured + lead; the lead ramps while a joint is off its goal and is cut as soon as a high-friction joint moves."""

    def __init__(self, ch, q_left, c, lead0):
        self.ch, self.ql, self.c = ch, q_left, c
        self.st = creep_state(lead0)
        self.goal = ch.q_b[0].cpu().numpy().astype(float).copy()
        self.prev = self.goal.copy()

    @property
    def lead(self):
        return self.st['lead']

    @lead.setter
    def lead(self, v):
        self.st['lead'] = np.asarray(v, float)

    def tick(self):
        ch = self.ch
        q = ch.q_b[0].cpu().numpy().astype(float)
        tgt = creep_law(q, self.prev, self.goal, self.st, self.c, ch.dt)
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
            self.src.fresh_next = i == 0 or c.get('fresh_frames', 'all') != 'first'
            ms.append(self.one(self.src, f'{tag}_{i}'))
            self.creep.tick()
        self.src.fresh_next = True
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
        rvs = [m.rv_deg for m in good if m.rv_deg is not None]
        out.rv_deg = np.median(np.stack(rvs), 0) if rvs else None
        if len(good) < min(int(c['min_good']), len(ms)):
            out.ok, out.reason = False, f'{len(good)} of {len(ms)} frames measured'
        return out, ms

    # ------------------------------------------------------------------ run
    def log_m(self, m, ms, m2, tip, dec, t):
        rec = dict(cycle=self.logic.cycle, tick=self.ch.k, t_stage=t, ok=m.ok, reason=m.reason, conf=m.conf, across=m.across_mm, up=m.up_mm, gap=m.gap_mm, sigma=m.sigma_mm, spread=getattr(m, 'spread', None),
                   frames=[dict(ok=x.ok, reason=x.reason, conf=x.conf, v=[x.across_mm, x.up_mm, x.gap_mm], ms=x.extra.get('ms'), truth=x.extra.get('truth'), panel_rv_deg=x.extra.get('panel_rv_deg')) for x in ms],
                   tip_rel=tip.tolist(), q_b=self.ch.q_b[0].tolist(), lead=self.creep.lead.tolist(), grip=float(self.ch.state['grip'][0]), decision=dec['kind'], decision_reason=dec.get('reason'), delta=np.asarray(dec.get('delta', np.zeros(3))).tolist(),
                   rv_deg=None if m.rv_deg is None else np.asarray(m.rv_deg).tolist(), rv_fused=None if self.logic.last_rv is None else self.logic.last_rv.tolist(), rot_applied=self.logic.rot_applied.tolist(), fused=None if self.logic.last_fused is None else self.logic.last_fused.tolist(), slope=self.logic.slope(), second=None if m2 is None else dict(ok=m2.ok, v=[m2.across_mm, m2.up_mm, m2.gap_mm]))
        tr = self.ch.truth()
        if tr is not None:
            rec['truth_mm'] = np.asarray(tr['plug_in_port_mm'], float).tolist()
            rec['truth_rot_deg'] = np.asarray(tr['plug_rot_deg'], float).tolist()
        self.ch.log_event('align_measure', **rec)
        print(f'align {self.logic.cycle:2d}: m {m.across_mm} {m.up_mm} {m.gap_mm} -> {dec["kind"]} {np.round(dec.get("delta", np.zeros(3)), 2).tolist()} {dec.get("reason", "")}', flush=True)

    def turn(self, rot_deg):
        """New orientation goal: rotate the hand by rot_deg (port axes) about the tip; the next step takes it there."""
        from scipy.spatial.transform import Rotation
        Rw = self.eR @ Rotation.from_rotvec(np.asarray(rot_deg, float), degrees=True).as_matrix() @ self.eR.T
        self.R0 = torch.as_tensor(Rw, dtype=torch.float32) @ self.R0
        self.ch.log_event('align_turn', rot_deg=np.asarray(rot_deg, float).tolist(), tick=self.ch.k)

    def step(self, delta):
        """One capped move, verified on FK. Returns dict: moved (3), fraction of the asked motion, timed out."""
        c, ch, cr = self.c, self.ch, self.creep
        start = self.tip_rel()
        goal = self.goal_of(delta)
        if goal is None:
            return dict(moved=np.zeros(3), frac=0., timeout=False, ik=False)
        cr.goal = goal
        n = int(round(c['step_timeout_s'] / ch.dt))
        reached = False
        k = 0
        hist = []
        for k in range(n):
            cr.tick()
            hist.append(self.tip_rel() - start)
            if reach_ok(c, hist, delta, ch.dt):
                reached = True
                break
        if reached and c.get('reach_mode') == 'loose' and c.get('reach_freeze'):      # stop here: the goal becomes the measured pose (no breakaway during the measurement)
            cr.goal = ch.q_b[0].cpu().numpy().astype(float).copy()
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
        self.sim_truth_angle()
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
        self.loop(out, t_start)
        out.update(cycles=self.logic.cycle, slope=self.logic.slope(), seconds=ch.k * ch.dt - t_start)
        out['last'] = None if self.logic.last_m is None else self.logic.last_m.tolist()
        return out

    def loop(self, out, t_start):
        """Cycles until done or fail: rest, measure, gates, one step."""
        c, ch = self.c, self.ch
        last = None
        while True:
            self.still()
            m, ms = self.measure(f'align_{self.logic.cycle:03d}')
            near = c.get('secondary_near_mm')
            want2 = near is None or (m.ok and m.gap_mm is not None and (np.abs(m.vec() - self.logic.t) <= self.logic.tol + float(near)).all())
            m2 = self.one(self.src2, f'align2_{self.logic.cycle:03d}') if self.src2 is not None and want2 else None
            tip = self.tip_rel()
            stuck = bool(last and last.get('timeout') and last.get('frac', 1.) < .25)
            emin = c.get('entry_min_gap_mm')
            if emin is not None and self.logic.cycle == 0 and m.ok and m.gap_mm is not None and m.gap_mm < emin:      # real run 2026-11-03: a 7 mm slip near the case. Slow steps start from a safe distance or not at all.
                out['ok'], out['reason'] = False, f'entry gap {m.gap_mm:.1f} mm below the {emin} mm limit'
                ch.log_event('align_entry_too_close', gap_mm=float(m.gap_mm), limit_mm=emin, tick=ch.k)
                break
            if ch.k * ch.dt - t_start > c['max_time_s']:
                dec = self.logic.fail('time limit')
            elif c['mode'] == 'measure':
                dec = self.logic.update(m, m2, tip, grip=float(ch.state['grip'][0]))
                self.logic.rot_applied = np.zeros_like(self.logic.rot_applied)      # nothing moves in this mode: real run 2026-11-03, this grew to 6.75 deg and would reach max_deg
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
                out['ok'], out['reason'] = False, dec['reason']
                break
            if dec['kind'] == 'done':
                out.update(ok=True, reason=dec['reason'])
                break
            if dec['kind'] == 'step':
                if dec.get('rot_deg') is not None:
                    self.turn(dec['rot_deg'])
                last = self.step(dec['delta'])
            else:
                last = None

    def sim_truth_angle(self):
        """Dry-run experiment only: turn the hand so that the plug has the wanted rotation against the port by sim truth (config sim_truth_angle: [rx, ry, rz] deg, axes selectable)."""
        c, ch = self.c, self.ch
        want = c.get('sim_truth_angle')
        if want is None or not getattr(ch.io, 'is_sim', False):
            return
        from blind_insertion.envs import kinematics as K
        tr = ch.truth()
        rv_now = torch.deg2rad(torch.as_tensor(np.asarray(tr['plug_rot_deg'], float), dtype=torch.float32))
        rv_want = torch.deg2rad(torch.as_tensor(np.asarray(want, float), dtype=torch.float32))
        pR = torch.as_tensor(tr['pc_R'], dtype=torch.float32)
        def mat(v):
            a = v.norm()
            return torch.eye(3) if float(a) < 1e-9 else K.axis_angle_mat(v / a, a)
        dR = pR @ mat(rv_want) @ mat(rv_now).T @ pR.T      # world rotation that brings the plug there
        self.R0 = dR @ self.R0
        d = self.goal_of(np.zeros(3))
        if d is not None:
            self.creep.goal = d
            for _ in range(int(2. / ch.dt)):
                self.creep.tick()
        self.still()
        ch.log_event('align_sim_truth_angle', want=list(want), before=tr['plug_rot_deg'], after=ch.truth()['plug_rot_deg'])
        p0, R0 = self.fk(ch.q_b[0])
        self.R0, self.tip0 = R0, (p0 - R0 @ self.s).numpy().astype(float)
        self.creep.goal = ch.q_b[0].cpu().numpy().astype(float).copy()

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
        # the press and the lead clip can move the hand: measure again, correct under the press if needed (few cycles, the press clock runs)
        self.logic.cycle = max(0, c['max_cycles'] - c['post_press_cycles'])
        self.logic.confirm = -1                               # three in a row under the press
        self.logic.last_m, self.logic.last_e = None, None
        self.logic.fused = []                                 # the press moved the plug against the port (dry run: 0.6 mm up), old cycles do not count
        self.logic.fused_rv = []
        res0 = dict(ok=False, reason='')
        self.loop(res0, ch.k * ch.dt)
        ok = res0['ok']
        m = PortMeasure(False, res0['reason'])
        if ok:
            m = PortMeasure(True, 'last', 1., *self.logic.final, None)
        lo_ = np.asarray(c['handover_lead_rad'], float)
        self.creep.lead = np.clip(self.creep.lead, lo_[:, 0], lo_[:, 1])
        if c.get('inhand_refresh', True) and ok:        # the in-hand net trained at the 6 mm stop: read it again at the hand-over
            self.inhand_refresh()
        q = ch.q_b[0].cpu().numpy().astype(float)
        cmd = torch.as_tensor(q + self.creep.lead, dtype=torch.float32, device=ch.dev)
        res = dict(ok=ok, reason=res0['reason'], gap_mm=float(m.gap_mm) if m.ok else None, lat_mm=[float(m.across_mm), float(m.up_mm)] if m.ok else None, lead=self.creep.lead.tolist())
        ch.log_event('align_handover', **res)
        return (cmd[None] if ok else None), res

    def inhand_refresh(self):
        """In-hand net read again at the hand-over gap (it trained at the 6 mm stop); sets ch.inhand_est6."""
        c, ch = self.c, self.ch
        if True:
            import torch.nn.functional as F
            net, scale, _ = ch.inhand_net
            fr = ch.render(('wrist_right',), tag='align_inhand')['wrist_cam']
            img = F.interpolate(fr.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
            with torch.no_grad():
                new = net(img) * scale
            rec = dict(before=ch.inhand_est6.reshape(-1).tolist(), after=new.reshape(-1).tolist())
            tr = ch.truth()
            if tr is not None:        # dry run: the K-J input against the true in-hand pose
                from blind_insertion.envs.inhand import inhand_error6, nominal_reader_in_site
                R0_, p0_ = nominal_reader_in_site(torch.device('cpu'))
                ip, iR = torch.as_tensor(tr['inhand_p'], dtype=torch.float32), torch.as_tensor(tr['inhand_R'], dtype=torch.float32)
                from blind_insertion.envs import kinematics as K
                true6 = torch.cat([(ip - p0_.float().reshape(3)) * 1e3, torch.rad2deg(K.rot_log(iR[None] @ R0_.float().reshape(1, 3, 3).transpose(1, 2))[0])])
                rec['true6'] = true6.tolist()
                if c.get('sim_truth_inhand'):
                    new = true6[None].to(new.device)
            ch.log_event('align_inhand', **rec)
            ch.inhand_est6 = new

    def back_off(self):
        if self.c.get('back_off_creep'):
            return self.back_off_creep()
        return back_off_closed(self.ch, self.eR, self.c['back_off_mm'], self.c['back_off_v_mm_s'], self.ql if not self.ch.pressing else self.q_hold)


    def back_off_creep(self):
        """Straight out along the port axis with the creep law: goal + lead, the holding lead stays. Real run 2026-11-03: setting the target to the measured pose dropped joint 5's 34 mrad lead,
        the wrist sagged 4.6 mm in 0.5 s, and joints 2 and 3 stayed in their friction band while the wrist made the whole move (6.6 mm low at the end)."""
        c, ch, cr = self.c, self.ch, self.creep
        chain = ch.scene.chain['right']
        dev = ch.scene.device
        q0 = ch.q_b[0:1].to(dev).float()
        p0, R0 = chain.fk(q0)
        z = torch.as_tensor(np.asarray(self.eR, float).reshape(3, 3)[:, 2], dtype=torch.float32, device=p0.device)
        mm, v = float(c['back_off_mm']), float(c['back_off_v_mm_s'])
        n = max(1, int(round(mm / (v * max(ch.speed, 1e-3)) / ch.dt)))
        q = q0
        cr.goal = q0[0].cpu().numpy().astype(float).copy()
        for k in range(n):
            u = (k + 1) / n
            u = u * u * (3 - 2 * u)
            q, _, _ = chain.ik(p0 + z[None] * mm * 1e-3 * u, R0, q, iters=10)
            cr.goal = q[0].cpu().numpy().astype(float).copy()
            cr.tick()
        for _ in range(int(round(1. / ch.dt))):       # let the joints that lag arrive
            cr.tick()
        ch.log_event('back_off', mm=mm, tick=ch.k, tip_q=ch.q_b[0].tolist(), creep=True, lead=cr.lead.tolist())


SERVO_DEFAULTS = dict(
    gain=1.5, ang_gain=1.0, vmax_mm_s=8.0, vmin_mm_s=1.5, wmax_deg_s=4.0, deadband_mm=0.4, deadband_deg=0.4, smooth=0.2, ki=0.0,
    target_cap_rad=0.05, damping=1e-4, ff=True, still_mm_s=2.0, still_window_s=0.1, max_frame_age_s=0.5, max_blind_s=3.0,
    max_tip_speed_mm_s=60.0, max_time_s=30.0, post_press_s=8.0, cam_latency_s=0.03, sim_latency_ticks=5, sim_period_ticks=5,
    hold_s=0.3, vision_worker='process',
    lead_cut=False, lead_cut_release=.3, lead_cut_eps_rad=.0005,    # cut a joint's target lead when it breaks free from rest (stick-slip; docs/real/ALIGN_FAST_20261113.md)
    near_mm=None, vmax_near_mm_s=3.0,                               # speed cap vmax_near_mm_s when the estimate is within near_mm of the goal; None: off
)


def servo_lead_cut(tgt, q, q_prev, st, sv):
    """Servo option lead_cut (the creep's cut, per tick): a joint that was at rest and now moves toward its target keeps only lead_cut_release of its
    target lead, so less stored spring is left when it breaks free (real 2026-11-10 servo runs: joints caught up their 5-35 mrad lag in jumps,
    tip speed p99 24-25 mm/s for a commanded 9-11 mm/s). Returns dict(tgt, rest)."""
    tgt, q, q_prev = (np.asarray(x, float) for x in (tgt, q, q_prev))
    e = tgt - q
    eps = float(sv['lead_cut_eps_rad'])
    rest = np.ones(6, bool) if st is None else st['rest']
    broke = rest & ((q - q_prev) * np.sign(e) >= eps) & (np.abs(e) > eps)
    out = np.where(broke, q + float(sv['lead_cut_release']) * e, tgt)
    return dict(tgt=out, rest=np.abs(q - q_prev) < eps * .5, cuts=int(broke.sum()))


class ServoStage(AlignStage):
    """Continuous vision servo (config align.servo; docs/real/K2_PROGRESS.md "Servo deploy").

    Every 20 ms tick: fused estimate = the newest port-vision result + the believed FK tip motion since that frame's capture tick (port axes);
    plug-tip twist = -gain x (estimate - goal) (speed floor outside a dead band, speed caps, first-order smoothing; the same law as
    blind_insertion/envs/align_task.ServoPolicy, sim: 99.6 % aligned, 4.4 s median, n = 256); joint increment by the believed tip Jacobian
    (damped least squares); the target accumulates within target_cap_rad of the measured joints; the chain's Coulomb feed-forward is on.
    Vision runs in a worker thread (vision_thread.py): the 50 Hz ticks never wait for it.

    Gates kept from the step stage: entry gap, gripper change, distance from the stop, minimum clearance, error growth, measurement against FK,
    measurement scale against FK motion (regression over the frames; probes only on an axis that did not move enough), angle limit,
    second-source disagreement at the hand-over, time limit. New: no valid frame for max_frame_age_s -> hold; for max_blind_s -> stop;
    tip faster than max_tip_speed_mm_s (0.1 s window) -> stop. Done: estimate within tol for confirm_n frames in a row and the tip still
    (0.1 s window: one-tick FK differences carry about 5 mm/s of encoder quantization)."""

    def __init__(self, ch, cmd, q_hold, eR, c):
        super().__init__(ch, cmd, q_hold, eR, c)
        sv = dict(SERVO_DEFAULTS)
        sv.update({k: v for k, v in (c.get('servo') or {}).items() if not k.startswith('_')})
        self.sv = sv
        self.hist = []                 # (tick, t, tip (3), R (3x3)) per tick
        self.tgt = None
        self.worker = None

    # ------------------------------------------------------------------ helpers
    def _rec(self):
        ch = self.ch
        p, R = self.fk(ch.q_b[0])
        tip = (p - R @ self.s).numpy().astype(float)
        rel = self.eR.T @ (tip - self.tip0) * 1e3
        self.hist.append((ch.k, float(ch.state.get('t', ch.k * ch.dt)), rel, R.numpy().astype(float)))
        if len(self.hist) > 400:
            self.hist = self.hist[-400:]
        return rel, R.numpy().astype(float)

    def _at_tick(self, tick):
        for h in reversed(self.hist):
            if h[0] <= tick:
                return h
        return self.hist[0]

    def _tick_of_time(self, t):
        for h in reversed(self.hist):
            if h[1] <= t:
                return h[0]
        return self.hist[0][0]

    def _speed(self):
        n = max(1, int(round(self.sv['still_window_s'] / self.ch.dt)))
        if len(self.hist) <= n:
            return 0.
        return float(np.linalg.norm(self.hist[-1][2] - self.hist[-1 - n][2]) / (n * self.ch.dt))

    def _rot_since_deg(self, R_cap, R_now):
        from scipy.spatial.transform import Rotation
        dR = self.eR.T @ (R_now @ R_cap.T) @ self.eR
        return Rotation.from_matrix(dR).as_rotvec(degrees=True)

    def _send(self, ql):
        ch = self.ch
        q = ch.q_b[0].cpu().numpy().astype(float)
        cap = float(self.sv['target_cap_rad'])
        self.tgt = np.clip(self.tgt, q - cap, q + cap)
        ch.hstep(torch.as_tensor(self.tgt, dtype=torch.float32, device=ch.dev), ql, 'align')

    def _twist_to_dq(self, v_mm_s, w_deg_s):
        ch = self.ch
        dev = ch.scene.device
        q = ch.q_b[0:1].to(dev).float()
        p, R, J = self.chain.fk(q, with_jac=True)
        r = (-(R[0] @ self.s.to(dev))).cpu().numpy().astype(float)          # tip - TCP (tip = p - R s)
        v = self.eR @ (np.asarray(v_mm_s, float) * 1e-3)
        w = self.eR @ np.radians(np.asarray(w_deg_s, float))
        x = np.r_[v - np.cross(w, r), w] * ch.dt
        Jn = J[0].cpu().numpy().astype(float)
        return Jn.T @ np.linalg.solve(Jn @ Jn.T + float(self.sv['damping']) * np.eye(6), x)

    # ------------------------------------------------------------------ vision
    def _start_vision(self):
        ch, sv = self.ch, self.sv
        self.last_frame_k = None
        if getattr(ch.io, 'is_sim', False) or not isinstance(self.src, PortVisionSource):
            from deploy.bimanual.vision_thread import DeferredWorker
            lat = int(sv['sim_latency_ticks']) if getattr(ch.io, 'is_sim', False) else 0
            self.worker = DeferredWorker(lambda tag: self.one(self.src, tag), latency_ticks=lat, period_ticks=int(sv['sim_period_ticks']))
            self.threaded = False
        else:
            self.src.plug_in_camera()          # load once on this thread
            err = lambda e: PortMeasure(False, f'{type(e).__name__}: {e}')  # noqa: E731
            if sv.get('vision_worker', 'process') == 'process':      # thread: port_vision holds the interpreter lock for part of its 82 ms (loop p99 33 ms, max 50 ms)
                from deploy.bimanual.vision_thread import ProcessWorker
                self.worker = ProcessWorker(self.src.kw, lambda: self.src.plug_z, self.src.convert, err)
            else:
                from deploy.bimanual.vision_thread import ThreadWorker
                self.worker = ThreadWorker(self.src.measure_raw, err)
            self.threaded = True
            ch.log_event('align_servo_vision_worker', worker=sv.get('vision_worker', 'process'))

    def _vision_tick(self, tag):
        """Submit the newest frame when the worker is free; return a finished Result or None. Never blocks."""
        ch = self.ch
        res = self.worker.poll(now_tick=ch.k)
        if not self.worker.busy:
            if self.threaded:
                pol, raw, t, cnt = ch.io.frames(('wrist_right',))['wrist_right']
                if raw is not None and cnt != self.last_frame_k:
                    self.last_frame_k = cnt
                    tick = self._tick_of_time(t - float(self.sv['cam_latency_s']))
                    if ch.logger is not None:
                        ch.logger.frame('wrist_right', raw, tag, t_capture=t, t_use=ch.state['t'], kind='raw', bgr=True, count=cnt, tick=ch.k)
                    self.worker.submit(raw, tick, t, tag)
            else:
                self.worker.submit(None, ch.k, ch.k * ch.dt, tag)
        return res

    def _valid(self, m):
        c = self.c
        if m is None or not m.ok or m.gap_mm is None or m.across_mm is None or m.up_mm is None:
            return False, f'no measurement ({m.reason if m else "none"})'
        if m.conf < c['min_conf']:
            return False, f'low confidence {m.conf:.2f}'
        if m.sigma_mm is not None and m.sigma_mm > c['max_sigma_mm']:
            return False, f'sigma {m.sigma_mm:.2f} mm'
        return True, ''

    # ------------------------------------------------------------------ run
    def run(self, log):
        c, ch = self.c, self.ch
        ch.log_event('align_start', cfg={k: v for k, v in c.items() if not k.startswith('_')}, source=c['source'] if not getattr(ch.io, 'is_sim', False) else c['sim_source'], servo=self.sv)
        t_start = ch.k * ch.dt
        ql = ch.q_b[1].clone() if c['press_release'] else self.q_hold
        q = ch.q_b[0].cpu().numpy().astype(float)
        lead0 = np.clip((self.cmd0.cpu().numpy().astype(float) - q), -np.asarray(c['lead_max_rad'], float), np.asarray(c['lead_max_rad'], float))
        ff_was = ch.rt.comp['right'] is not None
        if self.sv['ff'] and ch.fric is not None:
            if not ff_was:            # turning the feed-forward on adds torque toward the target: take it out of the target first
                tau = FC.feedforward(torch.as_tensor(lead0, dtype=torch.float32, device=ch.dev), *ch.fric)
                lead0 = lead0 - (tau / ch.scene.kp[0].to(ch.dev)).cpu().numpy().astype(float)
            ch.rt.comp['right'] = ch.fric
        elif ff_was and ch.fric is not None:
            tau = FC.feedforward(self.cmd0 - ch.q_b[0], *ch.fric)
            lead0 = lead0 + (tau / ch.scene.kp[0].to(ch.dev)).cpu().numpy().astype(float)
            ch.rt.comp['right'] = None
        self.tgt = q + lead0
        self.ql = ql
        if c['press_release']:
            ch.pressing = False
            ch.log_event('align_press_off', q_left=ql.tolist(), tick=ch.k)
        ch.log_event('align_windup_at_entry', lead=lead0.tolist(), cmd=self.cmd0.tolist(), q_b=ch.q_b[0].tolist(), ff=bool(ch.rt.comp['right'] is not None))
        p0, R0 = self.fk(ch.q_b[0])
        self.R0, self.tip0 = R0, (p0 - R0 @ self.s).numpy().astype(float)
        self.R_start = R0.numpy().astype(float)
        out = dict(ok=False, reason='', cycles=0, gap_mm=None, lat_mm=None)
        for _ in range(int(round(self.sv['hold_s'] / ch.dt))):
            self._rec(); self._send(self.ql)
        if hasattr(self.src, 'prepare'):
            def between():
                self._rec(); self._send(self.ql)
            pr = self.src.prepare(ch, between)
            ch.log_event('align_prepare', **pr)
            print(f'align: shell depth for this grasp {pr}', flush=True)
        self._start_vision()
        try:
            self.servo(out, t_start, float(self.sv['max_time_s']), confirm_n=int(c['confirm_n']), ql_fn=lambda k: self.ql, entry=True)
        finally:
            if self.worker is not None and c['mode'] != 'full':
                self.worker.close()
        out.update(cycles=self.n_frames, slope=self.logic.slope(), seconds=ch.k * ch.dt - t_start)
        out['last'] = None if self.est is None else self.est[:3].tolist()
        return out

    def servo(self, out, t_start, max_s, confirm_n, ql_fn, entry=False, count_from_tick=None):
        """The servo loop until done or fail. Fills out (ok, reason). self.est: last fused estimate (6: mm, deg).
        count_from_tick: frames captured before this tick do not count toward done (the hand-over: only frames under the full press)."""
        c, ch, sv, lg = self.c, self.ch, self.sv, self.logic
        tgt_mm = np.asarray(c['target_mm'], float)
        tol = np.asarray(c['tol_mm'], float)
        ax = list((c.get('angle') or {}).get('axes', []))
        ang = c.get('angle') or {}
        self.n_frames = getattr(self, 'n_frames', 0)
        self.est = getattr(self, 'est', None)
        meas = None                    # (v (3), rv (3) or None, capture tick, tip at capture, R at capture)
        last_valid_t = ch.k * ch.dt
        prev_tw = np.zeros(6)
        integ = np.zeros(3)
        confirm, pred_bad, best_e = 0, 0, float('inf')
        prev_cap = None
        grip0 = getattr(self, 'grip0', None)
        probe = []                     # queued goal offsets (mm) for the scale gate
        probe_off = np.zeros(3)
        extra = int(c['extra_probes'])
        k0 = ch.k
        while True:
            t_now = ch.k * ch.dt
            tip, R = self._rec()
            res = self._vision_tick(f'align_servo_{self.n_frames:04d}')
            new_valid = False
            if res is not None:
                self.n_frames += 1
                ok, why = self._valid(res.m)
                h = self._at_tick(res.tick)
                if ok:
                    v = res.m.vec()
                    if entry and meas is None and prev_cap is None and c.get('entry_min_gap_mm') is not None and v[2] < c['entry_min_gap_mm']:
                        out['ok'], out['reason'] = False, f'entry gap {v[2]:.1f} mm below the {c["entry_min_gap_mm"]} mm limit'
                        ch.log_event('align_entry_too_close', gap_mm=float(v[2]), limit_mm=c['entry_min_gap_mm'], tick=ch.k)
                        break
                    if prev_cap is not None:         # measurement against FK between two frames
                        dm, dtip = v - prev_cap[0], h[2] - prev_cap[1]
                        lim = c['pred_mm'] + 2 * float(res.m.sigma_mm or 0.)
                        pred_bad = pred_bad + 1 if (np.abs(dm - dtip) > lim).any() else 0
                        if pred_bad >= c['pred_n']:
                            out['ok'], out['reason'] = False, f'measurement against FK: changed {dm.round(2).tolist()} mm, hand moved {dtip.round(2).tolist()} mm'
                            break
                    prev_cap = (v, h[2])
                    if not lg.scale_ok:
                        lg.levels.append(np.r_[h[2], v])
                    rv = None if res.m.rv_deg is None else np.asarray(res.m.rv_deg, float)
                    meas = (v, rv, res.tick, h[2], h[3])
                    last_valid_t = t_now
                    new_valid = True
                ch.log_event('align_servo_frame', tick=ch.k, cap_tick=int(res.tick), ok=bool(ok), why=why, ms=float(res.ms), v=None if not ok else res.m.vec().tolist(),
                             rv_deg=None if (not ok or res.m.rv_deg is None) else np.asarray(res.m.rv_deg).tolist(), truth=res.m.extra.get('truth') if res.m is not None else None)
            # gates every tick
            if t_now - t_start > max_s:
                out['ok'], out['reason'] = False, 'time limit'
                break
            g = float(ch.state['grip'][0])
            grip0 = g if grip0 is None else grip0
            self.grip0 = grip0
            if abs(g - grip0) > c['grip_change']:
                out['ok'], out['reason'] = False, f'gripper reading changed {grip0:.3f} -> {g:.3f}'
                break
            if np.abs(tip[:2]).max() > c['max_lat_mm'] or -tip[2] > c['max_advance_mm']:
                out['ok'], out['reason'] = False, f'moved too far from the stop ({tip.round(1).tolist()} mm)'
                break
            spd = self._speed()
            if spd > sv['max_tip_speed_mm_s']:
                out['ok'], out['reason'] = False, f'tip moved {spd:.0f} mm/s (limit {sv["max_tip_speed_mm_s"]})'
                break
            age = t_now - last_valid_t
            if meas is None or age > sv['max_blind_s']:
                if age > sv['max_blind_s']:
                    out['ok'], out['reason'] = False, f'no valid frame for {age:.1f} s'
                    break
                self._send(ql_fn(ch.k - k0))
                continue
            v, rv, cap, tip_cap, R_cap = meas
            est = v + (tip - tip_cap)
            rot = self._rot_since_deg(R_cap, R)
            est_rv = None if rv is None else rv + rot
            self.est = np.r_[est, est_rv if est_rv is not None else np.full(3, np.nan)]
            e = float(np.linalg.norm(est - tgt_mm))
            best_e = min(best_e, e)
            if e > best_e + c['grow_abs_mm']:
                out['ok'], out['reason'] = False, f'correction grows: {e:.1f} mm, best {best_e:.1f} mm'
                break
            floor_ = c['min_clear_fail_mm'] if c.get('min_clear_fail_mm') is not None else c['min_clear_mm']
            if est[2] < floor_:
                out['ok'], out['reason'] = False, f'gap {est[2]:.1f} mm below the {floor_} mm limit'
                break
            applied = self._rot_since_deg(self.R_start, R)
            if ang and np.abs(applied[ax]).max() > ang['max_deg']:
                out['ok'], out['reason'] = False, f'angle correction above {ang["max_deg"]} deg: {applied.round(1).tolist()}'
                break
            if ang and ang.get('required', True) and est_rv is None:
                out['ok'], out['reason'] = False, 'no angle measurement'
                break
            # goal: funnel (hold the gap until lined up), probes for the scale gate
            goal = tgt_mm + probe_off
            lat = float(np.hypot(*(est[:2] - tgt_mm[:2])))
            if lat > c['funnel_lat_mm']:
                goal = goal.copy()
                goal[2] = max(goal[2], c['gap_hold_mm'])
            err = est - goal
            rot_err = np.zeros(3)
            if est_rv is not None and ax:
                rot_err[ax] = est_rv[ax]
            # done / probes
            within = (np.abs(err) <= tol).all() and (np.abs(rot_err) <= ang.get('tol_deg', 1e9)).all() and not probe_off.any()
            still = spd <= sv['still_mm_s']
            if new_valid:
                early = count_from_tick is not None and meas[2] < count_from_tick
                confirm = confirm + 1 if (within and still and not early) else 0
            if probe_off.any() and np.abs(err).max() <= max(.5, float(tol.min())):     # probe reached: next offset (back to 0 last)
                probe_off = probe.pop(0) if probe else np.zeros(3)
                confirm = 0
            if confirm >= confirm_n:
                if not lg.scale_ok:
                    okk, axn, why = lg.slope_check()
                    if not okk and axn is not None and extra > 0:
                        extra -= 1
                        d = np.zeros(3)
                        d[axn] = c['extra_probe_mm'][axn]
                        probe_off, probe = d, [np.zeros(3)]
                        confirm = 0
                        ch.log_event('align_servo_probe', axis=int(axn), mm=float(d[axn]), why=why, tick=ch.k)
                    elif not okk:
                        out['ok'], out['reason'] = False, 'scale gate: ' + why
                        break
                    else:
                        lg.scale_ok = True
                if lg.scale_ok:
                    if self.src2 is not None:
                        m2 = self.one(self.src2, f'align2_servo_{self.n_frames:04d}')
                        if c.get('secondary_required') and not (m2 and m2.ok):
                            out['ok'], out['reason'] = False, 'second source has no reading at the hand-over'
                            break
                        if m2 is not None and m2.ok and c['disagree_mode'] == 'gate' and (np.abs(est - m2.vec()) > np.asarray(c['disagree_mm'], float)).any():
                            out['ok'], out['reason'] = False, f'sources disagree at the hand-over {np.abs(est - m2.vec()).round(2).tolist()} mm'
                            break
                    lg.final = est.copy()
                    lg.last_m = est.copy()
                    out.update(ok=True, reason='within tolerance')
                    break
            # twist law (ServoPolicy)
            inside = float(np.linalg.norm(err)) <= sv['deadband_mm']
            if not inside:
                integ = integ + err * ch.dt
            vp = -sv['gain'] * err - sv['ki'] * integ
            nv = float(np.linalg.norm(vp))
            if sv['vmin_mm_s'] > 0 and 1e-6 < nv < sv['vmin_mm_s']:
                vp = vp / nv * sv['vmin_mm_s']
            if inside:
                vp = np.zeros(3)
            if est[2] < c['min_clear_mm']:                  # too close: out first
                vp[2] = max(vp[2], sv['vmin_mm_s'])
            va = -sv['ang_gain'] * rot_err
            if float(np.linalg.norm(rot_err)) <= sv['deadband_deg']:
                va = np.zeros(3)
            vmax = sv['vmax_mm_s']
            if sv.get('near_mm') is not None and float(np.linalg.norm(err)) < sv['near_mm']:
                vmax = min(vmax, sv['vmax_near_mm_s'])
            tw = np.r_[np.clip(vp, -vmax, vmax), np.clip(va, -sv['wmax_deg_s'], sv['wmax_deg_s'])]
            if age > sv['max_frame_age_s']:                # stale estimate: hold
                tw = np.zeros(6)
            tw = prev_tw + sv['smooth'] * (tw - prev_tw)
            prev_tw = tw
            self.tgt = self.tgt + self._twist_to_dq(tw[:3], tw[3:])
            if sv.get('lead_cut'):
                qn = ch.q_b[0].cpu().numpy().astype(float)
                self._lc = servo_lead_cut(self.tgt, qn, getattr(self, '_q_prev', qn), getattr(self, '_lc', None), sv)
                self.tgt = self._lc['tgt']
                self._q_prev = qn
            if ch.logger is not None:
                ch.logger.row('servo', k=ch.k, est=est, err=err, rot_err=rot_err, twist=tw, tip=tip, speed=spd, age=age, confirm=confirm)
            self._send(ql_fn(ch.k - k0))
        tr = ch.truth()
        ch.log_event('align_servo_end', ok=bool(out['ok']), reason=out['reason'], tick=ch.k, est=None if self.est is None else self.est.tolist(),
                     truth_mm=None if tr is None else np.asarray(tr['plug_in_port_mm'], float).tolist(), truth_rot_deg=None if tr is None else np.asarray(tr['plug_rot_deg'], float).tolist(),
                     frames=self.n_frames, seconds=ch.k * ch.dt - t_start)
        print(f'align servo: {"OK" if out["ok"] else "STOP"} {out["reason"]} ({self.n_frames} frames, {ch.k * ch.dt - t_start:.1f} s, est {None if self.est is None else self.est[:3].round(2).tolist()})', flush=True)

    def handover(self, log):
        """As AlignStage.handover: press back on (servo holding), a few frames under the press, the joint target lead clipped into the
        range K-J saw, feed-forward folded into the target and off. Returns (cmd (1, 6) or None, dict)."""
        c, ch = self.c, self.ch
        n = int(round(c['repress_s'] / ch.dt))
        ql0, ql1 = self.ql.clone(), self.q_hold
        if c['press_release']:
            ch.pressing = True

            def ql_fn(k):
                u = min(1., (k + 1) / n)
                u = u * u * (3 - 2 * u)
                return ql0 + (ql1 - ql0) * u
        else:
            def ql_fn(k):
                return ql1
        res0 = dict(ok=False, reason='')
        settle = int(round(c['settle_s'] / ch.dt))
        self.servo(res0, ch.k * ch.dt, float(self.sv['post_press_s']) + c['repress_s'], confirm_n=int(c['confirm_n']) + 1, ql_fn=ql_fn,
                   count_from_tick=ch.k + (n if c['press_release'] else 0) + settle)
        if c['press_release']:
            ch.log_event('align_press_on', tick=ch.k)
        self.ql = ql1
        self.worker.close()
        ok = res0['ok']
        if ok and float(self.sv.get('preload_mm', 0.)) > 0:
            self._preload(float(self.sv['preload_mm']), float(self.sv.get('preload_v_mm_s', 2.)))
        q = ch.q_b[0].cpu().numpy().astype(float)
        lead = self.tgt - q
        if ch.rt.comp['right'] is not None and ch.fric is not None:       # fold the feed-forward into the target, then off (as the step stage)
            tau = FC.feedforward(torch.as_tensor(lead, dtype=torch.float32, device=ch.dev), *ch.fric)
            lead = lead + (tau / ch.scene.kp[0].to(ch.dev)).cpu().numpy().astype(float)
            ch.rt.comp['right'] = None
        lo = np.asarray(c['handover_lead_rad'], float)
        lead = np.clip(lead, lo[:, 0], lo[:, 1])
        self.tgt = q + lead
        if c.get('inhand_refresh', True) and ok:
            self.inhand_refresh()
        est = self.est
        cmd = torch.as_tensor(q + lead, dtype=torch.float32, device=ch.dev)
        res = dict(ok=ok, reason=res0['reason'], gap_mm=float(est[2]) if ok else None, lat_mm=[float(est[0]), float(est[1])] if ok else None, lead=lead.tolist(), servo=True)
        ch.log_event('align_handover', **res)
        return (cmd[None] if ok else None), res

    def _preload(self, mm, v_mm_s):
        """Backlash preload (servo.preload_mm; docs/real/RETRY_SPEED_20261108.md): the last motion before the hand-over is a straight move of
        mm toward the port (gap down), so the joint gears carry their lash in the insertion direction when the policy starts. Run the servo
        to a goal gap preload_mm larger (align.target_mm). Open loop on the believed FK (a displacement); the fused estimate moves with it."""
        ch = self.ch
        tip0, _ = self._rec()
        n = max(1, int(round(mm / v_mm_s / ch.dt)))
        for k in range(n):          # speed profile sin: zero at both ends, mean v_mm_s, integral mm
            self.tgt = self.tgt + self._twist_to_dq(np.array([0., 0., -mm / (n * ch.dt) * np.pi / 2 * np.sin(np.pi * (k + .5) / n)]), np.zeros(3))
            self._rec()
            self._send(self.ql)
        for _ in range(int(round(max(float(self.c['settle_s']), .3) / ch.dt))):
            self._rec()
            self._send(self.ql)
        tip1, _ = self._rec()
        d = tip1 - tip0
        if self.est is not None:
            self.est[:3] = self.est[:3] + d
        tr = ch.truth()
        ch.log_event('align_preload', mm=mm, tip_moved_mm=d.tolist(), est=None if self.est is None else self.est.tolist(),
                     truth_mm=None if tr is None else np.asarray(tr['plug_in_port_mm'], float).tolist(), tick=ch.k)

    def back_off(self):
        """Straight out along the port axis keeping the joint target lead (as back_off_creep), then the feed-forward folded into the target and off."""
        c, ch = self.c, self.ch
        if getattr(self, 'worker', None) is not None:
            self.worker.close()
        chain = self.chain
        dev = ch.scene.device
        q0 = ch.q_b[0:1].to(dev).float()
        p0, R0 = chain.fk(q0)
        z = torch.as_tensor(self.eR[:, 2], dtype=torch.float32, device=p0.device)
        mm, v = float(c['back_off_mm']), float(c['back_off_v_mm_s'])
        n = max(1, int(round(mm / (v * max(ch.speed, 1e-3)) / ch.dt)))
        if self.tgt is None:
            self.tgt = q0[0].cpu().numpy().astype(float)
        lead = self.tgt - q0[0].cpu().numpy().astype(float)
        ql = self.ql if not ch.pressing else self.q_hold
        q = q0
        for k in range(n):
            u = (k + 1) / n
            u = u * u * (3 - 2 * u)
            q, _, _ = chain.ik(p0 + z[None] * mm * 1e-3 * u, R0, q, iters=10)
            self.tgt = q[0].cpu().numpy().astype(float) + lead
            self._send(ql)
        for _ in range(int(round(1. / ch.dt))):
            self._send(ql)
        if ch.rt.comp['right'] is not None and ch.fric is not None:
            qm = ch.q_b[0].cpu().numpy().astype(float)
            tau = FC.feedforward(torch.as_tensor(self.tgt - qm, dtype=torch.float32, device=ch.dev), *ch.fric)
            self.tgt = self.tgt + (tau / ch.scene.kp[0].to(ch.dev)).cpu().numpy().astype(float)
            ch.rt.comp['right'] = None
            for _ in range(int(round(.2 / ch.dt))):
                self._send(ql)
        ch.log_event('back_off', mm=mm, tick=ch.k, tip_q=ch.q_b[0].tolist(), servo=True, lead=(self.tgt - ch.q_b[0].cpu().numpy().astype(float)).tolist())


def back_off_closed(ch, eR, mm, v_mm_s, q_left):
    """Hold the measured pose (no push), then straight out along the port axis, jaws closed. The reader stays in the hand."""
    move_tip_closed(ch, eR, [0., 0., mm], v_mm_s, q_left, kind='back_off')


def move_tip_closed(ch, eR, d_mm, v_mm_s, q_left, kind='tip_move', hold_s=.2):
    """Hold the measured pose for hold_s (no push), then a straight TCP move of d_mm (port axes: across, up, out of the port) at v_mm_s,
    orientation kept, jaws closed (smoothstep profile). Believed FK: a displacement, so the encoder offsets cancel to first order."""
    eR = np.asarray(eR, float).reshape(3, 3)
    for _ in range(int(round(hold_s / ch.dt))):
        ch.hstep(ch.q_b[0].clone(), q_left, ch.label)
    chain = ch.scene.chain['right']
    dev = ch.scene.device
    q0 = ch.q_b[0:1].to(dev).float()
    p0, R0 = chain.fk(q0)
    d = torch.as_tensor(eR @ np.asarray(d_mm, float), dtype=torch.float32, device=p0.device)
    mm = float(np.linalg.norm(d_mm))
    n = max(1, int(round(mm / (v_mm_s * max(ch.speed, 1e-3)) / ch.dt)))
    q = q0
    for k in range(n):
        u = (k + 1) / n
        u = u * u * (3 - 2 * u)
        q, _, _ = chain.ik(p0 + d[None] * 1e-3 * u, R0, q, iters=10)
        ch.hstep(q[0].to(ch.dev), q_left, ch.label)
    ch.log_event(kind, mm=mm, d_mm=[float(x) for x in d_mm], tick=ch.k, tip_q=ch.q_b[0].tolist())


def run_align(ch, cmd, q_hold, ep, eR, log, c):
    """Stage entry. Returns dict(ok, cmd (1, 6) or None, ...); on failure the plug is backed off with closed jaws and q_left in res['q_left']."""
    st = (ServoStage if c.get('servo') and c['mode'] != 'measure' else AlignStage)(ch, cmd, q_hold, eR, c)     # measure mode never moves: the step stage's measure loop
    out = st.run(log)
    out['q_left'] = st.ql
    out['cmd'] = None
    if out['ok'] and c['mode'] == 'full':
        out['cmd'], h = st.handover(log)
        out['handover'] = h
        if out['cmd'] is None:
            out.update(ok=False, reason='after the press: ' + str(h.get('reason')))
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
