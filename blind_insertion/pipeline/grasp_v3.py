"""Guarded grasp of the reader (real pick 2026-11-01: 2 of 8 closes held; docs/real/PICK_GRASP_V3.md).

Root causes (results/real/pick_20261101_analysis.md, v2, grasp-sim): the servo target rolls the closing axis -3.4..-5.6 deg
(held |roll| <= 2.87, empty >= 3.78 deg, post hoc, n = 10); its position estimate sits 3-13 mm below the height where the hand
stops, so the target presses; step closes jump over the reader. The reader lies flat, so the closing axis must be horizontal.

The servo descent (phases 0, 1 of PickExpert) keeps its position correction; every servo rotation is levelled (level_roll). From there this expert runs
its own phases, on the believed encoder frame, batched (sim eval) or one world (deploy/bimanual/chain.py):

  TD      settle, then descend slowly with the wrist on the planned orientation; integral frozen, so the command
          cannot wind into the table. TouchDetector finds the table. No touch by td_max_below_mm: planned height.
  HOLD    table height + td_lift_mm, planned orientation, bounded integral; wrist gains ramp up.
  CLOSE   low-force close until the jaws stall (hold-down active). Reading falls through the held band: abort.
  SQUEEZE ramp to the final squeeze target, dwell.
  TEST    lift test_lift_mm, read the gripper. In the held band: LIFT. Else REOPEN, RISE, and re-servo from the hover.
  LIFT, END  as PickExpert (lift, stability hold). DONE.

Each strategy has a switch (GraspV3Cfg): level_roll, touchdown, wrist_kp (None = off), gentle_close, holddown_mm, max_attempts.
Hooks (task side, see SimHooks in scripts/policy/grasp_v3_eval.py and ChainHooks in deploy/bimanual/chain.py):
  grip_reading() -> (N,) i2rt reading; grip(mode (N,), u (N,)): 0 = plain open/close sign, 1 = gentle, 2 = squeeze at
  ramp fraction u; wrist(alpha (N,)): 0 nominal .. 1 stiff; friction_ff(mask (N,)): feed-forward during the descent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickExpert

G_NONE, G_TD, G_HOLD, G_CLOSE, G_SQUEEZE, G_TEST, G_LIFT, G_END, G_DONE, G_REOPEN, G_RISE, G_GIVEUP = range(12)
NAMES = ('none', 'td', 'hold', 'close', 'squeeze', 'test', 'lift', 'end', 'done', 'reopen', 'rise', 'giveup')


@dataclass
class GraspV3Cfg:
    # 0. closing axis horizontal: the servo rotation keeps tilt and yaw, its roll is limited to level_roll_max_deg
    level_roll: bool = True
    level_roll_max_deg: float = 0.0
    roll_gate_deg: float | None = 2.5  # measured roll above this at the end of HOLD: no close, next attempt
    # 1. guarded touch-down (rule: results/real/pick_20261101_analysis.md; margin 1.31 mm, not validated on a slow descent)
    touchdown: bool = True
    td_settle_s: float = 0.3          # at the last wrist frame, before the descent
    td_speed_m_s: float = 0.01        # the rule needs <= 15 mm/s
    td_max_below_mm: float = 8.0      # deepest command below the planned grasp
    td_drop_mm: float = 8.0           # D: target fall since the anchor
    td_follow_frac: float = 0.2       # anchor reset: the measured TCP followed more than this share of D ...
    td_eps_mm: float = 0.5            # ... plus this
    td_step_mm_s: float = 30.0        # anchor reset: the target moved faster than this
    td_tip_max_mm: float | None = None  # optional: fingertip FK height at the trigger above this = on the reader (FK height has a run-to-run bias: off by default)
    td_lift_mm: float = 0.5
    td_friction_ff: bool = True
    hold_settle_tol: tuple = (6e-4, 0.5)   # m, deg
    hold_min_s: float = 0.3
    hold_max_s: float = 1.5
    hold_integ_cap_rad: float = 0.02
    # 2. wrist gains from HOLD to the end of TEST
    wrist_kp: tuple | None = (40., 30., 30.)
    wrist_kd: tuple | None = (1.5, 2., 2.)
    wrist_ramp_s: float = 0.3
    # 3. hold-down and gentle close
    holddown_mm: float = 3.0          # below the hold point, while the jaws close
    gentle_close: bool = True
    gentle_rate: float = 0.2          # i2rt reading units per s: the target runs down at this rate ...
    gentle_delta: float = 0.12        # ... and stays at most this far below the reading (low force on a position-PD gripper)
    close_min_s: float = 0.8
    close_max_s: float = 3.0
    stall_window_s: float = 0.24
    stall_eps: float = 0.006          # i2rt reading units over the window
    squeeze_s: float = 0.6
    dwell_s: float = 0.3
    # 4. automatic retry
    max_attempts: int = 3
    test_lift_mm: float = 15.0
    test_v_m_s: float = 0.02
    test_dwell_s: float = 0.3
    held_band: tuple = (0.275, 0.36)  # real stall 0.30-0.305, pads load at 0.345, ride-over jumps at 0.246-0.269; sim 0.32
    lost_ticks: int = 5               # during the lift: reading below the band this long = lost (pick11: 2.26 s after the lift start)
    reopen_s: float = 0.8
    rise_mm: float = 40.0
    # no touch-down: the close happens from the plan's settle ('press' keeps its target, 'measured' = measured + hold_offset_mm)
    hold_from: str = 'measured'
    hold_offset_mm: float = 0.0
    # Real commissioning, jaws stay open and the hand rises after HOLD. 'touchdown': on the table. 'air': from the first
    # wrist frame, on a plan raised by air_plan_up_mm: a slow free descent, no contact; the peak D measures the false-positive margin.
    test_mode: str | None = None
    air_plan_up_mm: float = 30.0

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        unknown = set(k for k in d if not k.startswith('_')) - names
        if unknown:
            raise ValueError(f'unknown pick.grasp_v3 keys: {sorted(unknown)}')
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kw)

    def to_dict(self):
        return asdict(self)


class TouchDetector:
    """Table contact from encoders only (rule of results/real/pick_20261101_analysis.md). D = target z fall since an anchor.
    The anchor resets when the measured TCP followed more than follow_frac x D + eps, when the target rises, or when it
    moves faster than step_m_s. Free descent keeps resetting it. Contact when D >= drop_m."""

    def __init__(self, n, drop_m, eps_m, device, follow_frac=.2, step_m_s=.03, dt=.02):
        self.drop_m, self.eps_m, self.frac, self.step = drop_m, eps_m, follow_frac, step_m_s * dt
        self.zc0, self.zm0 = torch.zeros(n, device=device), torch.zeros(n, device=device)
        self.prev = torch.full((n,), float('nan'), device=device)
        self.drop = torch.zeros(n, device=device)

    def reset(self, mask):
        self.prev[mask] = float('nan')
        self.drop[mask] = 0

    def update(self, z_cmd, z_meas):
        dz = self.prev - z_cmd               # target fall this tick; NaN after a reset
        d = self.zc0 - z_cmd
        follow = self.zm0 - z_meas
        rst = ~(dz >= -1e-6) | (dz > self.step) | (follow > self.frac * d + self.eps_m)
        self.zc0 = torch.where(rst, z_cmd, self.zc0)
        self.zm0 = torch.where(rst, z_meas, self.zm0)
        self.prev = z_cmd.clone()
        self.drop = torch.where(rst, torch.zeros_like(d), d)
        return self.drop >= self.drop_m, self.drop


class NullHooks:
    def grip_reading(self):
        raise NotImplementedError

    def grip(self, mode, u):
        pass

    def wrist(self, alpha):
        pass

    def friction_ff(self, mask):
        pass

    def tick(self, expert):
        pass


class GraspV3Expert(PickExpert):
    """PickExpert with the guarded grasp; on_event(kind, world, **fields) logs each transition."""
    PARENT_STATE = ('p_cmd', 'R_cmd', 'q_cmd', 'integ', 'timer', 'phase', 'done', 'grip', 'need_frame', 'finger_prev', 'finger_still')

    def __init__(self, task, *args, cfg=None, hooks=None, on_event=None, **kw):
        super().__init__(task, *args, **kw)
        self.gc, self.hooks, self.on_event = cfg or GraspV3Cfg(), hooks or NullHooks(), on_event

    def reset(self):
        super().reset()
        N, dev = len(self.phase), self.t.device
        z = lambda: torch.zeros(N, device=dev)   # noqa: E731
        self.g = torch.zeros(N, dtype=torch.long, device=dev)
        self.gt = torch.zeros(N, dtype=torch.long, device=dev)
        self.attempt = torch.ones(N, dtype=torch.long, device=dev)
        self.gave_up = torch.zeros(N, dtype=torch.bool, device=dev)
        self.det = TouchDetector(N, self.gc.td_drop_mm * 1e-3, self.gc.td_eps_mm * 1e-3, dev, self.gc.td_follow_frac, self.gc.td_step_mm_s * 1e-3, self.dt)
        self.p_hold = torch.zeros(N, 3, device=dev)
        self.p_rise = torch.zeros(N, 3, device=dev)
        self.integ0 = torch.zeros(N, 6, device=dev)
        self.td_mode = torch.zeros(N, dtype=torch.long, device=dev)         # 0 descending, 1 touch, 2 fallback
        self.alpha = z()
        self.alpha_goal = z()
        self.rd_hist = torch.zeros(N, max(2, round(self.gc.stall_window_s / self.dt) + 1), device=dev)
        # per-attempt records (evaluation and logs); first-close values kept in *_first
        self.touch_z, self.rot_close, self.tip_rise = z().fill_(float('nan')), z().fill_(float('nan')), z().fill_(float('nan'))
        self.z_close0 = z().fill_(float('nan'))
        self.first_held = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.t_grasp = z().fill_(float('nan'))
        self.k_act = 0
        self.lost_n = torch.zeros(N, dtype=torch.long, device=dev)
        self.d_max = z()
        self.z_touch = z().fill_(float('nan'))
        self.roll_close_b = z().fill_(float('nan'))

    # ------------------------------------------------------------------ helpers
    def _ev(self, kind, mask, **kw):
        if self.on_event is None:
            return
        for w in torch.where(mask)[0].tolist():
            self.on_event(kind, w, attempt=int(self.attempt[w]), tick=self.k_act,
                          **{k: (float(v[w]) if torch.is_tensor(v) and v.dim() == 1 else (v[w].tolist() if torch.is_tensor(v) else v)) for k, v in kw.items()})

    def _go(self, mask, g, name=None, **kw):
        if mask.any():
            self.g[mask] = g
            self.gt[mask] = 0
            self._ev(name or NAMES[g], mask, **kw)

    def servo_update(self, mask, delta):
        r0 = self.R_g.clone()
        super().servo_update(mask, delta)
        c = self.gc
        if not c.level_roll:
            return
        m = mask & self.ik_ok
        if not m.any():
            return
        R = self.R_g
        y = R[:, :, 1]
        roll = torch.asin(y[:, 2].clamp(-1, 1))                # closing axis from horizontal (the analysis' convention)
        lim = torch.deg2rad(torch.tensor(c.level_roll_max_deg, device=R.device, dtype=R.dtype))
        change = roll.clamp(-lim, lim) - roll
        zv = torch.tensor([0., 0., 1.], device=R.device, dtype=R.dtype).expand(len(R), 3)
        ax = torch.linalg.cross(y, zv)                         # as grasp_fidelity.apply_servo_err: changes asin(R[2,1]) by the angle
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)
        Rl = K.axis_angle_mat(ax, change) @ R
        self.R_g = torch.where(m[:, None, None], Rl, R)
        upv = torch.tensor([0., 0., 1.], device=R.device, dtype=R.dtype)
        # the approach line follows the levelled tool (the frame's position correction stays)
        from blind_insertion.scenes import reader_grasp as G
        self.p_pre = torch.where(m[:, None], self.p_g - G.APPROACH_BACK * self.R_g[:, :, 2] + G.APPROACH_UP * upv, self.p_pre)
        self._ev('roll_fix', m, roll_servo_deg=torch.rad2deg(roll), roll_kept_deg=torch.rad2deg(roll + change),
                 plan_change_deg=torch.rad2deg(K.rot_log(r0 @ self.R_g.transpose(1, 2)).norm(dim=1)))

    def _enter_td(self, mask, q):
        self.td_mode[mask] = 0
        self.d_max[mask] = 0
        self.det.reset(mask)
        self.integ0[mask] = self.integ[mask]
        self._go(mask, G_TD, 'td_start', z_plan=self.p_g[:, 2])

    def _hold_from_touch(self, mask, p_a):
        """Hold: planned xy and orientation; height = measured at the trigger + td_lift_mm (the target is D below the hand),
        or the planned height after a descent without contact. The integral may correct, within its cap."""
        z_ref = torch.where(self.td_mode == 1, p_a[:, 2] + self.gc.td_lift_mm * 1e-3, self.p_g[:, 2])
        ph = torch.cat([self.p_g[:, :2], z_ref[:, None]], 1)
        self.p_hold[mask] = ph[mask]
        self.integ0[mask] = self.integ[mask]
        self.alpha_goal[mask] = 1.

    # ------------------------------------------------------------------ one tick
    def act(self):
        c, t = self.gc, self.t
        dt, N = self.dt, len(self.phase)
        self.k_act += 1
        nf = len(self.servo_fracs)
        if c.touchdown:
            air = c.test_mode == 'air'
            enter = (self.g == G_NONE) & (self.phase == 1) & (self.servo_k >= (min(1, nf) if air else nf)) & (self.refined | (nf == 0)) & self.ik_ok
            if enter.any():
                if air:
                    self.p_g[enter, 2] += c.air_plan_up_mm * 1e-3
                self._enter_td(enter, None)
        pm = self.g == G_NONE
        a = torch.zeros(N, 14, device=t.device)
        a[:, 13] = 1.
        if pm.any():
            snap = {k: getattr(self, k).clone() for k in self.PARENT_STATE} if (~pm).any() else None
            a_p = super().act()
            a = torch.where(pm[:, None], a_p, a)
            if snap is not None:
                for k, v in snap.items():
                    cur = getattr(self, k)
                    m = ~pm
                    setattr(self, k, torch.where(m.view(-1, *([1] * (cur.dim() - 1))), v, cur))
            if not c.touchdown:      # the plan's settle ended: this tick its close starts, here from the chosen hold
                ent = pm & (self.phase >= 3)
                if ent.any():
                    q = self._q()
                    p_a, _ = t.chain.fk(q)
                    if c.hold_from == 'press':
                        self.p_hold[ent] = self.p_cmd[ent]
                    else:
                        self.p_hold[ent] = p_a[ent] + torch.tensor([0., 0., c.hold_offset_mm * 1e-3], device=q.device)
                    self.integ0[ent] = self.integ[ent]
                    self.alpha_goal[ent] = 1.
                    self._ev('close_from_settle', ent, z=p_a[:, 2])
                    self._start_close(ent, p_a)
        gm = self.g != G_NONE
        if gm.any():
            a_g = self._guarded(gm)
            a = torch.where(gm[:, None], a_g, a)
            self.done = (self.g == G_DONE) | (self.g == G_GIVEUP)
        else:
            self.done = torch.zeros_like(self.done)
        self._apply_hooks()
        return a

    def _start_close(self, mask, p_a):
        c = self.gc
        self.z_close0[mask] = p_a[:, 2][mask]
        self.rd_hist[mask] = 0
        self._go(mask, G_CLOSE if c.gentle_close else G_SQUEEZE, 'close_start' if c.gentle_close else 'squeeze_start', z=p_a[:, 2])

    def _guarded(self, gm):
        c, t = self.gc, self.t
        dt = self.dt
        dev, N = t.device, len(self.g)
        up = torch.tensor([0., 0., 1.], device=dev)
        g = self.g
        q = self._q()
        p_a, R_a, J = t.chain.fk(q, with_jac=True)
        e_r_vec = K.rot_log(self.R_g @ R_a.transpose(1, 2))
        e_r = torch.rad2deg(e_r_vec.norm(dim=1))
        n_settle = round(c.td_settle_s / dt)
        descend = (g == G_TD) & (self.gt >= n_settle)
        # commanded Cartesian goal per phase
        p_wp = self.p_cmd.clone()
        v = torch.full((N,), c.td_speed_m_s, device=dev)
        goal_td = self.p_g - up * c.td_max_below_mm * 1e-3
        above = self.p_cmd[:, 2] > self.p_g[:, 2] + 1e-5
        p_wp = torch.where((descend & above)[:, None], self.p_g, p_wp)
        p_wp = torch.where((descend & ~above)[:, None], goal_td, p_wp)
        hold_pt = self.p_hold
        m = g == G_HOLD
        p_wp = torch.where(m[:, None], hold_pt, p_wp)
        v = torch.where(m, torch.full_like(v, .02), v)
        m = (g == G_CLOSE) | (g == G_SQUEEZE)
        p_wp = torch.where(m[:, None], hold_pt - up * c.holddown_mm * 1e-3, p_wp)
        v = torch.where(m, torch.full_like(v, .01), v)
        m = g == G_TEST
        p_wp = torch.where(m[:, None], hold_pt + up * c.test_lift_mm * 1e-3, p_wp)
        v = torch.where(m, torch.full_like(v, c.test_v_m_s), v)
        m = (g == G_LIFT) | (g == G_END)
        p_wp = torch.where(m[:, None], self.p_lift, p_wp)
        v = torch.where(m, torch.full_like(v, self.v), v)
        m = g == G_RISE
        p_wp = torch.where(m[:, None], self.p_rise, p_wp)
        v = torch.where(m, torch.full_like(v, self.v * .5), v)
        pc0, Rc0, qc0 = self.p_cmd.clone(), self.R_cmd.clone(), self.q_cmd.clone()
        dist, ang = self._cart_step(p_wp, self.R_g, v[:, None])
        q_ik, _, _ = self._ik(self.p_cmd, self.R_cmd, self.q_cmd, 12)
        self.q_cmd = q_ik
        # integral: HOLD (bounded around its entry value, target above the table), LIFT/END; frozen otherwise
        hold = g == G_HOLD
        integrate = hold | (g == G_LIFT) | (g == G_END)
        p_tgt = torch.where(hold[:, None], self.p_hold, self.p_cmd)
        e6 = torch.cat([p_tgt - p_a, .5 * e_r_vec], 1)
        JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, device=dev)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e6[..., None]))[..., 0]
        integ = torch.where(integrate[:, None], (self.integ + self.ki * dq).clamp(-self.icap, self.icap), self.integ)
        cap = c.hold_integ_cap_rad
        integ = torch.where(hold[:, None], torch.max(torch.min(integ, self.integ0 + cap), self.integ0 - cap), integ)
        self.integ = integ
        q_phys = self.q_cmd + self.integ
        # commands of the other worlds are restored by the caller's where(); parent worlds keep their state
        pm = ~gm
        self.p_cmd = torch.where(pm[:, None], pc0, self.p_cmd)
        self.R_cmd = torch.where(pm[:, None, None], Rc0, self.R_cmd)
        self.q_cmd = torch.where(pm[:, None], qc0, self.q_cmd)
        self.gt = self.gt + gm.long()
        reading = self.hooks.grip_reading()
        self.rd_hist = torch.cat([self.rd_hist[:, 1:], reading[:, None]], 1)
        self._transitions(gm, descend, p_a, R_a, e_r, dist, ang, reading)
        self.grip = torch.where((self.g >= G_CLOSE) & (self.g <= G_END), -torch.ones_like(self.grip), torch.ones_like(self.grip))
        self.grip = torch.where(self.g == G_DONE, -torch.ones_like(self.grip), self.grip)
        a = torch.zeros(N, 14, device=dev)
        believed = q_phys if self.believed_frame else q_phys - t.enc_error[:, 0]
        a[:, :6] = ((believed - t.joint_target[:, 0]) / t.cfg.action_scale_rad).clamp(-1, 1)
        a[:, 6] = self.grip
        a[:, 13] = 1.
        return a

    def _transitions(self, gm, descend, p_a, R_a, e_r, dist, ang, reading):
        c, dt = self.gc, self.dt
        g = self.g
        roll = torch.rad2deg(torch.asin(R_a[:, 2, 1].clamp(-1, 1)))
        # ---- touch-down
        touched, drop = self.det.update(self.p_cmd[:, 2], p_a[:, 2])
        self.d_max = torch.where(descend, torch.max(self.d_max, drop), self.d_max)
        self.det.reset((g == G_TD) & ~descend)
        at_depth = descend & (dist < 2e-4) & (self.p_cmd[:, 2] < self.p_g[:, 2] - c.td_max_below_mm * 1e-3 + 1e-4)
        hit = descend & (touched | (at_depth & (drop > .5 * self.det.drop_m)))   # stalled, but the depth limit came first
        blocked = torch.zeros_like(hit)
        tip = None
        if hit.any():
            tip = self.t.tip_height(self._q(), 0.)      # logged for the rig; a gate only if td_tip_max_mm is set
            if c.td_tip_max_mm is not None:
                blocked = hit & (tip > c.td_tip_max_mm * 1e-3)
                hit = hit & ~blocked
        deep = at_depth & ~hit & ~blocked
        if hit.any():
            self.td_mode[hit] = 1
            self.z_touch[hit] = p_a[:, 2][hit]
            self._hold_from_touch(hit, p_a)
            self._go(hit, G_HOLD, 'touchdown', z_touch=p_a[:, 2], z_plan=self.p_g[:, 2], d_mm=drop * 1e3, tip_fk_mm=tip * 1e3)
        if deep.any():
            self.td_mode[deep] = 2
            self._hold_from_touch(deep, p_a)
            self._go(deep, G_HOLD, 'touchdown_fallback', z_meas=p_a[:, 2], z_plan=self.p_g[:, 2])
        if blocked.any():
            self._ev('td_blocked', blocked, z_meas=p_a[:, 2], z_plan=self.p_g[:, 2], d_mm=drop * 1e3)
            self._after_empty(blocked, p_a)
        # ---- hold
        m = g == G_HOLD
        e_p = (self.p_hold - p_a).norm(dim=1)
        settled = (e_p < c.hold_settle_tol[0]) & (e_r < c.hold_settle_tol[1]) & (self.gt >= round(c.hold_min_s / dt))
        done = m & (settled | (self.gt >= round(c.hold_max_s / dt)))
        if done.any():
            self._ev('hold_settled', done, err_mm=e_p * 1e3, rot_deg=e_r, roll_deg=roll, ticks=self.gt.float())
            self.rot_close[done] = e_r[done]
            self.roll_close_b[done] = roll[done]
            bad_roll = done & (roll.abs() > c.roll_gate_deg) if c.roll_gate_deg is not None else torch.zeros_like(done)
            if c.test_mode is not None:
                self.p_rise[done] = self.p_cmd[done] + torch.tensor([0., 0., c.rise_mm * 1e-3], device=p_a.device)
                self.alpha_goal[done] = 0.
                self.attempt[done] = c.max_attempts
                self._go(done, G_RISE, 'test_rise', z=p_a[:, 2], rot_deg=e_r, roll_deg=roll, d_max_mm=self.d_max * 1e3, td_mode=self.td_mode.float())
            else:
                if bad_roll.any():
                    self._ev('roll_gate', bad_roll, roll_deg=roll, limit=float(c.roll_gate_deg))
                    self._after_empty(bad_roll, p_a)
                if (done & ~bad_roll).any():
                    self._start_close(done & ~bad_roll, p_a)
        # ---- gentle close
        m = g == G_CLOSE
        win = self.rd_hist.shape[1]
        moved = (self.rd_hist[:, 0] - reading).abs()
        stall = m & (self.gt >= round(c.close_min_s / dt)) & (self.gt >= win) & (moved < c.stall_eps)
        late = m & (self.gt >= round(c.close_max_s / dt))
        through = m & (reading < c.held_band[0]) & (self.gt >= win)     # jaws passed the reader: pads over it
        if through.any():
            self._ev('close_fell_through', through, reading=reading)
            self._close_record(through, p_a, e_r)
            self._after_empty(through, p_a)
        go = (stall | late) & ~through
        if go.any():
            self._ev('close_stall', go, reading=reading, timed_out=late.float(), rot_deg=e_r, roll_deg=roll, rise_mm=(p_a[:, 2] - self.z_close0) * 1e3)
            self._go(go, G_SQUEEZE, 'squeeze_start', reading=reading)
        # ---- squeeze and dwell
        m = g == G_SQUEEZE
        go = m & (self.gt >= round((c.squeeze_s + c.dwell_s) / dt))
        if go.any():
            self._close_record(go, p_a, e_r)
            self._ev('squeeze_end', go, reading=reading, rot_deg=e_r, roll_deg=roll, rise_mm=(p_a[:, 2] - self.z_close0) * 1e3)
            self._go(go, G_TEST, 'test_lift', reading=reading)
        # ---- test lift
        m = g == G_TEST
        go = m & (dist < 2e-4) & (self.gt >= round(c.test_dwell_s / dt) + round(c.test_lift_mm * 1e-3 / c.test_v_m_s / dt))
        if go.any():
            held = (reading >= c.held_band[0]) & (reading <= c.held_band[1])
            self.first_held = torch.where(go & (self.first_held < 0), held.long(), self.first_held)
            self.alpha_goal[go] = 0.
            self._ev('test_result', go, reading=reading, held=held.float())
            ok, no = go & held, go & ~held
            if ok.any():
                self.t_grasp = torch.where(ok & torch.isnan(self.t_grasp), torch.full_like(self.t_grasp, self.k_act * dt), self.t_grasp)
                self._go(ok, G_LIFT, 'lift')
            if no.any():
                self._after_empty(no, p_a)
        # ---- lift, end: keep checking (pick11 stalled at 0.301 and lost the reader 2.26 s after the lift start)
        m = (g == G_LIFT) | (g == G_END)
        self.lost_n = torch.where(m & (reading < c.held_band[0]), self.lost_n + 1, torch.zeros_like(self.lost_n))
        lost = m & (self.lost_n >= c.lost_ticks)
        if lost.any():
            self._ev('lost_in_lift', lost, reading=reading)
            self.lost_n[lost] = 0
            self._after_empty(lost, p_a)
        m = (g == G_LIFT) & ~lost
        go = m & (dist < 5e-4) & (self.gt >= 10) & ((e_r < self.lift_tol) | (self.gt >= round(3. / dt)))
        if go.any():
            self._go(go, G_END, 'end_hold', reading=reading, rot_deg=e_r)
        go = (g == G_END) & (self.gt >= self.hold_n) & ~lost
        if go.any():
            self._go(go, G_DONE, 'done', reading=reading)
        # ---- retry
        go = (g == G_REOPEN) & (self.gt >= round(c.reopen_s / dt))
        if go.any():
            self._go(go, G_RISE, 'rise', reading=reading)
        go = (g == G_RISE) & (dist < 5e-4) & (self.gt >= 10)
        if go.any():
            last = go & (self.attempt >= c.max_attempts)
            if last.any():
                self.gave_up[last] = True
                self._go(last, G_GIVEUP, 'giveup')
            again = go & ~last
            if again.any():
                self.attempt[again] += 1
                self.g[again] = G_NONE
                self.gt[again] = 0
                self.servo_k[again] = 0
                self.refined[again] = False
                self.need_frame[again] = False
                self.hover_iters[again] = 0
                ok_r = self.rehover(again)
                self.hover_iters[again] = 0
                bad = again & ~ok_r
                self.gave_up[bad] = True
                self.g[bad] = G_GIVEUP
                self._ev('retry_rehover', again, ik_ok=ok_r.float())

    def _close_record(self, mask, p_a, e_r):
        self.rot_close[mask] = e_r[mask]
        self.tip_rise[mask] = ((p_a[:, 2] - self.z_close0) * 1e3)[mask]

    def _after_empty(self, mask, p_a):
        c = self.gc
        self.first_held = torch.where(mask & (self.first_held < 0), torch.zeros_like(self.first_held), self.first_held)
        self.alpha_goal[mask] = 0.
        self.p_rise[mask] = self.p_cmd[mask] + torch.tensor([0., 0., c.rise_mm * 1e-3], device=p_a.device)
        self._go(mask, G_REOPEN, 'reopen', attempts_max=float(c.max_attempts))

    def _apply_hooks(self):
        c, dt = self.gc, self.dt
        rate = dt / max(c.wrist_ramp_s, dt)
        if c.wrist_kp is not None:
            d = self.alpha_goal - self.alpha
            self.alpha = self.alpha + d.clamp(-rate, rate)
            self.hooks.wrist(self.alpha)
        mode = torch.zeros_like(self.g)
        u = torch.zeros_like(self.alpha)
        mode[self.g == G_CLOSE] = 1
        sq = self.g == G_SQUEEZE
        if c.gentle_close:
            mode[sq] = 2
            u[sq] = (self.gt[sq].float() * dt / max(c.squeeze_s, dt)).clamp(max=1.)
        self.hooks.grip(mode, u)
        if c.td_friction_ff:
            self.hooks.friction_ff(self.g == G_TD)
        self.hooks.tick(self)
