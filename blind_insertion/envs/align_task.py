"""Final alignment before the hand-over (docs/real/K2_PROGRESS.md, "Alignment"): bring the plug from a real-like stop
(12-24 mm out, several mm off) to the hand-over pose (0, 0, 6) mm in front of port 2, smoothly, from what the wrist
camera sees. Same physics, grasp, DR and joint interface as InsertionTask; different action, observation and reward.

Action (6, in [-1, 1]): plug-tip twist in the ESTIMATED port frame: velocity x align_vmax_mm_s (across, up, out) and
angular velocity x align_wmax_deg_s. It becomes a joint increment by the believed tip Jacobian (damped least squares)
and goes through BimanualTask's accumulated, capped joint targets (so a stuck joint cannot wind the target more than
target_cap_rad ahead).

Actor observation (no absolute encoder FK):
  [0:3]   last port-vision plug position (mm / 10), [3:6] its plug-vs-panel rotation vector (deg / 10)
  [6]     frame valid (1) or dropped (0), [7] age of the last valid frame (s)
  [8:11]  tip motion since that frame was captured, believed FK, estimated port axes (mm / 10); [11:14] rotation (deg / 10)
  [14:20] right encoder angles - their value at the reset (rad), [20:26] right encoder velocity (noisy, rad/s)
  [26:32] right joint target - believed angles (rad x 10), [32:38] previous action
Vision model (port_vision, docs/real/PORT_VISION.md): a frame every align_vis_every steps (10 Hz), the truth from
align_vis_latency steps earlier, x slope, + run bias (per episode) + a smooth position-dependent error field + frame
noise; dropped with align_vis_drop.

Episode ends: success = the vision-consistent error (truth + run bias, no field or frame noise) within tolerance and the
tip slower than align_still_mm_s for align_hold_s; failures: the plug within align_touch_mm of the panel or touching
(pushed), off by more than align_far_mm (far), reader moved in the hand (slip), non-finite (bad).

Scripted baselines on the same env: `step_policy` (today's stage: 1 mm steps, settle, 5 frames, repeat) and
`servo_policy` (continuous vision servo on vision + motion since the frame).
"""
from __future__ import annotations

import math

import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.insertion_task import InsertionTask


def _rotvec_deg(R):
    return K.rot_log(R) * 180 / math.pi


class AlignTask(InsertionTask):
    ACT = 6

    def __init__(self, cfg=None, seed=0):
        self._ready = False
        super().__init__(cfg, seed)
        n, dev = self.N, self.device
        z = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        L = int(self.icfg.align_vis_latency) + 1
        self.truth_buf = torch.zeros(L, n, 6, device=dev)        # plug pose in port (mm, deg) history
        self.tip_buf = torch.zeros(L, n, 6, device=dev)          # believed tip pose (mm, deg) history
        self.meas, self.meas_tip, self.meas_age, self.meas_valid = z(6), z(6), z(), z()
        self.run_bias, self.slope = z(6), torch.ones(n, 3, device=dev)
        self.field_k, self.field_ph, self.field_a = z(3, 3, 3), z(3, 3), z(3, 3)
        self.vis_phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.q0 = z(6)
        self.still_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_tip = z(3)
        self.max_speed, self.path_mm, self.min_gap, self.start_err = z(), z(), z(), z()
        self.port_R_err = torch.eye(3, device=dev).expand(n, 3, 3).clone()
        self.align_metrics = {}
        self.hist = torch.zeros(n, self.icfg.history, self.obs_dim, device=dev)
        self._ready = True
        self.reset_all()

    def reset_all(self):
        if getattr(self, '_ready', False):      # InsertionTask.__init__ resets before these buffers exist
            super().reset_all()

    # ------------------------------------------------------------------ dims
    @property
    def obs_dim(self):
        return 38

    @property
    def act_dim(self):
        return 6

    @property
    def critic_dim(self):
        return 6 + 6 + 3 + 6 + 6 + 1

    # ------------------------------------------------------------------ helpers
    def _truth(self):
        p, R = self.runtime.plug_in_port()
        return torch.cat([p * 1e3, _rotvec_deg(R)], 1)

    def _est_port_R(self):
        _, R = self._port_pose(self.pc_start)
        return R @ self.port_R_err

    def _tip_believed(self):
        qm = self._measured_model_angles()[:, 0]
        p, R, J = self.right_chain.fk(qm, with_jac=True)
        s = self.ref_rel[:, :3, 3]
        r = (R @ s[..., None])[..., 0]
        return p + r, R, J, r, qm

    def _tip_in_port(self):
        tip, R, _, _, _ = self._tip_believed()
        Rp = self._est_port_R()
        pp = (Rp.transpose(1, 2) @ tip[..., None])[..., 0] * 1e3
        return torch.cat([pp, _rotvec_deg(Rp.transpose(1, 2) @ R)], 1)

    # ------------------------------------------------------------------ reset
    def reset(self, ids=None, starts=None):
        super().reset(ids, starts)
        if ids is None:
            ids = torch.arange(self.N, device=self.device)
        if len(ids) == 0:
            return
        c, dev, g, k = self.icfg, self.device, self.gen, len(ids)
        rn = lambda *s: torch.randn(*s, device=dev, generator=g)  # noqa: E731
        ru = lambda *s: torch.rand(*s, device=dev, generator=g)  # noqa: E731
        self.run_bias[ids] = rn(k, 6) * torch.tensor(list(c.align_vis_run_bias_mm) + [c.align_vis_ang_run_bias_deg] * 3, device=dev)
        self.slope[ids] = 1 + (ru(k, 3) * 2 - 1) * c.align_vis_slope_err
        lam = c.align_vis_field_wavelength_mm[0] + (c.align_vis_field_wavelength_mm[1] - c.align_vis_field_wavelength_mm[0]) * ru(k, 3, 3)
        d = rn(k, 3, 3, 3)
        d = d / d.norm(dim=-1, keepdim=True)
        self.field_k[ids] = d * (2 * math.pi / lam)[..., None]
        self.field_ph[ids] = ru(k, 3, 3) * 2 * math.pi
        self.field_a[ids] = torch.tensor(c.align_vis_field_mm, device=dev)[None, :, None].expand(k, 3, 3) * math.sqrt(2 / 3)
        ang = rn(k, 3) * c.align_port_R_err_deg * math.pi / 180
        self.port_R_err[ids] = K.rpy_mat(ang)
        self.vis_phase[ids] = torch.randint(0, int(c.align_vis_every), (k,), device=dev, generator=g)
        tr, tp = self._truth()[ids], self._tip_in_port()[ids]
        self.truth_buf[:, ids], self.tip_buf[:, ids] = tr[None], tp[None]
        self.meas[ids] = self._vision(tr, ids)
        self.meas_tip[ids], self.meas_age[ids], self.meas_valid[ids] = tp, 0, 1
        self.q0[ids] = self._measured_model_angles()[ids, 0]
        self.still_steps[ids] = 0
        self.prev_tip[ids] = tp[:, :3]
        self.max_speed[ids], self.path_mm[ids], self.min_gap[ids] = 0, 0, tr[:, 2]
        self.start_err[ids] = (tr[:, :3] - self.goal()[:3]).norm(dim=1)
        self.previous_action[ids] = 0
        o = self._actor_obs()
        self.hist[ids] = o[ids, None].expand(-1, self.icfg.history, -1)
        self.phi[ids] = self._potential()[ids]

    def _vision(self, truth, ids):
        c = self.icfg
        p = truth[:, :3]
        pos = self.slope[ids] * p + self.run_bias[ids, :3] + self._field_ids(p, ids) \
            + torch.randn(len(ids), 3, device=self.device, generator=self.gen) * torch.tensor(c.align_vis_frame_mm, device=self.device)
        ang = truth[:, 3:] + self.run_bias[ids, 3:] + torch.randn(len(ids), 3, device=self.device, generator=self.gen) * c.align_vis_ang_frame_deg
        return torch.cat([pos, ang], 1)

    def _field_ids(self, p, ids):
        out = torch.zeros_like(p)
        fk, fph, fa = self.field_k[ids], self.field_ph[ids], self.field_a[ids]
        for a in range(3):
            ph = (fk[:, a] @ p[..., None])[..., 0] + fph[:, a]
            out[:, a] = (fa[:, a] * torch.sin(ph)).sum(1)
        return out

    def consistent_error(self):
        """Vision-consistent plug pose (truth + run bias, x slope): what the stage can align to. (n, 6) mm, deg."""
        t = self._truth()
        return torch.cat([self.slope * t[:, :3] + self.run_bias[:, :3], t[:, 3:] + self.run_bias[:, 3:]], 1)

    def goal(self):
        c = self.icfg
        return torch.tensor(list(c.align_goal_mm) + [0., 0., 0.], device=self.device)

    # ------------------------------------------------------------------ obs
    def est_now(self):
        """Fused estimate: last frame + believed motion since its capture (n, 6)."""
        tp = self._tip_in_port()
        return self.meas + (tp - self.meas_tip)

    def _actor_obs(self):
        c = self.icfg
        tp = self._tip_in_port()
        qm = self._measured_model_angles()[:, 0]
        o = self.observation()
        v = o['encoder_velocity_rad_s'].reshape(-1, 2, 6)[:, 0]
        if c.vel_obs_noise is not None:
            v = v + torch.randn(v.shape, device=self.device, generator=self.gen) * torch.tensor(c.vel_obs_noise, device=self.device) * self.vel_noise_scale
        lead = self.joint_target[:, 0] - qm
        return torch.nan_to_num(torch.cat([self.meas / 10, self.meas_valid[:, None], self.meas_age[:, None],
                                           (tp - self.meas_tip) / 10, qm - self.q0, v, lead * 10, self.previous_action[:, :6]], 1), 0., 0., 0.)

    def critic_obs(self):
        e = self.consistent_error() - self.goal()
        tp = self._tip_in_port()
        vel = (tp[:, :3] - self.prev_tip) / self.dt
        return torch.nan_to_num(torch.cat([e / 10, (self._truth() - self.goal()) / 10, vel / 10, self.run_bias / 10,
                                           (self.joint_target[:, 0] - self._measured_model_angles()[:, 0]) * 10,
                                           (self.steps.float() / self.max_steps)[:, None]], 1), 0., 0., 0.)

    def current_labels(self):
        return torch.zeros(self.N, self.n_labels, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ step
    def twist_to_action(self, tw):
        """(n, 6) normalized twist in the estimated port frame -> (n, 6) normalized joint action."""
        c = self.icfg
        tw = tw.clamp(-1, 1)
        Rp = self._est_port_R()
        v = (Rp @ (tw[:, :3] * c.align_vmax_mm_s * 1e-3)[..., None])[..., 0]
        w = (Rp @ (tw[:, 3:] * c.align_wmax_deg_s * math.pi / 180)[..., None])[..., 0]
        _, _, J, r, _ = self._tip_believed()
        v_tcp = v - torch.cross(w, r, dim=1)
        x = torch.cat([v_tcp, w], 1) * self.dt
        JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, x[..., None]))[..., 0]
        return (dq / self.cfg.action_scale_rad).clamp(-1, 1)

    def step(self, action):
        tw = torch.as_tensor(action, device=self.device, dtype=torch.float32).clamp(-1, 1)
        a = torch.zeros(self.N, 14, device=self.device)
        a[:, :6] = self.twist_to_action(tw)
        a[:, 6] = -1.
        a[:, 13] = -1.
        self.done[:] = False
        from blind_insertion.envs.bimanual_task import BimanualTask
        _, _, terminated, truncated, info = BimanualTask.step(self, a)
        self.previous_action[:, :6] = tw
        return self._post_step(terminated, truncated, info, torch.zeros(self.N, dtype=torch.bool, device=self.device))

    def _transition(self):
        rt, c = self.runtime, self.icfg
        self.steps += 1
        # vision frames
        tr, tp = self._truth(), self._tip_in_port()
        self.truth_buf = torch.cat([self.truth_buf[1:], tr[None]], 0)
        self.tip_buf = torch.cat([self.tip_buf[1:], tp[None]], 0)
        self.meas_age += self.dt
        new = ((self.steps + self.vis_phase) % int(c.align_vis_every)) == 0
        ids = new.nonzero()[:, 0]
        if len(ids):
            m = self._vision(self.truth_buf[0, ids], ids)
            ok = torch.rand(len(ids), device=self.device, generator=self.gen) >= c.align_vis_drop
            good = ids[ok]
            self.meas[good] = m[ok]
            self.meas_tip[good] = self.tip_buf[0, good]
            self.meas_age[good] = c.align_vis_latency * self.dt
            self.meas_valid[ids] = ok.float()
        # status
        e = self.consistent_error() - self.goal()
        lat, gap_e, ang = e[:, :2].norm(dim=1), e[:, 2].abs(), e[:, 3:].norm(dim=1)
        speed = (tp[:, :3] - self.prev_tip).norm(dim=1) / self.dt
        self.prev_tip = tp[:, :3].clone()
        self.max_speed = torch.maximum(self.max_speed, speed)
        self.path_mm += speed * self.dt
        self.min_gap = torch.minimum(self.min_gap, tr[:, 2])
        within = (lat <= c.align_tol_mm[0]) & (gap_e <= c.align_tol_mm[1]) & (ang <= c.align_tol_deg)
        still = speed <= c.align_still_mm_s
        self.still_steps = torch.where(within & still, self.still_steps + 1, torch.zeros_like(self.still_steps))
        success = self.still_steps >= int(round(c.align_hold_s / self.dt))
        rel = self._reader_in_tcp()
        d_rel = (rel[:, :3, 3] - self.ref_rel[:, :3, 3]).norm(dim=1)
        slip = d_rel > c.slip_m
        gap_true = tr[:, 2]
        pushed = gap_true < c.align_touch_mm
        far = (tr[:, :2].norm(dim=1) > c.align_far_mm) | (gap_true > 45.)
        bad = ~torch.isfinite(rt.v).all(1) | ~torch.isfinite(tr).all(1)
        failure = (bad | slip | pushed | far) & ~success
        terminated = success | failure
        truncated = (self.steps >= self.max_steps) & ~terminated
        self.done = terminated | truncated
        self._last = dict(success=success, slip=slip, pushed=pushed, far=far, bad=bad, lat=lat, gap_e=gap_e, ang=ang, speed=speed,
                          true_lat=(tr[:, :2] - self.goal()[:2]).norm(dim=1), true_gap=tr[:, 2], steps=self.steps.clone().float(),
                          max_speed=self.max_speed.clone(), path_mm=self.path_mm.clone(), overshoot_mm=(self.goal()[2] - self.min_gap).clamp(min=0),
                          start_err=self.start_err.clone())
        info = {'success': success.clone(), 'nonfinite': bad, 'slip': slip, 'pushed': pushed, 'far': far, 'steps': self.steps.clone()}
        return success.float(), terminated, truncated, info

    def _potential(self):
        e = self.consistent_error() - self.goal()
        d = e[:, :2].norm(dim=1) * 2 + e[:, 2].abs() + e[:, 3:].norm(dim=1) * 0.5
        return -(d / self.icfg.align_shaping_mm).clamp(max=10.)

    def _post_step(self, terminated, truncated, info, warm):
        c, L = self.icfg, self._last
        phi2 = self._potential()
        failed = L['slip'] | L['pushed'] | L['far']
        da = (self.previous_action[:, :6] - self._prev_tw).pow(2).sum(1) if hasattr(self, '_prev_tw') else torch.zeros(self.N, device=self.device)
        reward = c.success_bonus * L['success'].float() + c.shaping * (phi2 - self.phi) - c.fail_penalty * (failed & ~L['success']).float() \
            - c.align_jerk_coef * da - c.align_speed_coef * (L['speed'] - c.align_speed_cap_mm_s).clamp(min=0) - c.align_time_coef
        reward = torch.where(L['bad'], torch.zeros_like(reward), torch.nan_to_num(reward, 0., 0., 0.))
        self._prev_tw = self.previous_action[:, :6].clone()
        self.phi = phi2
        o = self._actor_obs()
        self.hist = torch.cat([self.hist[:, 1:], o[:, None]], 1)
        crit = self.critic_obs()
        done = terminated | truncated
        info.update(timeout=truncated, valid=torch.ones_like(done), labels=self.current_labels())
        ids = done.nonzero()[:, 0]
        if len(ids):
            info['final'] = {k: v[ids].clone() for k, v in L.items() if torch.is_tensor(v)}
            info['final_ids'] = ids
            self.reset(ids)
            self._prev_tw[ids] = 0
            crit[ids] = self.critic_obs()[ids]
        return self.hist, crit, reward, done, info


# ---------------------------------------------------------------------- scripted baselines
class ServoPolicy:
    """Continuous vision servo: twist = -gain x (fused estimate - goal), speed-limited, smoothed."""

    def __init__(self, env, gain=1.5, ang_gain=1.0, vmax_mm_s=8., wmax_deg_s=4., smooth=0.2):
        self.env, self.g, self.ga, self.vmax, self.wmax, self.s = env, gain, ang_gain, vmax_mm_s, wmax_deg_s, smooth
        self.prev = torch.zeros(env.N, 6, device=env.device)

    def __call__(self, hist=None):
        e = self.env
        c = e.icfg
        err = e.est_now() - e.goal()
        v = (-self.g * err[:, :3]).clamp(-self.vmax, self.vmax) / c.align_vmax_mm_s
        w = (-self.ga * err[:, 3:]).clamp(-self.wmax, self.wmax) / c.align_wmax_deg_s
        tw = torch.cat([v, w], 1)
        self.prev = self.prev + self.s * (tw - self.prev)
        return self.prev.clone()

    def reset(self, ids):
        self.prev[ids] = 0


class StepPolicy:
    """Today's align stage, simplified: settle until still, collect `frames` valid frames, take one step of at most
    step_mm toward the goal (median of the frames) at step_speed_mm_s, repeat."""

    def __init__(self, env, step_mm=1.0, step_deg=0.75, step_speed_mm_s=5., settle_s=0.3, frames=5, still_mm_s=1.0):
        self.env, self.step_mm, self.step_deg, self.v, self.settle, self.frames, self.still = env, step_mm, step_deg, step_speed_mm_s, settle_s, frames, still_mm_s
        n, dev = env.N, env.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)       # 0 settle, 1 measure, 2 move
        self.cnt = torch.zeros(n, dtype=torch.long, device=dev)
        self.acc = torch.zeros(n, 6, device=dev)
        self.nacc = torch.zeros(n, device=dev)
        self.move = torch.zeros(n, 6, device=dev)
        self.left = torch.zeros(n, device=dev)
        self.last_age = torch.zeros(n, device=dev)

    def reset(self, ids):
        self.phase[ids] = 0; self.cnt[ids] = 0; self.acc[ids] = 0; self.nacc[ids] = 0; self.left[ids] = 0

    def __call__(self, hist=None):
        e, c, dt = self.env, self.env.icfg, self.env.dt
        out = torch.zeros(e.N, 6, device=e.device)
        speed = e._last['speed'] if hasattr(e, '_last') else torch.zeros(e.N, device=e.device)
        self.cnt += 1
        # settle -> measure
        s = (self.phase == 0) & (self.cnt * dt >= self.settle) & (speed <= self.still)
        self.phase[s], self.acc[s], self.nacc[s] = 1, 0, 0
        # measure: accumulate fresh valid frames (meas_age resets on a new valid frame)
        m = self.phase == 1
        fresh = m & (e.meas_valid > 0) & (e.meas_age <= c.align_vis_latency * dt + 1e-6)
        self.acc[fresh] += e.meas[fresh]; self.nacc[fresh] += 1
        go = m & (self.nacc >= self.frames)
        if go.any():
            err = self.acc[go] / self.nacc[go, None] - e.goal()
            d = -err
            dp = d[:, :3]
            dp = dp * (self.step_mm / dp.abs().max(1, keepdim=True).values.clamp(min=self.step_mm))
            da = d[:, 3:]
            da = da * (self.step_deg / da.abs().max(1, keepdim=True).values.clamp(min=self.step_deg))
            dist = dp.norm(dim=1)
            T = (dist / self.v).clamp(min=dt)
            self.move[go] = torch.cat([dp / T[:, None] / c.align_vmax_mm_s, da / T[:, None] / c.align_wmax_deg_s], 1)
            self.left[go] = T
            self.phase[go], self.cnt[go] = 2, 0
        mv = self.phase == 2
        out[mv] = self.move[mv]
        self.left[mv] -= dt
        done = mv & (self.left <= 0)
        self.phase[done], self.cnt[done] = 0, 0
        return out
