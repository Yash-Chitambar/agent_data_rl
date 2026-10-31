"""Part 1 (pick) on the GPU bimanual runtime: task layer with grasp DR, and a privileged scripted
expert for the top-down side grasp (`scenes.reader_grasp`).

PickTask = BimanualTask (arm dynamics, PC/reader layout, latency, encoder-offset DR) plus contact
DR resampled per world on reset: pad friction, reader-table friction, gripper force. The left arm
holds its reset pose. `pick_metrics()` measures lift, reader tilt and the in-hand pose.

PickExpert uses privileged state (true joint angles and the reader pose): joint-space move to the
pre-grasp (IK solved at reset), Cartesian descent, settle-before-close (joint integral until the
true TCP is within tolerance), close, lift, hold. Its actions go through the task's normalized
accumulate interface (`BimanualTask.step`), so recorded actions are valid policy actions.

    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=1024, device='cuda:0')))
    ex = PickExpert(task); task.reset(); ex.reset()
    while not ex.done.all(): task.step(ex.act())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import warp as wp
import mujoco_warp as mjw

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.envs.bimanual_task import BimanualTask, BimanualTaskCfg
from blind_insertion.scenes import reader_grasp as G


@dataclass
class PickTaskCfg(BimanualTaskCfg):
    physics: BimanualCfg = field(default_factory=BimanualCfg)
    episode_s: float = 12.
    # Contact DR (per world, on reset). Pads have contact priority, so their friction sets the
    # pad-reader pair; the table is set to the lower bound so the reader value sets reader-table.
    # Pad friction and squeeze are unmeasured. Fixed values chosen for the lowest in-hand creep with a
    # reliable lift (sweep 2026-10-31, n = 1024 each, 10 s hold): higher pad mu creeps more (numerical;
    # mu 2.0: 70 % within 1 deg over 10 s), mu 1.0-1.2 hold. (lo, hi) per world; None keeps 1.68.
    pad_friction_range: tuple | None = (1.2, 1.2)
    reader_table_friction_range: tuple | None = (0.4, 1.0)   # <= 0.3 gave non-finite PC states (MJWarp solver) in ~2-8 % of worlds
    grip_force_scale_range: tuple | None = None        # x the 12 N squeeze; 0.8-1.2 made no difference
    # Contact impedance ratio for this task. The bimanual scene uses 100 (fixes a numerical grasp
    # creep of the top-down grasp); with the side grasp's ~3 mm contact band, 100 makes the pad
    # friction fail outright below mu ~1.3 (CPU and GPU: reader stays on the table) while 1-10 hold
    # (FINDINGS 2026-10-31). None keeps the scene value.
    # Left arm parked during the pick (joint angles, model frame): the recorded reference pose holds
    # the left hand at (0.03, 0.44, 0.17) m, over the reader region in front of the PC. Joint 1 = 0
    # points the folded arm at the wall from its base (y 0.62), clear of the region. None: reference.
    left_park_q: tuple | None = (0.0, 0.83, 0.29, -0.02, -0.73, -0.78)
    # Right gripper opened only this far for the approach (finger joint, m; full open 0.0475, pad gap
    # 92 mm). 0.024: 45 mm pad gap, 5 mm per side around the 34.6 mm body, so the open fingers clear
    # the arm mounting plate and the PC front in the strip in front of the PC. The real gripper is
    # position controlled; in sim the finger joint limit is lowered. None: full opening.
    finger_open: float | None = 0.024
    # PC pose box for the pick task (x m, y m, yaw deg), replacing BimanualCfg.pc_pose_range: the
    # reader strip in front of the PC needs the PC x within +/-1 cm (user: acceptable for v1).
    pc_pose_range: tuple | None = (0.01, 0.015, 5.)
    impratio: float | None = 30.   # 30: least creep (10 s hold rotation p50 0.11 deg vs 0.22 at 10, 0.66 at 3)


class PickTask(BimanualTask):
    def __init__(self, cfg=None):
        cfg = cfg or PickTaskCfg()
        if cfg.pc_pose_range is not None:
            cfg.physics.pc_pose_range = tuple(cfg.pc_pose_range)
        super().__init__(cfg)
        rt, m = self.runtime, self.runtime.mjm
        dev = self.device
        self.qadr = rt.indices['right'][0]
        self.lay = rt.lay
        self.chain = K.Chain(m, rt.lay.tcp_site, list(self.qadr), device=dev)
        self.pads = torch.tensor([g for g in range(m.ngeom) if m.geom(g).name.startswith(('left_pad_s', 'right_pad_s'))], device=dev)
        self.reader_geoms = torch.tensor(list(rt.lay.peg_slots), device=dev)
        self.table = m.geom('table').id
        self.table_top = float(m.geom('table').pos[2] + m.geom('table').size[2])
        if (cfg.pad_friction_range or cfg.reader_table_friction_range) and not hasattr(rt, 'geom_friction'):
            raise ValueError('contact DR needs batched geom_friction (BimanualCfg.pc_friction_range)')
        if cfg.reader_table_friction_range:
            rt.geom_friction[:, self.table, 0] = cfg.reader_table_friction_range[0]
        if cfg.grip_force_scale_range:
            rt.grip_scale = torch.ones(cfg.physics.nworld, device=dev)
        if cfg.finger_open is not None:
            jr = wp.to_torch(rt.m.jnt_range)
            for f in ('left', 'right'):
                jid = m.joint(f + '_finger').id
                (jr[:, jid, 1] if jr.dim() == 3 else jr[jid, 1:2]).fill_(cfg.finger_open)
            rt.initial_q[[int(m.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]] = cfg.finger_open
        if cfg.impratio is not None:
            wp.to_torch(rt.m.opt.impratio_invsqrt).fill_(1. / math.sqrt(cfg.impratio))
        self.reader0 = torch.zeros(cfg.physics.nworld, 7, device=dev)
        self.finger_qadr = [int(m.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
        import mujoco
        from blind_insertion.scenes import physical_rig as Rig
        d = mujoco.MjData(m); Rig.reference_pose(m, d, rt.lay); mujoco.mj_forward(m, d)
        self.hand = G.hand_geometry(m, d)
        # offsets[(tilt, flip)]: grasp_site in the reader frame for the normal and the flipped grasp
        self.offsets, self.pad_contact_height = {}, {}
        for tilt in (G.TILT_DEG, *G.TILT_FALLBACK_DEG):
            for flip in (False, True):
                self.offsets[(tilt, flip)], self.pad_contact_height[(tilt, flip)] = G.grasp_offset(rt.lay, self.hand, tilt, flip=flip)

    def grasp_flip(self):
        """Per world: True when the plug points back towards the right base (|plug bearing - base
        bearing| > 90 deg), so the hand reaches from the plug side (reader_grasp.site_rotation flip)."""
        p_r, R_r = self.reader_pose()
        plug = -R_r[:, :2, 2]                                  # reader z points from the plug to the body end
        rel = torch.atan2(plug[:, 1], plug[:, 0]) - torch.atan2(p_r[:, 1], p_r[:, 0])
        return torch.cos(rel) < 0

    def reset(self, ids=None):
        if ids is None: ids = torch.arange(self.cfg.physics.nworld, device=self.device)
        obs = super().reset(ids)
        c, rt, n = self.cfg, self.runtime, len(ids)
        u = lambda r, *s: torch.rand(*s, device=self.device) * (r[1] - r[0]) + r[0]   # noqa: E731
        if c.left_park_q is not None:
            lq = rt.indices['left'][0]; la = rt.indices['left'][2]
            park = torch.tensor(c.left_park_q, device=self.device, dtype=rt.q.dtype)
            rt.q[ids[:, None], torch.tensor(lq, device=self.device)] = park
            rt.ctrl[ids[:, None], torch.tensor(la, device=self.device)] = park
            with wp.ScopedDevice(c.physics.device): mjw.forward(rt.m, rt.d)
            self.joint_target[ids] = self._measured_model_angles()[ids]
            obs = self.observation()
        if c.pad_friction_range:
            rt.geom_friction[ids[:, None], self.pads, 0] = u(c.pad_friction_range, n, 1).expand(-1, len(self.pads))
        if c.reader_table_friction_range:
            rt.geom_friction[ids[:, None], self.reader_geoms, 0] = u(c.reader_table_friction_range, n, 1).expand(-1, len(self.reader_geoms))
        if c.grip_force_scale_range:
            rt.grip_scale[ids] = u(c.grip_force_scale_range, n)
        self.reader0[ids] = self.reader_pose_q()[ids]
        if not hasattr(self, 'failed'): self.failed = torch.zeros(self.cfg.physics.nworld, dtype=torch.bool, device=self.device)
        self.failed[ids] = False
        if not hasattr(self, 'nonfinite'): self.nonfinite = torch.zeros_like(self.failed)
        self.nonfinite[ids] = False
        if not hasattr(self, 'nonfinite_part'):
            self.nonfinite_part = torch.zeros(self.cfg.physics.nworld, dtype=torch.long, device=self.device); self.nonfinite_step = torch.zeros_like(self.nonfinite_part)
        self.nonfinite_part[ids] = 0; self.nonfinite_step[ids] = 0
        return obs

    def step(self, action):
        """BimanualTask.step without its insertion-episode termination: the pick episode length and
        outcome are managed by the caller (see pick_metrics); failures are recorded in `self.failed`."""
        self.done[:] = False
        out = super().step(torch.nan_to_num(torch.as_tensor(action, device=self.device), 0., 0., 0.))
        info = out[4]
        rt = self.runtime
        bad = ~torch.isfinite(rt.q).all(1) | ~torch.isfinite(rt.v).all(1)
        if bad.any():   # which part went first (diagnostic): 1 right arm, 2 left arm, 4 PC, 8 reader
            q = rt.q; part = torch.zeros(len(q), dtype=torch.long, device=self.device)
            for bit, sl in ((1, self.qadr), (2, rt.indices['left'][0]), (4, list(range(rt.pc_qadr, rt.pc_qadr + 7))),
                            (8, list(range(self.lay.peg_qadr, self.lay.peg_qadr + 7)))):
                part |= bit * (~torch.isfinite(q[:, sl]).all(1)).long()
            new = bad & ~self.nonfinite
            self.nonfinite_part[new] = part[new]; self.nonfinite_step[new] = self.steps[new]
        self.nonfinite |= bad
        self.failed |= info['nonfinite'] | info['dropped'] | bad
        if bad.any():   # quarantine: park the world at its reset state so NaNs do not propagate
            rt.q[bad] = rt.initial_q; rt.v[bad] = 0
            self.joint_target[bad] = self._measured_model_angles()[bad]
        self.done[:] = False
        return out

    # --- privileged state -----------------------------------------------------------------------
    def reader_pose_q(self):
        pa = self.lay.peg_qadr
        return self.runtime.q[:, pa:pa + 7].clone()

    def reader_pose(self):
        q = self.reader_pose_q()
        return q[:, :3], K.quat_to_mat(q[:, 3:])

    def q_true(self):
        return self.runtime.q[:, self.qadr]

    def tcp_pose(self):
        return self.chain.fk(self.q_true())

    def inhand(self):
        """Reader pose in the grasp_site frame: p (N,3), R (N,3,3)."""
        p_r, R_r = self.reader_pose()
        p_t, R_t = self.tcp_pose()
        Rt = R_t.transpose(1, 2)
        return (Rt @ (p_r - p_t)[..., None])[..., 0], Rt @ R_r

    def pick_metrics(self):
        p_r, R_r = self.reader_pose()
        lift = p_r[:, 2] - self.reader0[:, 2]
        tilt = torch.rad2deg(torch.acos(R_r[:, 2, 1].clamp(-1, 1)))     # reader thickness axis vs world up
        p_i, R_i = self.inhand()
        pc = self.runtime.q[:, self.runtime.pc_qadr:self.runtime.pc_qadr + 3]
        return {'lift_m': lift, 'tilt_deg': tilt, 'inhand_p': p_i, 'inhand_R': R_i, 'pc_pos': pc,
                'finite': torch.isfinite(self.runtime.q).all(1)}


class PickExpert:
    """Privileged scripted pick with the tipped side grasp (see module docstring)."""
    PHASES = ('to_pregrasp', 'descend', 'settle', 'close', 'lift', 'hold', 'done')

    def __init__(self, task: PickTask, v_max=0.10, w_max=1.0, joint_speed=1.2, settle_tol=(6e-4, 0.4),
                 settle_max_s=2.0, close_s=0.6, hold_s=1.0, integral_gain=0.03, integral_cap=0.08,
                 jitter=None, margin=0.05, lift_tol_deg=1.0, seed=0):
        self.t, self.v, self.w, self.js = task, v_max, w_max, joint_speed
        self.settle_tol, self.dt = settle_tol, task.dt
        self.settle_max = round(settle_max_s / task.dt)
        self.close_n, self.hold_n = round(close_s / task.dt), round(hold_s / task.dt)
        self.ki, self.icap = integral_gain, integral_cap
        # Optional per-episode grasp-point jitter (reader frame, m) to widen demo coverage.
        self.jitter, self.margin, self.lift_tol = jitter, margin, lift_tol_deg
        self.gen = torch.Generator(device=task.device); self.gen.manual_seed(seed + 101)

    def _ik(self, p, R, q0, iters):
        return self.t.chain.ik(p, R, q0, iters=iters)

    def reset(self):
        t, dev = self.t, self.t.device
        N = t.cfg.physics.nworld
        self.phase = torch.zeros(N, dtype=torch.long, device=dev)
        self.timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)
        self.integ = torch.zeros(N, 6, device=dev)
        self.q0 = t.q_true().clone()
        p_r, R_r = t.reader_pose()
        # Per world: the first tilt (60 deg, then the fallbacks) whose pre-grasp, grasp and lift all
        # solve with >= margin rad inside the joint limits.
        self.tilt = torch.full((N,), float('nan'), device=dev)
        self.flip = t.grasp_flip()
        self.offset = torch.zeros(N, 3, device=dev)
        self.p_g = torch.zeros(N, 3, device=dev); self.R_g = torch.zeros(N, 3, 3, device=dev)
        self.p_pre = torch.zeros(N, 3, device=dev); self.p_lift = torch.zeros(N, 3, device=dev)
        best_q = self.q0.clone()
        aim = torch.atan2(p_r[:, 1], p_r[:, 0])
        lo, hi = t.chain.lo + self.margin, t.chain.hi - self.margin
        for tilt in (G.TILT_DEG, *G.TILT_FALLBACK_DEG):
            todo = torch.isnan(self.tilt)
            if not todo.any(): break
            off = torch.where(self.flip[:, None], torch.as_tensor(t.offsets[(tilt, True)], device=dev, dtype=torch.float32),
                              torch.as_tensor(t.offsets[(tilt, False)], device=dev, dtype=torch.float32)).clone()
            if self.jitter is not None:
                off = off + (torch.rand(N, 3, device=dev, generator=self.gen) * 2 - 1) * torch.as_tensor(self.jitter, device=dev, dtype=torch.float32)
            W = G.grasp_poses(p_r, R_r, off, tilt, flip=self.flip)
            q_pre, e_pre = None, None
            for sd in ([1.0, 1.0, -.5, 0, 0], [1.6, 1.3, -.9, 0, 0], [2.2, 1.8, -1.2, 0, 0], [.6, .5, -.3, 0, 0]):
                seed = torch.cat([aim[:, None], torch.tensor(sd, device=dev).expand(N, 5)], 1)
                seed = torch.max(torch.min(seed, t.chain.hi - .05), t.chain.lo + .05)
                q, pe, re = self._ik(*W['pregrasp'], seed, 80)
                e = pe + .05 * re
                if q_pre is None: q_pre, e_pre = q, e
                else:
                    better = e < e_pre
                    q_pre = torch.where(better[:, None], q, q_pre); e_pre = torch.where(better, e, e_pre)
            q_g, pe_g, re_g = self._ik(*W['grasp'], q_pre, 80)
            q_l, pe_l, re_l = self._ik(*W['lift'], q_g, 80)
            inside = lambda q: ((q >= lo) & (q <= hi)).all(1)   # noqa: E731
            ok = todo & (e_pre < 1e-3) & (pe_g + .05 * re_g < 1e-3) & (pe_l + .05 * re_l < 1e-3) & inside(q_pre) & inside(q_g) & inside(q_l)
            self.tilt[ok] = tilt; self.offset[ok] = off[ok]
            self.p_pre[ok], self.p_g[ok], self.p_lift[ok], self.R_g[ok] = W['pregrasp'][0][ok], W['grasp'][0][ok], W['lift'][0][ok], W['grasp'][1][ok]
            best_q[ok] = q_pre[ok]
        bad = torch.isnan(self.tilt)       # unreachable: keep the 60 deg poses (episode will fail)
        if bad.any():
            off = torch.where(self.flip[:, None], torch.as_tensor(t.offsets[(G.TILT_DEG, True)], device=dev, dtype=torch.float32),
                              torch.as_tensor(t.offsets[(G.TILT_DEG, False)], device=dev, dtype=torch.float32))
            W = G.grasp_poses(p_r, R_r, off, G.TILT_DEG, flip=self.flip)
            self.offset[bad] = off[bad]
            self.p_pre[bad], self.p_g[bad], self.p_lift[bad], self.R_g[bad] = W['pregrasp'][0][bad], W['grasp'][0][bad], W['lift'][0][bad], W['grasp'][1][bad]
        self.q_pre = best_q
        self.ik_ok = ~bad
        dq = (self.q_pre - self.q0).abs().max(1).values
        self.move_n = torch.clamp((dq / self.js / self.dt).ceil(), min=10).long()
        self.q_cmd = self.q0.clone()
        self.p_cmd, self.R_cmd = t.chain.fk(self.q_cmd)
        self.grip = torch.ones(N, device=dev)
        self.finger_prev = t.runtime.q[:, t.finger_qadr].sum(1)
        self.finger_still = torch.zeros(N, dtype=torch.long, device=dev)

    def _cart_step(self, p_wp, R_wp, v):
        dp = p_wp - self.p_cmd
        dist = dp.norm(dim=1, keepdim=True)
        self.p_cmd = self.p_cmd + dp * torch.clamp(v * self.dt / dist.clamp(min=1e-9), max=1.)
        w = K.rot_log(R_wp @ self.R_cmd.transpose(1, 2))
        ang = w.norm(dim=1, keepdim=True)
        w = w * torch.clamp(self.w * self.dt / ang.clamp(min=1e-9), max=1.)
        self.R_cmd = K.axis_angle_mat(w / w.norm(dim=1, keepdim=True).clamp(min=1e-9), w.norm(dim=1)) @ self.R_cmd
        return dist[:, 0], ang[:, 0]

    def act(self):
        """Normalized 14-d action for BimanualTask.step (left arm and gripper hold)."""
        t, ph = self.t, self.phase
        N = len(ph)
        # phase 0: joint-space min-jerk to the pre-grasp
        s = (self.timer.float() + 1) / self.move_n.float()
        b = s.clamp(max=1); b = b * b * (3 - 2 * b)
        q_joint = self.q0 + (self.q_pre - self.q0) * b[:, None]
        p_c, R_c = t.chain.fk(q_joint)
        m0 = ph == 0
        self.p_cmd = torch.where(m0[:, None], p_c, self.p_cmd)
        self.R_cmd = torch.where(m0[:, None, None], R_c, self.R_cmd)
        # Cartesian targets for later phases
        p_wp = torch.where((ph == 4)[:, None] | (ph >= 5)[:, None], self.p_lift, self.p_g)
        v = torch.where(ph >= 4, torch.full((N,), self.v, device=t.device), torch.full((N,), .05, device=t.device))
        dist, ang = self._cart_step(p_wp, self.R_g, v[:, None])
        self.p_cmd = torch.where(m0[:, None], p_c, self.p_cmd)
        self.R_cmd = torch.where(m0[:, None, None], R_c, self.R_cmd)
        q_ik, _, _ = self._ik(self.p_cmd, self.R_cmd, self.q_cmd, 12)
        self.q_cmd = torch.where(m0[:, None], q_joint, q_ik)
        # settle-before-close: integral on the true joint error once the command has arrived
        q = t.q_true()
        p_a, R_a = t.chain.fk(q)
        e_p = (self.p_g - p_a).norm(dim=1)
        e_r = torch.rad2deg(K.rot_log(self.R_g @ R_a.transpose(1, 2)).norm(dim=1))
        # Cartesian integral (true TCP vs its waypoint) mapped through the Jacobian: removes servo
        # sag and IK residuals alike. Active from the settle phase on.
        integrate = (ph >= 2)
        _, _, J = t.chain.fk(q, with_jac=True)
        p_tgt = torch.where((ph >= 4)[:, None], self.p_cmd, self.p_g)
        e6 = torch.cat([p_tgt - p_a, .5 * K.rot_log(self.R_g @ R_a.transpose(1, 2))], 1)
        JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, device=t.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e6[..., None]))[..., 0]
        self.integ = torch.where(integrate[:, None], (self.integ + self.ki * dq).clamp(-self.icap, self.icap), self.integ)
        q_phys = self.q_cmd + self.integ
        q_phys = torch.where(self.ik_ok[:, None], q_phys, self.q0)     # unreachable worlds hold (counted as failures)
        # phase logic
        self.timer += 1
        adv = torch.zeros_like(self.done)
        adv |= (ph == 0) & (self.timer >= self.move_n + 5) & self.ik_ok
        adv |= (ph == 1) & (dist < 2e-4) & (ang < 5e-3)
        settled = (e_p < self.settle_tol[0]) & (e_r < self.settle_tol[1])
        adv |= (ph == 2) & ((settled & (self.timer >= 5)) | (self.timer >= self.settle_max))
        # closed: the fingers have stopped (opening changes < 0.05 mm per step for 5 steps; the
        # real gripper reports its opening), at least close_s, at most 2.5 s. The squeeze force is
        # randomized and the fingers are damped, so closing time varies (0.5-0.9 s).
        fo = t.runtime.q[:, t.finger_qadr].sum(1)
        still = (fo - self.finger_prev).abs() < 5e-5
        self.finger_still = torch.where((ph == 3) & still, self.finger_still + 1, torch.zeros_like(self.finger_still))
        self.finger_prev = fo
        adv |= (ph == 3) & (((self.timer >= self.close_n) & (self.finger_still >= 5)) | (self.timer >= round(2.5 / self.dt)))
        # lifted: the command has arrived and the true tool orientation has settled (< lift_tol_deg;
        # weak wrist gains under the reader-plus-hand load take ~1 s at some poses), at most 3 s
        adv |= (ph == 4) & (dist < 5e-4) & (self.timer >= 10) & ((e_r < self.lift_tol) | (self.timer >= round(3.0 / self.dt)))
        adv |= (ph == 5) & (self.timer >= self.hold_n)
        if not hasattr(self, 'settle_err'):
            self.settle_err = torch.zeros(N, 2, device=t.device)
        rec = (ph == 2) & adv
        self.settle_err[rec] = torch.stack([e_p, e_r], 1)[rec]
        self.phase = torch.where(adv & (ph < 6), ph + 1, ph)
        self.timer = torch.where(adv, torch.zeros_like(self.timer), self.timer)
        self.done = self.phase >= 6
        self.grip = torch.where(self.phase >= 3, -torch.ones_like(self.grip), torch.ones_like(self.grip))
        # absolute physical target -> believed-frame target -> normalized accumulate increment
        believed = q_phys - t.enc_error[:, 0]
        a = torch.zeros(N, 14, device=t.device)
        a[:, :6] = ((believed - t.joint_target[:, 0]) / t.cfg.action_scale_rad).clamp(-1, 1)
        a[:, 6] = self.grip
        a[:, 13] = 1.
        return a
