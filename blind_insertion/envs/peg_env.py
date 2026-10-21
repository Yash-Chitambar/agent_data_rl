"""Batched blind peg insertion on MuJoCo Warp (paper §2, App. A.1).

All per-world state lives on the GPU. Physics: 1 kHz; control: 50 Hz (20 substeps,
CUDA-graph captured). The actor sees a 30-step history of [q, qdot, EE pose (6), EE twist
(6), previous action] computed from joint readings only. The critic additionally sees the
peg pose/velocity relative to the fixture and the 64 contact labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from blind_insertion.assets import builder as B
from blind_insertion.envs import kinematics as K

FREE, CONTACT, INSIDE = 0, 1, 2


@dataclass
class EnvCfg:
    robot: str = "flexiv"
    nworld: int = 4096
    seed: int = 0
    device: str = "cuda:0"
    substeps: int = 20
    episode_s: float = 5.0
    history: int = 30
    # poses (Table 1, peg column)
    fix_x: tuple = (0.54, 0.76)
    fix_y: tuple = (-0.29, 0.17)
    fix_z: tuple = (-0.20, 0.29)       # top face; with table_top: lift of the lowest corner
    table_top: float | None = None     # YAM lab workcell table (builder.YAM_TABLE); fixture rests on it
    fix_tilt_deg: float = 5.0
    grip_rot_deg: float = 5.0
    peg_offset: float = 0.0125          # per axis, uniform
    grasp_along: float = 0.0055         # along the pads
    inhand_tilt_deg: float = 10.0       # about the finger (closing) axis
    grasp_height_frac: float = 0.75     # pad centre height on the peg (fraction of length)
    start_gap: float = 0.0005           # lowest peg point above the plate at reset
    # contact (log-uniform ranges)
    plate_friction: tuple = (0.5, 2.0)
    plate_timeconst: tuple = (0.0035, 0.0071)
    finger_friction_scale: tuple = (1.41, 2.0)
    finger_timeconst_scale: tuple = (0.78, 1.34)
    grip_force_scale: tuple = (1.41, 2.83)
    grip_force_base: float | None = None   # N per pad (default: robot's nominal)
    pad_timeconst: float = 0.004           # pad contact solref time constant (x finger scale)
    pad_solimp: tuple | None = None        # e.g. (0.99, 0.999, 0.001) for light parts
    pad_spheres: int = 0                   # >0: n x n sphere grid per pad face (robust grasp contacts)
    mj_opt: dict | None = None             # MjModel.opt overrides, e.g. {"cone": 1, "impratio": 10,
                                           # "iterations": 100, "ls_iterations": 50} (cone 1 = elliptic)
    # labels / success / termination
    free_height: float = 0.001
    seat_tol: float = 0.0015
    far_xy_margin: float = 0.02
    far_z: float = 0.05
    slip_dist: float = 0.02
    action_clip: float = 3.0
    auto_reset: bool = True             # False for scripted checks
    success_tilt_deg: float = 0.0       # >0: also require the part axis within this tilt (USB-C: 2)
    speed_cap: float = 0.0              # >0: penalize EE speed above this (m/s) near the port
    speed_cap_radius: float = 0.005     # ... when the part is within this distance of the opening
    speed_coef: float = 1.0
    camera: bool = False                # render the wrist camera every step (visual policies)
    cam_res: int = 64
    cam_depth: bool = False
    # observation corruption (Phase C); 0 = off
    q_noise: float = 0.0
    qd_noise: float = 0.0
    q_quant_bits: int = 0
    latency: int = 0
    # buffers
    nconmax: int = 64
    njmax: int = 512
    action_scale: float | None = None
    gains: tuple | None = None
    backlash_arcmin: float = 0.0
    variants: list | None = None        # list[B.Variant]; default: study pegs
    n_variants: int = 1024
    variant_seed: int = 0
    variant_sampler: object = None
    extra: dict = field(default_factory=dict)


def _log_uniform(lo, hi, n, gen, device):
    u = torch.rand(n, generator=gen, device=device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _uniform(lo, hi, shape, gen, device):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


class PegEnv:
    def __init__(self, cfg: EnvCfg):
        self.cfg = cfg
        dev = self.device = torch.device(cfg.device)
        wp.init()
        variants = cfg.variants
        if variants is None:
            sampler = cfg.variant_sampler or B.sample_study_pegs
            variants = sampler(cfg.n_variants, cfg.variant_seed)
        self.mjm, self.lay = B.build(cfg.robot, variants, gains=cfg.gains,
                                     backlash_arcmin=cfg.backlash_arcmin, pad_spheres=cfg.pad_spheres,
                                     table_top=cfg.table_top)
        lay = self.lay
        if cfg.pad_solimp is not None:
            self.mjm.geom_solimp[lay.pad_geoms, :3] = cfg.pad_solimp
        for k_, v_ in (cfg.mj_opt or {}).items():
            setattr(self.mjm.opt, k_, v_)
        N = self.N = cfg.nworld
        self.nK = len(variants)
        self.n_arm = len(lay.arm_qadr)
        self.scale = cfg.action_scale or lay.robot.action_scale
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(cfg.seed)

        batched = ["geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound",
                   "geom_friction", "geom_solref", "body_pos", "body_quat", "body_mass",
                   "body_inertia", "body_ipos", "body_iquat", "body_invweight0", "dof_invweight0"]
        with wp.ScopedDevice(cfg.device):
            self.m = mjw.put_model(self.mjm, batch_sizes={f: N for f in batched})
            self.d = mjw.make_data(self.mjm, nworld=N, nconmax=cfg.nconmax, njmax=cfg.njmax)
            self.m.opt.warn_overflow = False  # we count overflow flags ourselves
        self.t_overflow = wp.to_torch(self.d.overflow)
        self.overflow_counts = torch.zeros(16, dtype=torch.long, device=dev)
        t = wp.to_torch
        self.t_qpos, self.t_qvel, self.t_ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.t_warm = t(self.d.qacc_warmstart)
        self.t_qacc = t(self.d.qacc)
        self.t_act = None
        self.t_time = t(self.d.time)
        self.t_mocap_pos, self.t_mocap_quat = t(self.d.mocap_pos), t(self.d.mocap_quat)
        self.f = {name: t(getattr(self.m, name)) for name in batched}

        # world -> variant (fixed: geometry is drawn once per world)
        self.variant = torch.arange(N, device=dev) % self.nK
        self._write_geometry()

        # per-variant tensors for labels / success
        tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
        P = max(len(p) for p in lay.hole_polys)
        polys = np.stack([np.vstack([p, np.repeat(p[-1:], P - len(p), 0)]) for p in lay.hole_polys])
        self.hole_poly = tt(polys)                                   # (K,P,2) ccw
        self.edge_pts = tt(lay.edge_points)                          # (K,64,2)
        self.depth = tt(lay.depth)
        self.peg_len = tt(lay.peg_length)
        self.half_short = tt(lay.peg_half_short)
        self.grasp_z = tt(lay.grasp_z)
        self.plate_half = tt([v.plate_size / 2 for v in variants])
        self.hole_r = tt([np.abs(p).max() for p in lay.hole_polys])
        self.fix_corners = tt(lay.fix_corners)                       # (K,8,3)

        self.chain = K.Chain(self.mjm, lay.tcp_site, lay.arm_qadr, device=dev)
        self.qpos0 = torch.as_tensor(self.mjm.qpos0, dtype=torch.float32, device=dev)
        self.home = torch.tensor(lay.robot.home, dtype=torch.float32, device=dev)
        # tcp in fixture: pointing into the fixture, rotated by the robot's nominal yaw
        yaw = math.radians(lay.robot.nom_yaw_deg)
        self.R_nom = (K.rpy_mat(torch.tensor([0.0, 0.0, yaw], device=dev))
                      @ K.rpy_mat(torch.tensor([math.pi, 0.0, 0.0], device=dev)))

        # episode state
        H, A = cfg.history, self.n_arm
        self.obs_dim = 2 * A + 12 + A
        self.hist = torch.zeros(N, H, self.obs_dim, device=dev)
        self.prev_action = torch.zeros(N, A, device=dev)
        self.act_queue = torch.zeros(N, max(cfg.latency, 0) + 1, A, device=dev)
        self.steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.captured = torch.zeros(N, dtype=torch.bool, device=dev)
        self.max_steps = int(round(cfg.episode_s / (cfg.substeps * self.mjm.opt.timestep)))
        self.fix_p = torch.zeros(N, 3, device=dev)
        self.fix_R = torch.eye(3, device=dev).repeat(N, 1, 1)
        self.grip_force = torch.zeros(N, device=dev)
        self.grasp_pt = torch.zeros(N, 3, device=dev)     # grasp point in peg frame
        self.start_xy = torch.zeros(N, 2, device=dev)
        self.ep_info = {}
        self.control_dt = cfg.substeps * self.mjm.opt.timestep
        self.m_travel = torch.zeros(N, device=dev)
        self.m_partial = torch.zeros(N, device=dev)
        self.m_tilt_partial = torch.zeros(N, device=dev)
        self.m_fail_inside = torch.zeros(N, dtype=torch.bool, device=dev)
        self.prev_xy = torch.zeros(N, 2, device=dev)

        self._graph = None
        if cfg.camera:
            with wp.ScopedDevice(cfg.device):
                self.rc = mjw.create_render_context(
                    self.mjm, nworld=N, cam_res=(cfg.cam_res, cfg.cam_res), render_rgb=True,
                    render_depth=cfg.cam_depth, use_textures=False, use_shadows=False)
                self._rgb = wp.zeros((N, cfg.cam_res, cfg.cam_res), dtype=wp.vec3)
                self._dep = wp.zeros((N, cfg.cam_res, cfg.cam_res), dtype=float)
            self.cam_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
            self.frame = torch.zeros(N, 3, cfg.cam_res, cfg.cam_res, dtype=torch.uint8, device=dev)
        self._physics(launch=False)   # compile + capture before the first reset (warm-up steps
        self.reset_all()              # must not advance real episodes)

    # ------------------------------------------------------------------ geometry
    def _write_geometry(self):
        lay, f, v = self.lay, self.f, self.variant.cpu().numpy()
        tt = lambda a: torch.as_tensor(np.asarray(a), device=self.device)  # noqa: E731
        fs, ps = torch.as_tensor(lay.fixture_slots), torch.as_tensor(lay.peg_slots)
        for slots, pre in ((fs, "fix"), (ps, "peg")):
            f["geom_dataid"][:, slots] = tt(getattr(lay, f"{pre}_dataid")[v]).to(torch.int32)
            f["geom_pos"][:, slots] = tt(getattr(lay, f"{pre}_pos")[v]).float()
            f["geom_quat"][:, slots] = tt(getattr(lay, f"{pre}_quat")[v]).float()
            f["geom_size"][:, slots] = tt(getattr(lay, f"{pre}_size")[v]).float()
            f["geom_aabb"][:, slots] = tt(getattr(lay, f"{pre}_aabb")[v]).float().reshape(
                self.N, len(slots), 2, 3)
            f["geom_rbound"][:, slots] = tt(getattr(lay, f"{pre}_rbound")[v]).float()
        b = lay.peg_body
        f["body_mass"][:, b] = tt(lay.peg_mass[v]).float()
        f["body_inertia"][:, b] = tt(lay.peg_inertia[v]).float()
        f["body_ipos"][:, b] = tt(lay.peg_ipos[v]).float()
        f["body_iquat"][:, b] = tt(lay.peg_iquat[v]).float()
        f["body_invweight0"][:, b] = tt(lay.peg_invweight[v]).float()
        a = lay.peg_dofadr
        f["dof_invweight0"][:, a:a + 6] = tt(lay.peg_dof_invweight[v]).float()

    def set_variants(self, mapping: torch.Tensor):
        """Reassign worlds to variants (curricula); takes effect immediately for geometry."""
        self.variant = mapping.to(self.device).long()
        self._write_geometry()
        if hasattr(self, "pool"):
            self.refresh_pool()

    # ------------------------------------------------------------------ reset
    def reset_all(self, desync: bool = False):
        self.reset(torch.arange(self.N, device=self.device))
        if desync:  # spread episode phases so training statistics are not synchronized
            self.steps[:] = torch.randint(0, self.max_steps, (self.N,), generator=self.gen,
                                          device=self.device)

    def sample_starts(self, worlds: torch.Tensor, gen: torch.Generator | None = None) -> dict:
        """Sample start states for the given worlds (with repetition allowed). IK on GPU."""
        cfg, dev, lay = self.cfg, self.device, self.lay
        gen = gen or self.gen
        n = worlds.numel()
        k = self.variant[worlds]
        deg = math.pi / 180
        todo = torch.ones(n, dtype=torch.bool, device=dev)
        out = dict(q=self.home.expand(n, -1).clone(), fix_p=torch.zeros(n, 3, device=dev),
                   fix_R=torch.eye(3, device=dev).repeat(n, 1, 1),
                   R_pg=torch.eye(3, device=dev).repeat(n, 1, 1), g_p=torch.zeros(n, 3, device=dev),
                   start_xy=torch.zeros(n, 2, device=dev))
        for _attempt in range(10):
            idx = todo.nonzero()[:, 0]
            m_ = idx.numel()
            if m_ == 0:
                break
            kk = k[idx]
            fp = torch.stack([_uniform(*cfg.fix_x, m_, gen, dev), _uniform(*cfg.fix_y, m_, gen, dev),
                              _uniform(*cfg.fix_z, m_, gen, dev)], 1)
            fR = K.rpy_mat(_uniform(-cfg.fix_tilt_deg, cfg.fix_tilt_deg, (m_, 3), gen, dev) * deg)
            if cfg.table_top is not None:   # lowest corner of the tilted fixture on the table
                low = (self.fix_corners[kk] @ fR.transpose(1, 2))[..., 2].min(1).values
                fp[:, 2] += cfg.table_top - low
            R_tf = self.R_nom @ K.rpy_mat(
                _uniform(-cfg.grip_rot_deg, cfg.grip_rot_deg, (m_, 3), gen, dev) * deg)
            # peg in tcp frame: peg z = -tcp z, tilted about the closing axis (tcp y)
            tilt = _uniform(-cfg.inhand_tilt_deg, cfg.inhand_tilt_deg, m_, gen, dev) * deg
            Rpg = K.rpy_mat(torch.stack([torch.full_like(tilt, math.pi), tilt, torch.zeros_like(tilt)], 1))
            gx = _uniform(-cfg.grasp_along, cfg.grasp_along, m_, gen, dev)
            gp = torch.stack([gx, torch.zeros_like(gx), self.grasp_z[kk]], 1)
            p_pt = -(Rpg @ gp[..., None])[..., 0]                 # peg origin in tcp frame
            R_pf = R_tf @ Rpg                                      # peg in fixture frame
            # peg origin xy offset; z so that the lowest bottom-edge point sits start_gap above
            off = _uniform(-cfg.peg_offset, cfg.peg_offset, (m_, 2), gen, dev)
            pts = torch.cat([self.edge_pts[kk], torch.zeros(m_, 64, 1, device=dev)], 2)
            zmin = (pts @ R_pf.transpose(1, 2))[..., 2].min(1).values
            p_pf = torch.cat([off, (cfg.start_gap - zmin)[:, None]], 1)
            p_tf = p_pf - (R_tf @ p_pt[..., None])[..., 0]
            Rw = fR @ R_tf
            pw = fp + (fR @ p_tf[..., None])[..., 0]
            q, pe, re = self.chain.ik(pw, Rw, self.home.expand(m_, -1), iters=60, q_rest=self.home)
            ok = (pe < 2e-4) & (re < 2e-3)
            good = idx[ok]
            out["q"][good] = q[ok]
            out["fix_p"][good], out["fix_R"][good] = fp[ok], fR[ok]
            out["R_pg"][good], out["g_p"][good], out["start_xy"][good] = Rpg[ok], gp[ok], off[ok]
            todo[good] = False
        self.ik_failures = int(todo.sum())
        out["plate_mu"] = _log_uniform(*cfg.plate_friction, n, gen, dev)
        out["plate_tc"] = _log_uniform(*cfg.plate_timeconst, n, gen, dev)
        out["finger_mu"] = _log_uniform(*cfg.finger_friction_scale, n, gen, dev)
        out["finger_tc"] = _log_uniform(*cfg.finger_timeconst_scale, n, gen, dev)
        base = lay.robot.grip_force if cfg.grip_force_base is None else (
            np.sign(lay.robot.grip_force) * cfg.grip_force_base)
        out["grip"] = base * _log_uniform(*cfg.grip_force_scale, n, gen, dev)
        return out

    def refresh_pool(self, per_world: int = 4):
        """Precompute `per_world` start states for every world (amortizes IK)."""
        w = torch.arange(self.N, device=self.device).repeat_interleave(per_world)
        st = self.sample_starts(w)
        self.pool = {k_: v.reshape(self.N, per_world, *v.shape[1:]) for k_, v in st.items()}
        self.pool_ptr = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.pool_size = per_world

    def reset(self, ids: torch.Tensor, starts: dict | None = None, render: bool = True):
        """Reset worlds `ids` from `starts` (dict of per-id tensors) or from the start pool."""
        if ids.numel() == 0:
            return
        if starts is None:
            if not hasattr(self, "pool"):
                self.refresh_pool()
            ptr = self.pool_ptr[ids]
            if (ptr >= self.pool_size).any():  # exhausted: refill those worlds lazily
                ex = ids[ptr >= self.pool_size]
                st = self.sample_starts(ex.repeat_interleave(self.pool_size))
                for k_, v in st.items():
                    self.pool[k_][ex] = v.reshape(ex.numel(), self.pool_size, *v.shape[1:])
                self.pool_ptr[ex] = 0
                ptr = self.pool_ptr[ids]
            starts = {k_: v[ids, ptr] for k_, v in self.pool.items()}
            self.pool_ptr[ids] += 1
        self._apply_starts(ids, starts)
        if render and self.cfg.camera and hasattr(self, "rc"):
            self.render()

    def _apply_starts(self, ids, st):
        cfg, dev, lay = self.cfg, self.device, self.lay
        k = self.variant[ids]
        q_sol = st["q"]
        # achieved tcp pose -> peg pose (world)
        p_a, R_a = self.chain.fk(q_sol)
        R_peg_w = R_a @ st["R_pg"]
        p_peg_w = p_a - (R_peg_w @ st["g_p"][..., None])[..., 0]
        qpos, qvel = self.t_qpos, self.t_qvel
        qpos[ids] = self.qpos0          # hidden joints (e.g. backlash) back to rest
        qpos[ids[:, None], torch.as_tensor(lay.arm_qadr, device=dev)] = q_sol
        half = self.half_short[k]
        fq = lay.robot.max_opening - half if lay.robot.name == "flexiv" else half
        for a in lay.finger_qadr:
            qpos[ids, int(a)] = fq
        pa = lay.peg_qadr
        qpos[ids, pa:pa + 3] = p_peg_w
        qpos[ids, pa + 3:pa + 7] = K.mat_to_quat(R_peg_w)
        qvel[ids] = 0
        self.t_warm[ids] = 0
        self.t_qacc[ids] = 0
        self.t_time[ids] = 0
        self.t_mocap_pos[ids, 0] = st["fix_p"]
        self.t_mocap_quat[ids, 0] = K.mat_to_quat(st["fix_R"])
        self.fix_p[ids], self.fix_R[ids] = st["fix_p"], st["fix_R"]
        self.grasp_pt[ids] = st["g_p"]
        self.start_xy[ids] = st["start_xy"]
        # contact parameters per reset
        fs = torch.as_tensor(lay.fixture_slots, device=dev)
        self.f["geom_friction"][ids[:, None], fs, 0] = st["plate_mu"][:, None]
        self.f["geom_solref"][ids[:, None], fs, 0] = st["plate_tc"][:, None]
        pads = torch.as_tensor(lay.pad_geoms, device=dev)
        self.f["geom_friction"][ids[:, None], pads, 0] = 1.0 * st["finger_mu"][:, None]
        self.f["geom_solref"][ids[:, None], pads, 0] = cfg.pad_timeconst * st["finger_tc"][:, None]
        self.grip_force[ids] = st["grip"]
        ctrl = self.t_ctrl
        ctrl[ids[:, None], torch.as_tensor(lay.arm_act, device=dev)] = q_sol
        for a in lay.grip_act:
            ctrl[ids, int(a)] = self.grip_force[ids]
        # episode bookkeeping
        self.steps[ids] = 0
        self.captured[ids] = False
        self.m_travel[ids] = 0
        self.m_partial[ids] = 0
        self.m_tilt_partial[ids] = 0
        self.m_fail_inside[ids] = False
        p0 = (st["fix_R"].transpose(1, 2) @ (p_peg_w - st["fix_p"])[..., None])[..., 0]
        self.prev_xy[ids] = p0[:, :2]
        self.prev_action[ids] = 0
        self.act_queue[ids] = 0
        o = self._actor_obs_now()[ids]
        self.hist[ids] = o[:, None, :].expand(-1, self.cfg.history, -1)

    # ------------------------------------------------------------------ physics
    def _physics(self, launch: bool = True):
        if self._graph is None:
            with wp.ScopedDevice(self.cfg.device):
                for _ in range(2):  # warm-up compiles kernels
                    mjw.step(self.m, self.d)
                with wp.ScopedCapture() as cap:
                    for _ in range(self.cfg.substeps):
                        mjw.step(self.m, self.d)
                self._graph = cap.graph
        if launch:
            wp.capture_launch(self._graph)

    # ------------------------------------------------------------------ observations
    def _measured(self):
        cfg, lay = self.cfg, self.lay
        q = self.t_qpos[:, torch.as_tensor(lay.arm_qadr, device=self.device)]
        qd = self.t_qvel[:, torch.as_tensor(lay.arm_dofadr, device=self.device)]
        if cfg.q_quant_bits > 0:
            res = 2 * math.pi / (2 ** cfg.q_quant_bits)
            q = torch.round(q / res) * res
        if cfg.q_noise > 0:
            q = q + cfg.q_noise * torch.randn(q.shape, generator=self.gen, device=self.device)
        if cfg.qd_noise > 0:
            qd = qd + cfg.qd_noise * torch.randn(qd.shape, generator=self.gen, device=self.device)
        return q, qd

    def render(self):
        """Wrist RGB (and depth) for every world at the current state -> self.frame (uint8)."""
        with wp.ScopedDevice(self.cfg.device):
            mjw.kinematics(self.m, self.d)
            mjw.camlight(self.m, self.d)
            mjw.refit_bvh(self.m, self.d, self.rc)
            mjw.render(self.m, self.d, self.rc)
            mjw.get_rgb(self.rc, self.cam_id, self._rgb)
            if self.cfg.cam_depth:
                mjw.get_depth(self.rc, self.cam_id, 1.0, self._dep)
        img = wp.to_torch(self._rgb)
        self.frame = (img.clamp(0, 1) * 255).to(torch.uint8).permute(0, 3, 1, 2).contiguous()
        if self.cfg.cam_depth:
            self.depth_img = wp.to_torch(self._dep).clone()
        return self.frame

    def _actor_obs_now(self):
        q, qd = self._measured()
        self._q_meas = q
        p, R, J = self.chain.fk(q, with_jac=True)
        tw = (J @ qd[..., None])[..., 0]
        eul = K.euler_xyz(self.R_world_nom().transpose(1, 2) @ R)
        return torch.cat([q, qd, p, eul, tw, self.prev_action], 1)

    def R_world_nom(self):
        return self.R_nom.expand(self.N, 3, 3)

    def peg_in_fixture(self):
        pa = self.lay.peg_qadr
        pw = self.t_qpos[:, pa:pa + 3]
        Rw = K.quat_to_mat(self.t_qpos[:, pa + 3:pa + 7])
        Rt = self.fix_R.transpose(1, 2)
        p = (Rt @ (pw - self.fix_p)[..., None])[..., 0]
        R = Rt @ Rw
        da = self.lay.peg_dofadr
        v = (Rt @ self.t_qvel[:, da:da + 3, None])[..., 0]
        w = (Rt @ Rw @ self.t_qvel[:, da + 3:da + 6, None])[..., 0]  # body -> fixture frame
        return p, R, v, w

    def edge_points_fixture(self, p, R):
        k = self.variant
        pts = torch.cat([self.edge_pts[k], torch.zeros(self.N, 64, 1, device=self.device)], 2)
        return pts @ R.transpose(1, 2) + p[:, None, :]

    def inside_outline(self, xy):
        """xy (N, M, 2) -> (N, M) inside the variant's convex ccw opening."""
        poly = self.hole_poly[self.variant]                       # (N,P,2)
        a = poly
        b = torch.roll(poly, -1, 1)
        e = (b - a)[:, None]                                      # (N,1,P,2)
        r = xy[:, :, None, :] - a[:, None]                        # (N,M,P,2)
        cross = e[..., 0] * r[..., 1] - e[..., 1] * r[..., 0]
        return (cross >= -1e-9).all(-1)

    def labels(self, pts):
        z = pts[..., 2]
        inside_xy = self.inside_outline(pts[..., :2])
        lab = torch.full(z.shape, CONTACT, dtype=torch.long, device=self.device)
        lab[z > self.cfg.free_height] = FREE
        lab[(z < 0) & inside_xy] = INSIDE
        return lab, inside_xy

    def critic_extra(self, p, R, v, w, lab):
        onehot = torch.nn.functional.one_hot(lab, 3).float().reshape(self.N, -1)
        rot6 = R[:, :, :2].transpose(1, 2).reshape(self.N, 6)
        return torch.cat([p * 10, rot6, v, w * 0.1, onehot], 1)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor):
        cfg, lay, dev = self.cfg, self.lay, self.device
        a = action.clamp(-cfg.action_clip, cfg.action_clip)
        if cfg.latency > 0:
            self.act_queue = torch.cat([self.act_queue[:, 1:], a[:, None]], 1)
            a_apply = self.act_queue[:, 0]
        else:
            a_apply = a
        q_t = self._q_meas + self.scale * a_apply
        self.t_ctrl[:, torch.as_tensor(lay.arm_act, device=dev)] = q_t
        torch.cuda.synchronize(dev)
        self._physics()
        wp.synchronize_device(cfg.device)
        ov = self.t_overflow
        if bool((ov != 0).any()):
            bits = torch.arange(16, device=dev)
            self.overflow_counts += ((ov[:, None] >> bits) & 1).sum(0)
            ov.zero_()
        self.prev_action = a.clone()
        self.steps += 1

        o = self._actor_obs_now()
        p, R, v, w = self.peg_in_fixture()
        pts = self.edge_points_fixture(p, R)
        lab, inside_xy = self.labels(pts)
        k = self.variant
        all_inside = (lab == INSIDE).all(1)
        seated = inside_xy.all(1) & (pts[..., 2] < -(self.depth[k] - cfg.seat_tol)[:, None]).all(1)
        tilt_now = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        if cfg.success_tilt_deg > 0:
            seated = seated & (tilt_now <= cfg.success_tilt_deg)
        # per-episode behaviour metrics (Table 5)
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        n_in = (lab == INSIDE).sum(1)
        partial = (n_in > 0) & (n_in < lab.shape[1])
        self.m_travel += (p[:, :2] - self.prev_xy).norm(dim=1)
        self.prev_xy = p[:, :2].clone()
        self.m_partial += partial.float()
        self.m_tilt_partial += tilt * partial.float()
        self.m_fail_inside |= all_inside
        new_capture = all_inside & ~self.captured
        self.captured |= all_inside
        reward = new_capture.float() + seated.float()
        if cfg.speed_cap > 0:
            near = (p[:, :2].norm(dim=1) < cfg.speed_cap_radius + self.hole_r[k]) & (p[:, 2] < 0.005)
            v_ee = o[:, 2 * self.n_arm + 6:2 * self.n_arm + 9].norm(dim=1)
            reward = reward - cfg.speed_coef * near.float() * (v_ee - cfg.speed_cap).clamp(min=0)
        # termination
        far = (p[:, :2].abs().max(1).values > self.plate_half[k] + cfg.far_xy_margin) | (p[:, 2] > cfg.far_z)
        tcp_p, tcp_R = self.chain.fk(self._q_meas)
        pa = lay.peg_qadr
        Rw = K.quat_to_mat(self.t_qpos[:, pa + 3:pa + 7])
        gp_w = self.t_qpos[:, pa:pa + 3] + (Rw @ self.grasp_pt[..., None])[..., 0]
        slip = (gp_w - tcp_p).norm(dim=1) > cfg.slip_dist
        bad = ~torch.isfinite(self.t_qpos).all(1) | ~torch.isfinite(o).all(1)
        timeout = self.steps >= self.max_steps
        terminated = seated | far | slip | bad
        done = terminated | timeout
        info = dict(success=seated, timeout=timeout & ~terminated, far=far, slip=slip, bad=bad,
                    captured=self.captured.clone(), labels=lab, steps=self.steps.clone(),
                    peg_p=p, peg_R=R, start_xy=self.start_xy.clone(), tilt=tilt,
                    travel=self.m_travel.clone(), t_partial=self.m_partial * self.control_dt,
                    tilt_partial=self.m_tilt_partial / self.m_partial.clamp(min=1),
                    n_inside=n_in)
        # history (before reset), then critic obs
        self.hist = torch.cat([self.hist[:, 1:], o[:, None]], 1)
        crit = self.critic_extra(p, R, v, w, lab)
        ids = done.nonzero()[:, 0] if cfg.auto_reset else done.new_zeros(0, dtype=torch.long)
        if ids.numel():
            info["final_hist"] = self.hist[ids].clone()
            self.reset(ids, render=False)
            p2, R2, v2, w2 = self.peg_in_fixture()
            lab2, _ = self.labels(self.edge_points_fixture(p2, R2))
            crit2 = self.critic_extra(p2, R2, v2, w2, lab2)
            crit[ids] = crit2[ids]
        reward[bad] = 0
        if self.cfg.camera:
            self.render()
        return self.hist, crit, reward, done, info

    def observe(self):
        p, R, v, w = self.peg_in_fixture()
        lab, _ = self.labels(self.edge_points_fixture(p, R))
        return self.hist, self.critic_extra(p, R, v, w, lab)

    @property
    def critic_dim(self):
        return 3 + 6 + 3 + 3 + 64 * 3

    def ncon(self):
        return wp.to_torch(self.d.nacon)
