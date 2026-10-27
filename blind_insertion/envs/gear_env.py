"""Batched blind gear meshing on MuJoCo Warp (paper §2.1, App. A.1, Table 1 gear column).

The robot holds an involute spur gear by its boss; the gear must slide down a shaft into
mesh with 1-2 hinged anchor gears and seat within 1 mm of the shaft's collar. Same actor
observation (33 per step, 30-step history), action scheme and physics loop as `PegEnv`
(which this subclasses for resets, rendering, observations and the CUDA graph).

Contact labels (128 x 3): 64 points on the shaft's top rim (free / on the gear / inside the
bore), then 64 points on the gear's tooth tips (free / on an anchor / in mesh).
Critic extra (401): gear pose rel. fixture (3 + 6D rot), lin/ang velocity (6), the one-hot
labels (384) and each anchor's tooth-phase error (2) -> with the 33 obs, 434 as the paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from blind_insertion.assets import gear_builder as GB
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.peg_env import CONTACT, FREE, INSIDE, EnvCfg, PegEnv, _log_uniform, _uniform

PAD_HALF = 0.01875      # half the pad height (builder._add_parallel_gripper)
N_RIM = 64
N_TIP = 64


@dataclass
class GearCfg(EnvCfg):
    episode_s: float = 10.0
    peg_offset: float = 0.0075            # initial gear position, per axis, from the shaft axis
    grasp_along: float = 0.0025
    inhand_tilt_deg: float = 5.0
    pad_bottom: tuple = (0.00675, 0.01025)  # pad bottoms above the gear's top face
    start_gap: float = 0.0005
    gains: tuple | None = ((40.0,) * 7, (20.0,) * 7)
    action_scale: float | None = 0.04
    # contact (log-uniform)
    plate_friction: tuple = (0.5, 2.0)    # plate, collar and shaft
    anchor_friction: tuple = (0.30, 0.67)
    hinge_damping: tuple = (3.2e-4, 3.2e-3)        # N m s / rad
    hinge_frictionloss: tuple = (2.7e-4, 1.9e-3)   # N m
    phase_min: float = 0.2                # |tooth-phase error| at reset, fraction of a pitch
    # labels / success / termination
    free_height: float = 0.001
    seat_tol: float = 0.001               # gear within 1 mm of the collar
    mesh_tol: float = 0.0005              # tip point this far below an anchor's top = in mesh
    mesh_depth: float = 0.001             # "meshed" with an anchor (bonus)
    far_xy: float = 0.035
    far_z: float = 0.06
    nconmax: int = 96                     # per world on average (pooled across worlds)
    nccdmax: int = 48                     # mesh-mesh (CCD) contacts per world, pooled
    njmax: int = 640
    pads_collide: bool = True             # False: scripted drop tests (the gear is not held)


class GearEnv(PegEnv):
    task = "gear"

    def __init__(self, cfg: GearCfg):  # noqa: PLR0915
        self.cfg = cfg
        dev = self.device = torch.device(cfg.device)
        wp.init()
        variants = cfg.variants
        if variants is None:
            sampler = cfg.variant_sampler or GB.sample_study_gears
            variants = sampler(cfg.n_variants, cfg.variant_seed)
        self.mjm, self.lay = GB.build(cfg.robot, variants, gains=cfg.gains,
                                      backlash_arcmin=cfg.backlash_arcmin)
        lay = self.lay
        if cfg.pad_solimp is not None:
            self.mjm.geom_solimp[lay.pad_geoms, :3] = cfg.pad_solimp
        if not cfg.pads_collide:   # pads and housing do not touch the part (drop tests)
            grip = list(lay.pad_geoms) + [mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, g)
                                          for g in lay.robot.gripper_geoms]
            self.mjm.geom_contype[grip] = 0
            self.mjm.geom_conaffinity[grip] = 0
        N = self.N = cfg.nworld
        self.nK = len(variants)
        self.n_arm = len(lay.arm_qadr)
        self.scale = cfg.action_scale or lay.robot.action_scale
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(cfg.seed)

        batched = ["geom_dataid", "geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound",
                   "geom_friction", "geom_solref", "body_pos", "body_quat", "body_mass",
                   "body_inertia", "body_ipos", "body_iquat", "body_invweight0", "dof_invweight0",
                   "dof_damping", "dof_frictionloss"]
        with wp.ScopedDevice(cfg.device):
            self.m = mjw.put_model(self.mjm, batch_sizes={f: N for f in batched})
            self.d = mjw.make_data(self.mjm, nworld=N, nconmax=cfg.nconmax, nccdmax=cfg.nccdmax,
                                   njmax=cfg.njmax)
            self.m.opt.warn_overflow = False
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

        self.variant = torch.arange(N, device=dev) % self.nK
        self._write_geometry()

        tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        self.tips = tt(lay.tips)                                   # (K,64,2) gear xy
        ang = torch.arange(N_RIM, device=dev) * (2 * math.pi / N_RIM)
        sr = tt(lay.shaft_r)
        self.rim = torch.stack([sr[:, None] * torch.cos(ang), sr[:, None] * torch.sin(ang),
                                tt(lay.shaft_top)[:, None].expand(-1, N_RIM)], 2)  # (K,64,3)
        self.ring = torch.stack([torch.cos(ang), torch.sin(ang)], 1)          # unit circle (64,2)
        self.bore_r, self.outer_r, self.shaft_r = tt(lay.bore_r), tt(lay.outer_r), sr
        self.shaft_top, self.fw = tt(lay.shaft_top), tt(lay.face_width)
        self.teeth = tt(lay.teeth)
        self.n_anchor = torch.as_tensor(lay.n_anchor, device=dev)
        self.a_present = torch.arange(GB.N_ANCHOR, device=dev)[None] < self.n_anchor[:, None]  # (K,2)
        self.a_center = tt(lay.anchor_center)                      # (K,2,2)
        self.a_outer = tt(lay.anchor_outer)                        # (K,2)
        self.a_teeth = tt(lay.anchor_teeth)
        self.a_azim = torch.atan2(self.a_center[..., 1], self.a_center[..., 0])
        self.a_qadr = torch.as_tensor(lay.anchor_qadr, device=dev)
        self.a_dofadr = torch.as_tensor(lay.anchor_dofadr, device=dev)
        # attributes PegEnv helpers / scripts expect
        self.depth = self.fw
        self.hole_r = self.bore_r
        self.plate_half = torch.full((self.nK,), GB.PLATE_HALF, device=dev)

        self.chain = K.Chain(self.mjm, lay.tcp_site, lay.arm_qadr, device=dev)
        self.qpos0 = torch.as_tensor(self.mjm.qpos0, dtype=torch.float32, device=dev)
        self.home = torch.tensor(lay.robot.home, dtype=torch.float32, device=dev)
        yaw = math.radians(lay.robot.nom_yaw_deg)
        self.R_nom = (K.rpy_mat(torch.tensor([0.0, 0.0, yaw], device=dev))
                      @ K.rpy_mat(torch.tensor([math.pi, 0.0, 0.0], device=dev)))

        H, A = cfg.history, self.n_arm
        assert cfg.action_mode == "relative", "GearEnv: relative actions only"
        self.obs_dim = 2 * A + 12 + A
        self.hist = torch.zeros(N, H, self.obs_dim, device=dev)
        self.prev_action = torch.zeros(N, A, device=dev)
        self.act_queue = torch.zeros(N, max(cfg.latency, 0) + 1, A, device=dev)
        self.steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.captured = torch.zeros(N, dtype=torch.bool, device=dev)
        self.meshed = torch.zeros(N, dtype=torch.bool, device=dev)
        self.max_steps = int(round(cfg.episode_s / (cfg.substeps * self.mjm.opt.timestep)))
        self.fix_p = torch.zeros(N, 3, device=dev)
        self.fix_R = torch.eye(3, device=dev).repeat(N, 1, 1)
        self.grip_force = torch.zeros(N, device=dev)
        self.grasp_pt = torch.zeros(N, 3, device=dev)
        self.start_xy = torch.zeros(N, 2, device=dev)
        self.start_phase = torch.zeros(N, GB.N_ANCHOR, device=dev)
        self.control_dt = cfg.substeps * self.mjm.opt.timestep
        self.m_travel = torch.zeros(N, device=dev)
        self.t_capture = torch.full((N,), -1.0, device=dev)
        self.t_mesh = torch.full((N,), -1.0, device=dev)
        self.prev_xy = torch.zeros(N, 2, device=dev)
        self.ep_info = {}

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
        self._physics(launch=False)
        self.reset_all()

    # ------------------------------------------------------------------ geometry
    def _write_geometry(self):
        lay, f, v = self.lay, self.f, self.variant.cpu().numpy()
        tt = lambda a: torch.as_tensor(a, device=self.device)  # noqa: E731
        for role, ids in GB.slot_groups(lay):
            s = lay.slot[role]
            sl = torch.as_tensor(ids)
            f["geom_dataid"][:, sl] = tt(s["dataid"][v]).to(torch.int32)
            f["geom_pos"][:, sl] = tt(s["pos"][v]).float()
            f["geom_quat"][:, sl] = tt(s["quat"][v]).float()
            f["geom_size"][:, sl] = tt(s["size"][v]).float()
            f["geom_aabb"][:, sl] = tt(s["aabb"][v]).float().reshape(self.N, len(ids), 2, 3)
            f["geom_rbound"][:, sl] = tt(s["rbound"][v]).float()
        bd = lay.body
        b = lay.gear_body
        f["body_mass"][:, b] = tt(bd["gear_mass"][v]).float()
        f["body_inertia"][:, b] = tt(bd["gear_inertia"][v]).float()
        f["body_ipos"][:, b] = tt(bd["gear_ipos"][v]).float()
        f["body_iquat"][:, b] = tt(bd["gear_iquat"][v]).float()
        f["body_invweight0"][:, b] = tt(bd["gear_invweight"][v]).float()
        a = lay.gear_dofadr
        f["dof_invweight0"][:, a:a + 6] = tt(bd["gear_dof_invweight"][v]).float()
        for j in range(GB.N_ANCHOR):
            ab = int(lay.anchor_body[j])
            f["body_mass"][:, ab] = tt(bd["anchor_mass"][v, j]).float()
            f["body_inertia"][:, ab] = tt(bd["anchor_inertia"][v, j]).float()
            f["body_ipos"][:, ab] = tt(bd["anchor_ipos"][v, j]).float()
            f["body_iquat"][:, ab] = tt(bd["anchor_iquat"][v, j]).float()
            f["body_invweight0"][:, ab] = tt(bd["anchor_invweight"][v, j]).float()
            f["body_pos"][:, ab] = tt(bd["anchor_pos"][v, j]).float()
            f["dof_invweight0"][:, int(lay.anchor_dofadr[j])] = tt(bd["anchor_dof_invweight"][v, j]).float()

    # ------------------------------------------------------------------ starts
    def _rest_height(self, k, R, off):
        """Gear origin z (fixture frame) so it rests start_gap above the shaft tip (or above an
        anchor's top where the tilted gear would otherwise touch it first)."""
        cfg, dev = self.cfg, self.device
        n = k.numel()
        gap = cfg.start_gap
        nrm = R[:, :, 2]                                          # gear z in fixture
        neg = torch.full((n,), -1e9, device=dev)
        p0 = torch.cat([off, self.shaft_top[k][:, None]], 1)
        Rt = R.transpose(1, 2)

        def below_plane(P, valid):
            # z such that every valid fixture point P lies `gap` below the gear's bottom plane
            zz = ((P * nrm[:, None]).sum(-1) - (nrm[:, None, :2] * off[:, None]).sum(-1) + gap) / nrm[:, None, 2]
            return torch.where(valid, zz, neg[:, None]).max(1).values

        def above(Q, zmin, valid):
            # z such that gear-frame points Q (z=0) lie at or above zmin
            h = (Q @ R.transpose(1, 2))[..., 2]
            return torch.where(valid, zmin[:, None] - h, neg[:, None]).max(1).values

        # 1. shaft rim points under the gear's annulus
        S = self.rim[k]
        qs = ((S - p0[:, None]) @ R)                               # gear frame (R^T x)
        rg = qs[..., :2].norm(dim=-1)
        z1 = below_plane(S, (rg >= self.bore_r[k][:, None]) & (rg <= self.outer_r[k][:, None]))
        # 2. bore edge points over the shaft's top face
        Qb = torch.cat([self.ring[None] * self.bore_r[k][:, None, None], torch.zeros(n, N_RIM, 1, device=dev)], 2)
        Pb = Qb @ R.transpose(1, 2) + torch.cat([off, torch.zeros(n, 1, device=dev)], 1)[:, None]
        z2 = above(Qb, self.shaft_top[k] + gap, Pb[..., :2].norm(dim=-1) <= self.shaft_r[k][:, None])
        zs = [z1, z2]
        # 3./4. tooth tips over anchors, anchor rims under the gear
        Qt = torch.cat([self.tips[k], torch.zeros(n, N_TIP, 1, device=dev)], 2)
        Pt = Qt @ R.transpose(1, 2) + torch.cat([off, torch.zeros(n, 1, device=dev)], 1)[:, None]
        for j in range(GB.N_ANCHOR):
            c, ro, pres = self.a_center[k, j], self.a_outer[k, j], self.a_present[k, j]
            dj = (Pt[..., :2] - c[:, None]).norm(dim=-1)
            zs.append(above(Qt, self.fw[k] + gap, (dj < ro[:, None] + 0.001) & pres[:, None]))
            A = torch.cat([c[:, None] + self.ring[None] * ro[:, None, None],
                           self.fw[k][:, None, None].expand(-1, N_RIM, 1)], 2)
            qa = (A - p0[:, None]) @ R
            ra = qa[..., :2].norm(dim=-1)
            zs.append(below_plane(A, (ra >= self.bore_r[k][:, None]) & (ra <= self.outer_r[k][:, None])
                                  & pres[:, None]))
        z = torch.stack(zs, 1).max(1).values
        del Rt
        return torch.where(z > -1e8, z, self.shaft_top[k] + gap)

    def sample_starts(self, worlds: torch.Tensor, gen: torch.Generator | None = None) -> dict:
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
            R_tf = self.R_nom @ K.rpy_mat(
                _uniform(-cfg.grip_rot_deg, cfg.grip_rot_deg, (m_, 3), gen, dev) * deg)
            tilt = _uniform(-cfg.inhand_tilt_deg, cfg.inhand_tilt_deg, m_, gen, dev) * deg
            Rpg = K.rpy_mat(torch.stack([torch.full_like(tilt, math.pi), tilt, torch.zeros_like(tilt)], 1))
            gx = _uniform(-cfg.grasp_along, cfg.grasp_along, m_, gen, dev)
            hp = _uniform(*cfg.pad_bottom, m_, gen, dev)
            gp = torch.stack([gx, torch.zeros_like(gx), self.fw[kk] + hp + PAD_HALF], 1)
            p_pt = -(Rpg @ gp[..., None])[..., 0]
            R_pf = R_tf @ Rpg
            off = _uniform(-cfg.peg_offset, cfg.peg_offset, (m_, 2), gen, dev)
            z = self._rest_height(kk, R_pf, off)
            p_pf = torch.cat([off, z[:, None]], 1)
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
        A = GB.N_ANCHOR
        sgn = torch.where(torch.rand(n, A, generator=gen, device=dev) < 0.5, -1.0, 1.0)
        out["phase"] = sgn * _uniform(cfg.phase_min, 0.5, (n, A), gen, dev)
        out["plate_mu"] = _log_uniform(*cfg.plate_friction, n, gen, dev)
        out["anchor_mu"] = _log_uniform(*cfg.anchor_friction, n, gen, dev)
        out["plate_tc"] = _log_uniform(*cfg.plate_timeconst, n, gen, dev)
        out["damp"] = _log_uniform(*cfg.hinge_damping, n * A, gen, dev).reshape(n, A)
        out["floss"] = _log_uniform(*cfg.hinge_frictionloss, n * A, gen, dev).reshape(n, A)
        out["finger_mu"] = _log_uniform(*cfg.finger_friction_scale, n, gen, dev)
        out["finger_tc"] = _log_uniform(*cfg.finger_timeconst_scale, n, gen, dev)
        base = lay.robot.grip_force if cfg.grip_force_base is None else (
            math.copysign(cfg.grip_force_base, lay.robot.grip_force))
        out["grip"] = base * _log_uniform(*cfg.grip_force_scale, n, gen, dev)
        return out

    def anchor_angles_for_phase(self, k, R_pf, e):
        """Hinge angles giving tooth-phase error e (fraction of a pitch) for gear rotation R_pf."""
        th_g = torch.atan2(R_pf[:, 1, 0], R_pf[:, 0, 0])
        pg = 2 * math.pi / self.teeth[k]
        pa = 2 * math.pi / self.a_teeth[k]                           # (n,2)
        phi = self.a_azim[k]
        u_g = torch.remainder((phi - th_g[:, None]) / pg[:, None], 1.0)
        u_a = torch.remainder(0.5 - u_g + e, 1.0)
        th_a = phi + math.pi - u_a * pa
        return torch.atan2(torch.sin(th_a), torch.cos(th_a))

    def phase_error(self, R=None):
        """(N,2) tooth-phase error of the gear vs each anchor, in pitches; 0 = in mesh."""
        if R is None:
            _, R, _, _ = self.peg_in_fixture()
        k = self.variant
        th_g = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        th_a = self.t_qpos[:, self.a_qadr]
        pg = 2 * math.pi / self.teeth[k]
        pa = 2 * math.pi / self.a_teeth[k]
        phi = self.a_azim[k]
        u_g = torch.remainder((phi - th_g[:, None]) / pg[:, None], 1.0)
        u_a = torch.remainder((phi + math.pi - th_a) / pa, 1.0)
        e = torch.remainder(u_g + u_a, 1.0) - 0.5
        return torch.where(self.a_present[k], e, torch.zeros_like(e))

    def _apply_starts(self, ids, st):
        cfg, dev, lay = self.cfg, self.device, self.lay
        k = self.variant[ids]
        q_sol = st["q"]
        p_a, R_a = self.chain.fk(q_sol)
        R_g_w = R_a @ st["R_pg"]
        p_g_w = p_a - (R_g_w @ st["g_p"][..., None])[..., 0]
        qpos, qvel = self.t_qpos, self.t_qvel
        qpos[ids] = self.qpos0
        qpos[ids[:, None], torch.as_tensor(lay.arm_qadr, device=dev)] = q_sol
        half = GB.BOSS_D / 2
        fq = lay.robot.max_opening - half if lay.robot.name == "flexiv" else half
        for a in lay.finger_qadr:
            qpos[ids, int(a)] = fq
        ga = lay.gear_qadr
        qpos[ids, ga:ga + 3] = p_g_w
        qpos[ids, ga + 3:ga + 7] = K.mat_to_quat(R_g_w)
        R_pf = st["fix_R"].transpose(1, 2) @ R_g_w
        qa = self.anchor_angles_for_phase(k, R_pf, st["phase"])
        qpos[ids[:, None], self.a_qadr] = qa
        qvel[ids] = 0
        self.t_warm[ids] = 0
        self.t_qacc[ids] = 0
        self.t_time[ids] = 0
        self.t_mocap_pos[ids, 0] = st["fix_p"]
        self.t_mocap_quat[ids, 0] = K.mat_to_quat(st["fix_R"])
        self.fix_p[ids], self.fix_R[ids] = st["fix_p"], st["fix_R"]
        self.grasp_pt[ids] = st["g_p"]
        self.start_xy[ids] = st["start_xy"]
        self.start_phase[ids] = st["phase"]
        # contact parameters per reset
        fs = torch.as_tensor(lay.fix_slots, device=dev)
        self.f["geom_friction"][ids[:, None], fs, 0] = st["plate_mu"][:, None]
        self.f["geom_solref"][ids[:, None], fs, 0] = st["plate_tc"][:, None]
        asl = torch.as_tensor(lay.anchor_slots.reshape(-1), device=dev)
        self.f["geom_friction"][ids[:, None], asl, 0] = st["anchor_mu"][:, None]
        self.f["geom_solref"][ids[:, None], asl, 0] = st["plate_tc"][:, None]
        self.f["dof_damping"][ids[:, None], self.a_dofadr] = st["damp"]
        self.f["dof_frictionloss"][ids[:, None], self.a_dofadr] = st["floss"]
        pads = torch.as_tensor(lay.pad_geoms, device=dev)
        self.f["geom_friction"][ids[:, None], pads, 0] = 1.0 * st["finger_mu"][:, None]
        self.f["geom_solref"][ids[:, None], pads, 0] = cfg.pad_timeconst * st["finger_tc"][:, None]
        self.grip_force[ids] = st["grip"]
        ctrl = self.t_ctrl
        ctrl[ids[:, None], torch.as_tensor(lay.arm_act, device=dev)] = q_sol
        for a in lay.grip_act:
            ctrl[ids, int(a)] = self.grip_force[ids]
        self.steps[ids] = 0
        self.captured[ids] = False
        self.meshed[ids] = False
        self.m_travel[ids] = 0
        self.t_capture[ids] = -1
        self.t_mesh[ids] = -1
        p0 = (st["fix_R"].transpose(1, 2) @ (p_g_w - st["fix_p"])[..., None])[..., 0]
        self.prev_xy[ids] = p0[:, :2]
        self.prev_action[ids] = 0
        self.act_queue[ids] = 0
        o = self._actor_obs_now()[ids]
        self.hist[ids] = o[:, None, :].expand(-1, self.cfg.history, -1)

    # ------------------------------------------------------------------ state
    def peg_in_fixture(self):
        """Gear pose / velocity in the fixture frame (name kept for PegEnv-compatible code)."""
        ga = self.lay.gear_qadr
        pw = self.t_qpos[:, ga:ga + 3]
        Rw = K.quat_to_mat(self.t_qpos[:, ga + 3:ga + 7])
        Rt = self.fix_R.transpose(1, 2)
        p = (Rt @ (pw - self.fix_p)[..., None])[..., 0]
        R = Rt @ Rw
        da = self.lay.gear_dofadr
        v = (Rt @ self.t_qvel[:, da:da + 3, None])[..., 0]
        w = (Rt @ Rw @ self.t_qvel[:, da + 3:da + 6, None])[..., 0]
        return p, R, v, w

    gear_in_fixture = peg_in_fixture

    def contact_state(self, p, R):
        """Labels (N,128) and the masks used for success / bonuses."""
        cfg, k = self.cfg, self.variant
        # shaft rim in the gear frame
        qs = (self.rim[k] - p[:, None]) @ R
        rg, zg = qs[..., :2].norm(dim=-1), qs[..., 2]
        inside = (rg < self.bore_r[k][:, None]) & (zg > 0)
        touch = (zg > -cfg.free_height) & (rg < self.outer_r[k][:, None] + 0.001)
        lab_s = torch.full(rg.shape, FREE, dtype=torch.long, device=self.device)
        lab_s[touch] = CONTACT
        lab_s[inside] = INSIDE
        # tooth tips in the fixture frame vs each anchor
        Q = torch.cat([self.tips[k], torch.zeros(self.N, N_TIP, 1, device=self.device)], 2)
        P = Q @ R.transpose(1, 2) + p[:, None]
        fw = self.fw[k][:, None]
        in_mesh = torch.zeros(P.shape[:2], dtype=torch.bool, device=self.device)
        on = torch.zeros_like(in_mesh)
        meshed = torch.ones(self.N, GB.N_ANCHOR, dtype=torch.bool, device=self.device)
        for j in range(GB.N_ANCHOR):
            pres = self.a_present[k, j][:, None]
            dj = (P[..., :2] - self.a_center[k, j][:, None]).norm(dim=-1)
            foot = (dj < self.a_outer[k, j][:, None]) & pres
            in_mesh |= foot & (P[..., 2] < fw - cfg.mesh_tol)
            on |= (dj < self.a_outer[k, j][:, None] + 0.0005) & pres & (P[..., 2] < fw + cfg.free_height)
            mj = (foot & (P[..., 2] < fw - cfg.mesh_depth)).any(1)
            meshed[:, j] = mj | ~pres[:, 0]
        lab_t = torch.full(in_mesh.shape, FREE, dtype=torch.long, device=self.device)
        lab_t[on] = CONTACT
        lab_t[in_mesh] = INSIDE
        return torch.cat([lab_s, lab_t], 1), dict(inside=inside, meshed=meshed)

    def current_labels(self):
        p, R, _, _ = self.peg_in_fixture()
        return self.contact_state(p, R)[0]

    n_labels = N_RIM + N_TIP

    def critic_extra(self, p, R, v, w, lab):
        onehot = torch.nn.functional.one_hot(lab, 3).float().reshape(self.N, -1)
        rot6 = R[:, :, :2].transpose(1, 2).reshape(self.N, 6)
        return torch.cat([p * 10, rot6, v, w * 0.1, onehot, self.phase_error(R)], 1)

    @property
    def critic_dim(self):
        return 3 + 6 + 3 + 3 + self.n_labels * 3 + GB.N_ANCHOR

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
        lab, cs = self.contact_state(p, R)
        cap_now = cs["inside"].all(1)
        mesh_now = cs["meshed"].all(1)
        seated = cap_now & (p[:, 2] < cfg.seat_tol)
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        self.m_travel += (p[:, :2] - self.prev_xy).norm(dim=1)
        self.prev_xy = p[:, :2].clone()
        t_now = self.steps.float() * self.control_dt
        new_cap = cap_now & ~self.captured
        new_mesh = mesh_now & ~self.meshed
        self.t_capture = torch.where(new_cap, t_now, self.t_capture)
        self.t_mesh = torch.where(new_mesh, t_now, self.t_mesh)
        self.captured |= cap_now
        self.meshed |= mesh_now
        reward = new_cap.float() + new_mesh.float() + seated.float()
        far = (p[:, :2].norm(dim=1) > cfg.far_xy) | (p[:, 2] > cfg.far_z)
        tcp_p, _ = self.chain.fk(self._q_meas)
        ga = lay.gear_qadr
        Rw = K.quat_to_mat(self.t_qpos[:, ga + 3:ga + 7])
        gp_w = self.t_qpos[:, ga:ga + 3] + (Rw @ self.grasp_pt[..., None])[..., 0]
        slip = (gp_w - tcp_p).norm(dim=1) > cfg.slip_dist
        bad = ~torch.isfinite(self.t_qpos).all(1) | ~torch.isfinite(o).all(1)
        timeout = self.steps >= self.max_steps
        terminated = seated | far | slip | bad
        done = terminated | timeout
        n_in = (lab[:, :N_RIM] == INSIDE).sum(1)
        info = dict(success=seated, timeout=timeout & ~terminated, far=far, slip=slip, bad=bad,
                    captured=self.captured.clone(), meshed=self.meshed.clone(), labels=lab,
                    steps=self.steps.clone(), peg_p=p, peg_R=R, start_xy=self.start_xy.clone(),
                    tilt=tilt, travel=self.m_travel.clone(), t_capture=self.t_capture.clone(),
                    t_mesh=self.t_mesh.clone(), n_inside=n_in,
                    n_mesh=(lab[:, N_RIM:] == INSIDE).sum(1), phase=self.phase_error(R))
        self.hist = torch.cat([self.hist[:, 1:], o[:, None]], 1)
        crit = self.critic_extra(p, R, v, w, lab)
        ids = done.nonzero()[:, 0] if cfg.auto_reset else done.new_zeros(0, dtype=torch.long)
        if ids.numel():
            info["final_hist"] = self.hist[ids].clone()
            self.reset(ids, render=False)
            p2, R2, v2, w2 = self.peg_in_fixture()
            lab2, _ = self.contact_state(p2, R2)
            crit2 = self.critic_extra(p2, R2, v2, w2, lab2)
            crit[ids] = crit2[ids]
        reward[bad] = 0
        if self.cfg.camera:
            self.render()
        return self.hist, crit, reward, done, info

    def observe(self):
        p, R, v, w = self.peg_in_fixture()
        lab, _ = self.contact_state(p, R)
        return self.hist, self.critic_extra(p, R, v, w, lab)
