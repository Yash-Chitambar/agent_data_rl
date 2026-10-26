"""Task H pick phase in MuJoCo Warp: the right YAM picks the SD reader off the table and brings
its plug in front of the mini PC's USB-C port. Batched worlds, the full scene of
`scenes.pc_bench` (same physics and geometry as the insertion env), two cameras (fixed top and
wrist D405), and a privileged scripted expert that generates ACT demonstrations.

Control at 50 Hz (20 x 1 ms substeps). Action (7): absolute arm joint targets (6) and a gripper
command (+1 open, -1 close); the fingers are the insertion env's constant-force motors.

    env = PickEnv(PickCfg(nworld=256)); ex = Expert(env)
    env.reset_all(); ex.reset()
    for t in range(env.T): a, done = ex.act(); env.step(a, noise=...); frames = env.render()
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
from blind_insertion.scenes import pc_bench as S


@dataclass
class PickCfg:
    nworld: int = 256
    device: str = "cuda:0"
    seed: int = 0
    episode_s: float = 12.0
    substeps: int = 20
    cams: tuple = ("top", "wrist_cam")
    cam_res: tuple = (320, 320)          # width, height
    render: bool = True
    physical_rig: bool = True
    use_textures: bool = True  # photo-referenced workbench material
    # layout randomization (arm base frame)
    port_xy: tuple = S.PORT_XY
    port_xy_range: tuple = (0.02, 0.04)  # +- x, y
    pc_yaw_deg: float = 10.0
    reader_xy: tuple = S.READER_XY
    reader_xy_range: tuple = (0.03, 0.05)
    reader_yaw_deg: float = 20.0
    start_q_noise: float = 0.08          # rad, around the ready pose
    # gripper (N per finger): open pushes the fingers apart, close squeezes
    grip_open: float = 4.0
    grip_close: float = -12.0
    # render randomization: top camera mount (uncalibrated: +-3 cm, +-4 deg), wrist camera mount,
    # lights, colours
    rand: dict = field(default_factory=lambda: dict(cam_pos=0.03, cam_rot_deg=4.0, wrist_pos=0.003,
                                                    wrist_rot_deg=2.0, light=True, colors=True))
    # physics / sensor randomization per reset ({} = nominal): friction scale of reader, pads and
    # table (log-uniform), grip squeeze (N), arm kp and kd scales (log-uniform), joint friction (N m,
    # log-uniform), command latency (control steps, one level per world), encoder noise (rad)
    phys: dict = field(default_factory=lambda: dict(friction=(0.6, 1.6), grip_close=(-15.0, -9.0),
                                                    kp_scale=(0.8, 1.25), kd_scale=(0.7, 1.4),
                                                    frictionloss=((.01,.4),(.4,2.),(.6,3.),(.08,.5),(.08,.5),(.04,.3)), latency=(0, 1),
                                                    q_noise=0.0005))
    nconmax: int = 160
    njmax: int = 1200


def _u(lo, hi, shape, gen, dev):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=dev)


class PickEnv:
    def __init__(self, cfg: PickCfg, variant: B.Variant | None = None):
        self.cfg = cfg
        dev = self.device = torch.device(cfg.device)
        if cfg.physical_rig and cfg.cam_res[0] != cfg.cam_res[1]:
            raise ValueError("Calibrated policy views require square cam_res; use physical_rig=False for legacy checkpoints")
        wp.init()
        self.gripper = "flexpoint" if cfg.physical_rig else "linear"
        self.mjm, self.lay = S.build_scene(variant, physical=cfg.physical_rig)
        mjm, lay = self.mjm, self.lay
        N = self.N = cfg.nworld
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(cfg.seed)
        batched = ["geom_rgba", "light_pos", "light_dir", "light_diffuse", "cam_pos", "cam_quat", "geom_friction",
                   "actuator_gainprm", "actuator_biasprm", "dof_frictionloss"]
        with wp.ScopedDevice(cfg.device):
            self.m = mjw.put_model(mjm, batch_sizes={f: N for f in batched})
            self.d = mjw.make_data(mjm, nworld=N, nconmax=cfg.nconmax, njmax=cfg.njmax)
            self.m.opt.warn_overflow = False
        t = wp.to_torch
        self.t_qpos, self.t_qvel, self.t_ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.t_warm, self.t_qacc, self.t_time = t(self.d.qacc_warmstart), t(self.d.qacc), t(self.d.time)
        self.t_mocap_pos, self.t_mocap_quat = t(self.d.mocap_pos), t(self.d.mocap_quat)
        self.f = {n: t(getattr(self.m, n)) for n in batched}
        self.f0 = {n: v[0].clone() for n, v in self.f.items()}
        self.qpos0 = torch.as_tensor(mjm.qpos0, dtype=torch.float32, device=dev)
        self.chain = K.Chain(mjm, lay.tcp_site, lay.arm_qadr, device=dev)
        self.arm_q = torch.as_tensor(lay.arm_qadr, device=dev)
        self.arm_act = torch.as_tensor(lay.arm_act, device=dev)
        self.home = torch.tensor(lay.robot.home, dtype=torch.float32, device=dev)
        self.R_base = K.rpy_mat(torch.tensor(S.PORT_BASE_RPY_DEG, device=dev) * math.pi / 180)
        self.R_top = torch.tensor(S.R_TOP_TCP, device=dev)
        v = lay.variants[0]
        self.body_z = float(v.peg_boxes[0][0][2])          # body centre along the part axis
        self.body_half_t = float(v.peg_boxes[0][1][1])
        self.T = int(round(cfg.episode_s / (cfg.substeps * mjm.opt.timestep)))
        self.control_dt = cfg.substeps * mjm.opt.timestep
        # ready pose: tcp 18 cm above the reader area, top-down, plug direction +x
        q, pe, _ = self.chain.ik(torch.tensor([[cfg.reader_xy[0], cfg.reader_xy[1], 0.18]], device=dev),
                                 (self.R_base @ self.R_top)[None], self.home[None], iters=200, q_rest=self.home)
        assert pe.item() < 1e-3, "ready pose IK failed"
        self.q_ready = q[0]
        self.cam_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cfg.cams]
        self.top_cam = self.cam_ids[cfg.cams.index("top")]
        self.wrist_cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        self.max_lat = max(cfg.phys.get("latency", (0,)))
        self.lat = torch.zeros(N, dtype=torch.long, device=dev)
        self.grip_close = torch.full((N,), cfg.grip_close, device=dev)
        self._graph = None
        if cfg.render:
            act = [i in self.cam_ids for i in range(mjm.ncam)]
            order = [i for i in range(mjm.ncam) if act[i]]    # render-context camera order
            self.rc_index = [order.index(c) for c in self.cam_ids]
            W, H = cfg.cam_res
            with wp.ScopedDevice(cfg.device):
                self.rc = mjw.create_render_context(mjm, nworld=N, cam_res=[(W, H)] * len(order),
                                                    render_rgb=True, render_depth=False, use_textures=cfg.use_textures,
                                                    use_shadows=False, cam_active=act)
                self._rgb = wp.zeros((N, H, W), dtype=wp.vec3)
        self._physics(launch=False)
        self.reset_all()

    # ------------------------------------------------------------------ reset
    def reset_all(self):
        cfg, dev, gen, lay, N = self.cfg, self.device, self.gen, self.lay, self.N
        deg = math.pi / 180
        self.t_qpos[:] = self.qpos0
        self.t_qvel[:] = 0
        self.t_warm[:] = 0
        self.t_qacc[:] = 0
        self.t_time[:] = 0
        # PC: port position and yaw
        pp = torch.stack([cfg.port_xy[0] + _u(-cfg.port_xy_range[0], cfg.port_xy_range[0], N, gen, dev),
                          cfg.port_xy[1] + _u(-cfg.port_xy_range[1], cfg.port_xy_range[1], N, gen, dev),
                          torch.full((N,), S.PORT_HEIGHT, device=dev)], 1)
        yaw = _u(-cfg.pc_yaw_deg, cfg.pc_yaw_deg, N, gen, dev) * deg
        self.port_p = pp
        self.port_R = self._yaw(yaw) @ self.R_base
        self.t_mocap_pos[:, 0] = pp
        self.t_mocap_quat[:, 0] = K.mat_to_quat(self.port_R)
        # reader flat on the table, plug towards +x (turned by yaw)
        ry = _u(-cfg.reader_yaw_deg, cfg.reader_yaw_deg, N, gen, dev) * deg
        Rr = self._yaw(ry) @ self.R_base
        c = torch.stack([cfg.reader_xy[0] + _u(-cfg.reader_xy_range[0], cfg.reader_xy_range[0], N, gen, dev),
                         cfg.reader_xy[1] + _u(-cfg.reader_xy_range[1], cfg.reader_xy_range[1], N, gen, dev)], 1)
        body = (Rr @ torch.tensor([0.0, 0.0, self.body_z], device=dev))       # body centre offset
        pa = lay.peg_qadr
        self.t_qpos[:, pa] = c[:, 0] - body[:, 0]
        self.t_qpos[:, pa + 1] = c[:, 1] - body[:, 1]
        self.t_qpos[:, pa + 2] = self.body_half_t + 1e-4
        self.t_qpos[:, pa + 3:pa + 7] = K.mat_to_quat(Rr)
        # arm: ready pose + noise, fingers open
        q = self.q_ready + _u(-cfg.start_q_noise, cfg.start_q_noise, (N, 6), gen, dev)
        self.t_qpos[:, self.arm_q] = q
        self.t_ctrl[:, self.arm_act] = q
        for a in lay.finger_qadr:
            self.t_qpos[:, int(a)] = 0.035
        self.q_hist = q[None].repeat(self.max_lat + 1, 1, 1)       # executed-command delay line
        self._randomize_phys()
        self.grip_cmd = torch.ones(N, device=dev)
        self._set_grip(self.grip_cmd)
        if cfg.rand:
            self._randomize_render()
        self.steps = 0

    def _randomize_phys(self):
        ph, gen, dev, N, f, f0, lay = self.cfg.phys, self.gen, self.device, self.N, self.f, self.f0, self.lay
        lu = lambda lo, hi, shape: torch.exp(_u(math.log(lo), math.log(hi), shape, gen, dev))  # noqa: E731
        for k in ("geom_friction", "actuator_gainprm", "actuator_biasprm", "dof_frictionloss"):
            f[k][:] = f0[k]
        if "friction" in ph:
            table = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, "table")
            for geoms in (list(lay.peg_slots), list(lay.pad_geoms), [table]):
                g = torch.as_tensor(geoms, device=dev)
                f["geom_friction"][:, g, 0] = f0["geom_friction"][g, 0] * lu(*ph["friction"], (N, 1))
        self.grip_close = (_u(*ph["grip_close"], N, gen, dev) if "grip_close" in ph
                           else torch.full((N,), self.cfg.grip_close, device=dev))
        act = self.arm_act
        if "kp_scale" in ph:
            kp = f0["actuator_gainprm"][act, 0] * lu(*ph["kp_scale"], (N, 1))
            f["actuator_gainprm"][:, act, 0] = kp
            f["actuator_biasprm"][:, act, 1] = -kp
        if "kd_scale" in ph:
            f["actuator_biasprm"][:, act, 2] = f0["actuator_biasprm"][act, 2] * lu(*ph["kd_scale"], (N, 1))
        if "frictionloss" in ph:
            dofs = torch.as_tensor(lay.arm_dofadr, device=dev)
            bounds = torch.as_tensor(ph["frictionloss"],device=dev)
            if bounds.ndim == 2:
                if bounds.shape != (6,2) or not bool((bounds > 0).all()):
                    raise ValueError("frictionloss must be positive [lo, hi] bounds for six joints")
                f["dof_frictionloss"][:, dofs] = torch.exp(bounds[:,0].log() +
                    torch.rand((N,6),generator=gen,device=dev)*(bounds[:,1].log()-bounds[:,0].log()))
            else:
                f["dof_frictionloss"][:, dofs] = lu(*ph["frictionloss"], (N, len(lay.arm_dofadr)))
        levels = torch.as_tensor(ph.get("latency", (0,)), device=dev)
        self.lat = levels[torch.randint(len(levels), (N,), generator=gen, device=dev)]

    def _yaw(self, a):
        z, o = torch.zeros_like(a), torch.ones_like(a)
        return torch.stack([a.cos(), -a.sin(), z, a.sin(), a.cos(), z, z, z, o], 1).reshape(-1, 3, 3)

    def _set_grip(self, cmd):
        f = torch.where(cmd > 0, torch.full_like(cmd, self.cfg.grip_open), self.grip_close)
        for a in self.lay.grip_act:
            self.t_ctrl[:, int(a)] = f

    def _randomize_render(self):
        r, gen, dev, N, f, f0 = self.cfg.rand, self.gen, self.device, self.N, self.f, self.f0
        c = self.top_cam
        if r.get("cam_pos") or r.get("cam_rot_deg"):
            f["cam_pos"][:, c] = f0["cam_pos"][c] + _u(-r.get("cam_pos", 0), r.get("cam_pos", 0), (N, 3), gen, dev)
            rpy = _u(-r.get("cam_rot_deg", 0), r.get("cam_rot_deg", 0), (N, 3), gen, dev) * math.pi / 180
            f["cam_quat"][:, c] = K.mat_to_quat(K.quat_to_mat(f0["cam_quat"][c].expand(N, 4)) @ K.rpy_mat(rpy))
        if (r.get("wrist_pos") or r.get("wrist_rot_deg")) and self.wrist_cam >= 0:
            c = self.wrist_cam
            f["cam_pos"][:, c] = f0["cam_pos"][c] + _u(-r.get("wrist_pos", 0), r.get("wrist_pos", 0), (N, 3), gen, dev)
            rpy = _u(-r.get("wrist_rot_deg", 0), r.get("wrist_rot_deg", 0), (N, 3), gen, dev) * math.pi / 180
            f["cam_quat"][:, c] = K.mat_to_quat(K.quat_to_mat(f0["cam_quat"][c].expand(N, 4)) @ K.rpy_mat(rpy))
        if r.get("light"):
            nl = self.mjm.nlight
            f["light_pos"][:] = f0["light_pos"] + _u(-0.4, 0.4, (N, nl, 3), gen, dev)
            dirs = f0["light_dir"] + _u(-0.3, 0.3, (N, nl, 3), gen, dev)
            f["light_dir"][:] = dirs / dirs.norm(dim=-1, keepdim=True)
            f["light_diffuse"][:] = f0["light_diffuse"] * _u(0.4, 1.6, (N, nl, 1), gen, dev)
        if r.get("colors"):
            m, lay = self.mjm, self.lay
            table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
            case = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"fixbox_{j}") for j in range(5)]
            case = [g for g in case if g >= 0] + list(lay.fixture_slots)
            case += [m.geom(n).id for n in ("detail_pc_body", "detail_pc_lid") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0]
            f["geom_rgba"][:] = f0["geom_rgba"]
            f["geom_rgba"][:, table, :3] = (f0["geom_rgba"][table, :3] * _u(0.5, 1.8, (N, 1), gen, dev)
                                            * _u(0.85, 1.15, (N, 3), gen, dev)).clamp(max=1)
            grey = _u(0.45, 0.9, (N, 1), gen, dev)
            f["geom_rgba"][:, torch.as_tensor(case, device=dev), :3] = (grey + _u(-0.04, 0.04, (N, 3), gen, dev))[:, None]
            body = [int(s) for s in lay.peg_slots if m.geom_rgba[s, 0] > 0.8]   # reader body (light grey)
            silver = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "detail_reader_silver")
            if silver >= 0:
                body.append(silver)
            f["geom_rgba"][:, torch.as_tensor(body, device=dev), :3] = _u(0.6, 0.95, (N, 1), gen, dev)[:, None].expand(N, len(body), 3)
            room = [i for i in range(m.ngeom) if m.geom(i).name in ("detail_wall", "detail_raceway") or
                    m.geom(i).name.startswith("detail_room_wall")]
            room = torch.as_tensor(room, device=dev)                 # wall paint and brightness
            f["geom_rgba"][:, room, :3] = (f0["geom_rgba"][room, :3] * _u(0.75, 1.2, (N, 1, 1), gen, dev)
                                           * _u(0.95, 1.05, (N, 1, 3), gen, dev)).clamp(max=1)

    # ------------------------------------------------------------------ physics
    def _physics(self, launch=True):
        if self._graph is None:
            with wp.ScopedDevice(self.cfg.device):
                for _ in range(2):
                    mjw.step(self.m, self.d)
                with wp.ScopedCapture() as cap:
                    for _ in range(self.cfg.substeps):
                        mjw.step(self.m, self.d)
                self._graph = cap.graph
        if launch:
            wp.capture_launch(self._graph)

    def step(self, action: torch.Tensor, noise: torch.Tensor | None = None):
        """action (N, 7): joint targets + gripper command; noise (N, 6) is added to the executed
        joint targets only (DART-style perturbation; the label stays the clean action)."""
        q_t = action[:, :6] + (noise if noise is not None else 0)
        if self.max_lat > 0:                          # world n executes the command from lat[n] steps ago
            self.q_hist = torch.roll(self.q_hist, 1, 0)
            self.q_hist[0] = q_t
            q_t = self.q_hist[self.lat, torch.arange(self.N, device=self.device)]
        self.t_ctrl[:, self.arm_act] = q_t
        self.grip_cmd = action[:, 6]
        self._set_grip(self.grip_cmd)
        self._physics()
        self.steps += 1

    # ------------------------------------------------------------------ observations
    def state(self):
        """(N, 7): arm joints (with encoder noise, PickCfg.phys q_noise) and the mean finger opening (m)."""
        q = self.t_qpos[:, self.arm_q]
        if self.cfg.phys.get("q_noise"):
            q = q + self.cfg.phys["q_noise"] * torch.randn(q.shape, generator=self.gen, device=self.device)
        fq = self.t_qpos[:, torch.as_tensor(self.lay.finger_qadr, device=self.device)].mean(1, keepdim=True)
        return torch.cat([q, fq], 1)

    def render(self) -> dict:
        """{camera: (N, H, W, 3) uint8} for every camera in cfg.cams."""
        with wp.ScopedDevice(self.cfg.device):
            mjw.kinematics(self.m, self.d)
            mjw.camlight(self.m, self.d)
            mjw.refit_bvh(self.m, self.d, self.rc)
            mjw.render(self.m, self.d, self.rc)
            out = {}
            for name, idx in zip(self.cfg.cams, self.rc_index):
                mjw.get_rgb(self.rc, idx, self._rgb)
                out[name] = (wp.to_torch(self._rgb).clamp(0, 1) * 255).to(torch.uint8)
        return out

    def reader_pose(self):
        pa = self.lay.peg_qadr
        return self.t_qpos[:, pa:pa + 3], K.quat_to_mat(self.t_qpos[:, pa + 3:pa + 7])

    def in_hand(self, tol=0.03):
        """Reader body centre within `tol` of the tcp (the grasp is on the body)."""
        p_r, R_r = self.reader_pose()
        c = p_r + (R_r @ torch.tensor([0.0, 0.0, self.body_z], device=self.device))
        return (c - self.tcp_pose()[0]).norm(dim=1) < tol

    def tcp_pose(self):
        return self.chain.fk(self.t_qpos[:, self.arm_q])

    def plug_in_port(self):
        """Reader pose in the port frame: plug-tip centre p (m) and rotation R (identity =
        aligned for insertion; p_z = gap in front of the panel)."""
        p, R = self.reader_pose()
        Rt = self.port_R.transpose(1, 2)
        return (Rt @ (p - self.port_p)[..., None])[..., 0], Rt @ R


class Expert:
    """Privileged scripted pick: approach, descend, close, lift, carry, line the plug up with the
    port at a random gap, hold. Cartesian waypoints, batched IK from the commanded joints."""
    PHASES = ("approach", "descend", "close", "lift", "carry", "align", "hold")

    def __init__(self, env: PickEnv, v_max=0.12, w_max=1.2, gap=(0.003, 0.010), grasp_y=None,
                 grasp_along=0.004, lateral=0.0005, close_steps=40, hold_steps=15, seed=0,
                 bias_gain=0.1, bias_max=0.02, near_r=0.02):
        self.env, self.v, self.w = env, v_max, w_max
        if grasp_y is None:     # middle of the insertion policy's in-hand range for this gripper
            grasp_y = float(np.mean(S.GRASP_Y[env.gripper]))
        self.gap, self.gy, self.ga, self.lat = gap, grasp_y, grasp_along, lateral
        self.close_steps, self.hold_steps = close_steps, hold_steps
        # integral correction at the port: joint friction (up to ~3 N m on joint 3, PickCfg.phys) holds
        # the arm up to ~2 cm short of its command, so the bias must reach that far
        self.bias_gain, self.bias_max, self.near_r = bias_gain, bias_max, near_r
        self.gen = torch.Generator(device=env.device)
        self.gen.manual_seed(seed + 17)

    def reset(self):
        env, N, dev, g = self.env, self.env.N, self.env.device, self.gen
        self.phase = torch.zeros(N, dtype=torch.long, device=dev)
        self.timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.q_cmd = env.t_qpos[:, env.arm_q].clone()
        self.p_cmd, self.R_cmd = env.chain.fk(self.q_cmd)
        self.grip = torch.ones(N, device=dev)
        self.g_off = torch.stack([torch.zeros(N, device=dev), torch.full((N,), self.gy, device=dev),
                                  self.env.body_z + _u(-self.ga, self.ga, N, g, dev)], 1)   # tcp in part frame
        self.gap_t = _u(*self.gap, N, g, dev)
        self.lat_t = _u(-self.lat, self.lat, (N, 2), g, dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)
        self.bias = torch.zeros(N, 3, device=dev)       # integral correction of servo sag (align)

    def _grasp_pose(self):
        p, R = self.env.reader_pose()
        return p + (R @ self.g_off[..., None])[..., 0], R @ self.env.R_top

    def _align_pose(self):
        """tcp pose that puts the plug tip gap_t in front of the port, using the in-hand pose
        measured now (privileged)."""
        env = self.env
        p_r, R_r = env.reader_pose()
        p_t, R_t = env.tcp_pose()
        R_tr = R_t.transpose(1, 2) @ R_r                      # reader in tcp
        p_tr = (R_t.transpose(1, 2) @ (p_r - p_t)[..., None])[..., 0]
        tgt = torch.cat([self.lat_t, self.gap_t[:, None]], 1)  # plug tip in the port frame
        R_rw = env.port_R                                     # aligned reader = port frame
        p_rw = env.port_p + (env.port_R @ tgt[..., None])[..., 0]
        R_tw = R_rw @ R_tr.transpose(1, 2)
        p_tw = p_rw - (R_tw @ p_tr[..., None])[..., 0]
        return p_tw, R_tw

    def act(self):
        env, ph = self.env, self.phase
        up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
        gp, gR = self._grasp_pose()
        p_wp, R_wp = self.p_cmd.clone(), self.R_cmd.clone()
        m = ph == 0
        p_wp[m], R_wp[m] = gp[m] + 0.08 * up, gR[m]
        m = ph == 1
        p_wp[m], R_wp[m] = gp[m], gR[m]
        m = ph == 3
        p_wp[m] = self.p_lift[m] if hasattr(self, "p_lift") else p_wp[m]
        if (ph >= 4).any():
            ap, aR = self._align_pose()
            m = ph == 4
            p_wp[m], R_wp[m] = ap[m] + 0.03 * up + 0.02 * env.port_R[m][:, :, 2], aR[m]   # above and behind
            m = ph >= 5
            p_wp[m], R_wp[m] = ap[m] + self.bias[m], aR[m]
        # move the command toward the waypoint (speed limits), IK from the commanded joints
        dp = p_wp - self.p_cmd
        dist = dp.norm(dim=1, keepdim=True)
        v = torch.where(ph[:, None] == 5, torch.full_like(dist, 0.04), torch.full_like(dist, self.v))
        self.p_cmd = self.p_cmd + dp * torch.clamp(v * env.control_dt / dist.clamp(min=1e-9), max=1.0)
        w = K.rot_log(R_wp @ self.R_cmd.transpose(1, 2))
        ang = w.norm(dim=1, keepdim=True)
        w = w * torch.clamp(self.w * env.control_dt / ang.clamp(min=1e-9), max=1.0)
        self.R_cmd = K.axis_angle_mat(w / w.norm(dim=1, keepdim=True).clamp(min=1e-9), w.norm(dim=1)) @ self.R_cmd
        self.q_cmd, _, _ = env.chain.ik(self.p_cmd, self.R_cmd, self.q_cmd, iters=6, q_rest=env.home)
        reached = (dist[:, 0] < 5e-4) & (ang[:, 0] < 0.01)
        p_a, R_a = env.tcp_pose()                               # the arm lags its command
        tgt = p_wp - torch.where(ph[:, None] >= 5, self.bias, torch.zeros_like(self.bias))
        settled = ((tgt - p_a).norm(dim=1) < 1.5e-3) & (K.rot_log(R_wp @ R_a.transpose(1, 2)).norm(dim=1) < 0.03)
        near = (ph >= 5) & ((tgt - p_a).norm(dim=1) < self.near_r)
        self.bias = torch.where(near[:, None], (self.bias + self.bias_gain * (tgt - p_a)).clamp(-self.bias_max, self.bias_max),
                                self.bias)
        # phase logic
        self.timer += 1
        adv = torch.zeros_like(reached)
        adv |= (ph == 0) & reached
        adv |= (ph == 1) & reached & settled
        adv |= (ph == 2) & (self.timer >= self.close_steps)
        adv |= (ph == 3) & reached & (self.timer > 5)
        adv |= (ph == 4) & reached
        adv |= (ph == 5) & reached & settled & ((tgt - p_a).norm(dim=1) < 3e-4)
        adv |= (ph == 6) & (self.timer >= self.hold_steps) & ~self.done
        if ((ph == 2) & adv).any():
            m = (ph == 2) & adv
            if not hasattr(self, "p_lift"):
                self.p_lift = self.p_cmd.clone()
            self.p_lift[m] = self.p_cmd[m] + 0.07 * up
        self.done |= (ph == 6) & adv
        self.phase = torch.where(adv & (ph < 6), ph + 1, ph)
        self.timer = torch.where(adv, torch.zeros_like(self.timer), self.timer)
        self.grip = torch.where(self.phase >= 2, -torch.ones_like(self.grip), torch.ones_like(self.grip))
        return torch.cat([self.q_cmd, self.grip[:, None]], 1), self.done.clone()

    def result(self):
        """Per world: reached the hold (done), plug tip lateral error and gap (mm), tilt (deg),
        and whether the reader is still in the hand."""
        env = self.env
        p, R = env.plug_in_port()
        lat = (p[:, :2] - self.lat_t).norm(dim=1) * 1e3
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        held = env.in_hand()
        ok = self.done & held & (lat < 1.0) & (tilt < 3.0)
        return dict(done=self.done, ok=ok, lat_mm=lat, gap_mm=p[:, 2] * 1e3, tilt_deg=tilt, held=held)
