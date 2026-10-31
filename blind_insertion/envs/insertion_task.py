"""Part 4: right-arm USB-C insertion episodes on top of the bimanual runtime.

Each episode starts with the SD reader already in the right hand (provisional top-down side-face grasp, see
``inhand_pose``) a few mm in front of PC port 2, with randomized pre-insert error, in-hand
pose error and PC pose. The left arm holds the PC by pressing on its top (scripted; Part 3
owns the real hold design). The policy commands the right arm's six joints only, through the
same 14-D accumulated-target interface as ``BimanualTask`` (gripper closed, left arm scripted).

Observations (actor, per step): the ``BimanualTask`` proprioception (12 encoder angles,
12 velocities, 14 previous actions, 12 target errors) plus the right TCP pose and twist from
forward kinematics of the believed joint angles (no object state). The critic additionally sees
the reader pose in the live port frame, contact labels of 64 plug-edge points, PC motion, the
in-hand pose and the randomized parameters.

Starts are precomputed in a pool (GPU IK) once per PPO iteration, as in ``PegEnv``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.envs.bimanual_task import BimanualTask, BimanualTaskCfg

ROOT = Path(__file__).resolve().parents[2]
FREE, CONTACT, INSIDE = 0, 1, 2

# Confirmed grasp (user, 2026-10-31; POLICY_PLAN Part 1 on policy/v1 6725f82; the pick agent owns
# the final in-hand distribution): the gripper reaches forward tipped 60 deg from vertical (tool
# axis 30 deg below horizontal, tipped about the finger-closing axis toward the reader); the fingers
# close horizontally on the body's two 8.8 mm side faces near the front of the body, behind the neck;
# the reader stays flat; the plug points across the closing direction. Reader frame: z = plug axis
# from the tip back to the body, y = body thickness (up when flat), x = width. With the reader at
# yaw 0 (plug along world +x): site y (opening) = world y = reader -x, site z = (sin60, 0, -cos60)
# world = reader (0, -cos60, -sin60), site x = y cross z = reader (0, -sin60, cos60).
TIP_DEG = 60.0
_c, _s = math.cos(math.radians(TIP_DEG)), math.sin(math.radians(TIP_DEG))
GRASP_R = ((0., -1., 0.), (-_s, 0., -_c), (_c, 0., -_s))   # rows: reader x, y, z; columns: tcp x, y, z
# TCP (grasp_site) position on the reader: centred across the width; along the plug axis so that the
# pad centroid (9.6 mm behind the site along -z) sits 8 mm behind the body front (reader z 49.2 mm);
# height solved in ``grasp_height`` so the lowest finger point is 1 mm below the reader's bottom plane
# (fingertips 1 mm above the table at the pick). GRASP_PAD_Z is the pad-centroid reader z.
GRASP_PAD_Z = 0.0572
GRASP_TIP_CLEARANCE = 0.001


@dataclass
class InsertionCfg:
    task: BimanualTaskCfg = field(default_factory=lambda: BimanualTaskCfg(
        episode_s=6., physics=BimanualCfg(contact_dr=dict(DEFAULT_CONTACT_DR), opt_overrides={"cone": 1})))
    history: int = 30
    # Pre-insert error of the plug relative to the port (port frame): lateral half-ranges (m) for
    # the tip across the aperture (x: long side, y: short side), gap of the tip in front of the
    # panel (m), and rotation half-range (deg, each axis).
    lateral_m: tuple = (0.002, 0.002)
    gap_m: tuple = (0.002, 0.008)
    angle_deg: float = 2.0
    # In-hand pose error around the nominal top-down grasp (TCP frame): translation half-ranges (m)
    # for tcp x (along the plug axis), y (closing axis) and z (vertical grasp height); rotation
    # half-ranges (deg) about tcp x, y (closing axis: reader pitch in the pad plane) and z. Flat
    # pads on flat faces fix the out-of-plane DOFs, so y and the x/z rotations are small.
    # The two fingers are joint-coupled (equality), so the reader is always centred between the
    # pads: the closing-axis offset is zero by construction (the y entry is ignored).
    inhand_pos_m: tuple = (0.003, 0.0, 0.003)
    inhand_rot_deg: tuple = (1.5, 8.0, 1.5)
    # Left arm: press on the PC top at the reachability search's pose (pads `press_m` below the
    # surface as a target; the episode starts with the pads 3 mm above it).
    left_hold: str = 'press'
    press_m: float = 0.006
    warmup_steps: int = 10              # actions ignored while the grasp and the press settle
    # termination (failure) thresholds
    far_m: float = 0.02                 # tip further than this from the port centre
    slip_m: float = 0.004               # reader moved in the hand since the end of warmup
    slip_deg: float = 6.0
    pc_move_m: float = 0.01             # PC translation since reset
    pc_tilt_deg: float = 5.0
    # reward: sparse capture + seated (Bronars et al.) + potential shaping on tip error
    shaping: float = 1.0
    shaping_scale_m: float = 0.01
    free_height: float = 0.001          # contact labels: point above the panel by this is free
    pool_per_world: int = 4
    grasp_p: tuple | None = None        # TCP in the reader frame; None: solved by InsertionTask.grasp_height


DEFAULT_CONTACT_DR = dict(port_friction=(0.15, 0.6), port_timeconst=(0.003, 0.008),
                          pad_friction=(0.8, 2.0), pad_timeconst=(0.002, 0.006),
                          grip_gain_scale=(0.6, 1.4), reader_friction=0.5)


def _rpy(x):
    return K.rpy_mat(x)


def inhand_pose(n, cfg: InsertionCfg, device, gen=None):
    """Reader pose in the right TCP frame for the top-down side-face grasp: (R_rt, p_rt), x_tcp = R_rt x_reader + p_rt.

    Nominal geometric grasp plus uniform error (cfg.inhand_pos_m, cfg.inhand_rot_deg). Replace this
    function (or its nominal) with the pick policy's measured grasp distribution when it exists.
    """
    R_pt = torch.tensor(GRASP_R, device=device)            # tcp axes in the reader frame
    p_pt = torch.tensor(cfg.grasp_p if cfg.grasp_p is not None else (0., -0.0015, 0.049), device=device)
    R_rt0, p_rt0 = R_pt.T, -(R_pt.T @ p_pt)
    u = lambda *s: torch.rand(*s, device=device, generator=gen) * 2 - 1
    rot = _rpy(u(n, 3) * torch.tensor(cfg.inhand_rot_deg, device=device) * math.pi / 180)
    dp = u(n, 3) * torch.tensor(cfg.inhand_pos_m, device=device)
    c = torch.tensor([0., 0., -0.0096], device=device)          # pad centroid (site frame): rotation centre
    return rot @ R_rt0, (rot @ (p_rt0 - c)[:, None])[..., 0] + c + dp


class InsertionTask(BimanualTask):
    """Batched insertion episodes with auto-reset. ``step`` takes the policy's 6 right-arm actions."""

    act_dim = 6

    def __init__(self, cfg: InsertionCfg | None = None, seed: int = 0):
        self.icfg = cfg or InsertionCfg()
        super().__init__(self.icfg.task)
        rt, dev = self.runtime, self.device
        m = rt.mjm
        self.N = rt.cfg.nworld
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        if rt.layout is None:
            raise ValueError('insertion starts need the task layout (PC pose distribution)')
        lay = rt.lay
        self.right_chain = K.Chain(m, lay.tcp_site, np.array(rt.indices['right'][0]), device=dev)
        self.left_chain = K.Chain(m, m.site('left_grasp_site').id, np.array(rt.indices['left'][0]), device=dev)
        self.finger_q = [int(m.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
        self.left_finger_q = [int(m.joint('left_' + f + '_finger').qposadr[0]) for f in ('left', 'right')]
        self.peg_dadr = int(m.joint(m.body('peg').jntadr[0]).dofadr[0]) if m.body('peg').jntnum[0] else None
        self.pc_dadr = int(m.joint('pc_free_joint').dofadr[0])
        v = lay.variants[0]
        self.body_half_width = float(v.peg_boxes[0][1][0])
        self.seat_depth = float((lay.seating_depth if lay.seating_depth is not None else lay.depth)[0])
        xy = torch.as_tensor(np.asarray(lay.edge_points[0]), device=dev, dtype=torch.float32)
        self.edge = torch.cat([xy, torch.zeros(len(xy), 1, device=dev)], 1)        # (64, 3) reader frame
        self.n_labels = len(xy)
        self.hole = torch.as_tensor(np.asarray(lay.hole_polys[0]), device=dev, dtype=torch.float32)
        table = m.geom('table')
        self.table_top = float(table.pos[2] + table.size[2])
        rep = json.loads((ROOT / rt.cfg.task_layout).read_text())['nominal']
        self.q_right_report = torch.tensor(rep['q_right'], device=dev)
        self.q_left_report = torch.tensor(rep['q_left'], device=dev)
        if self.icfg.grasp_p is None:
            self.icfg.grasp_p = self.grasp_height()
        self._init_nominal()
        # buffers
        n = self.N
        self.hist = torch.zeros(n, self.icfg.history, self.obs_dim, device=dev)
        self.start = {}
        self.ref_rel = torch.zeros(n, 4, 4, device=dev)       # reader in TCP at end of warmup
        self.pc_start = torch.zeros(n, 7, device=dev)
        self.captured = torch.zeros(n, dtype=torch.bool, device=dev)
        self.phi = torch.zeros(n, device=dev)
        self.left_press_q = torch.zeros(n, 6, device=dev)
        self.overflow_counts = torch.zeros(16, dtype=torch.long, device=dev)
        self.refresh_pool()
        self.reset_all()

    # ------------------------------------------------------------------ nominal poses
    def grasp_height(self):
        """TCP position in the reader frame for the confirmed grasp (CPU geometry, fingers at the body width).

        Collects the right fingers' collision geometry (pad spheres, tip hulls) in the site frame with the
        fingers closed on the 34.7 mm body, then puts the lowest point GRASP_TIP_CLEARANCE below the
        reader's bottom plane. Also records the pad coverage of the side face (``self.grasp_info``).
        """
        import mujoco
        rt = self.runtime
        m = rt.mjm
        d = mujoco.MjData(m)
        d.qpos[:] = rt.initial_q.cpu().numpy()
        for j in self.finger_q:
            d.qpos[j] = self.body_half_width + 0.0012
        mujoco.mj_forward(m, d)
        site = rt.lay.tcp_site
        Rs, ps = d.site_xmat[site].reshape(3, 3), d.site_xpos[site]
        bodies = {m.body(n).id for n in ('left_finger', 'right_finger')}
        pts, pads = [], []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in bodies or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            Rg, pg = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                w = v @ Rg.T + pg
            else:
                r = float(m.geom_rbound[g])
                w = pg + r * np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
            loc = (w - ps) @ Rs                                    # site frame
            pts.append(loc)
            if m.geom(g).name.startswith(('left_pad_s', 'right_pad_s')):
                pads.append((pg - ps) @ Rs)
        pts, pads = np.concatenate(pts), np.array(pads)
        R = np.array(GRASP_R)                                      # site axes in the reader frame
        rad = 0.0012
        # pad centroid along -z (site frame) mapped to the reader: choose the site so the centroid
        # lies at reader z = GRASP_PAD_Z, x = 0; y from the clearance of the lowest finger point
        pc_site = pads.mean(0)
        low = (pts @ R.T)[:, 1].min()                              # lowest finger point: reader y offset from the site
        bottom = -float(self.runtime.lay.variants[0].peg_boxes[0][1][1])
        y = bottom - GRASP_TIP_CLEARANCE - low
        z = GRASP_PAD_Z - (R @ pc_site)[2]
        pad_y = (pads @ R.T)[:, 1] + y
        top = -bottom
        cover = (min(pad_y.max() + rad, top) - max(pad_y.min() - rad, bottom))
        self.grasp_info = dict(tcp_in_reader_m=[0.0, float(y), float(z)], pad_y_range_m=[float(pad_y.min()), float(pad_y.max())],
                               side_face_coverage_m=float(max(cover, 0.)), lowest_finger_below_reader_m=GRASP_TIP_CLEARANCE)
        return (0.0, float(y), float(z))


    def _port_pose(self, pc_qpos):
        """Port frame (= PC fixture body frame) in world from the PC free-joint qpos."""
        return pc_qpos[:, :3], K.quat_to_mat(pc_qpos[:, 3:7])

    def _init_nominal(self):
        """Nominal grasp arm configuration and the left press pose in the PC frame."""
        rt, dev = self.runtime, self.device
        pc0 = self._pc_qpos(torch.zeros(1, device=dev), torch.zeros(1, device=dev), torch.zeros(1, device=dev))
        port_p, port_R = self._port_pose(pc0)
        R_rt, p_rt = inhand_pose(1, InsertionCfg(inhand_pos_m=(0, 0, 0), inhand_rot_deg=(0, 0, 0), grasp_p=self.icfg.grasp_p), dev)
        p_tip = port_p + (port_R @ torch.tensor([0., 0., .005], device=dev)[:, None])[..., 0]
        R_tw = port_R @ R_rt.transpose(1, 2)
        p_tw = p_tip - (R_tw @ p_rt[..., None])[..., 0]
        lo, hi = self.right_chain.lo, self.right_chain.hi
        seeds = [self.q_right_report, rt.initial_q[rt.indices['right'][0]]]
        g = torch.Generator(device=dev); g.manual_seed(1)
        seeds += [lo + (hi - lo) * torch.rand(6, device=dev, generator=g) for _ in range(254)]
        Q0 = torch.stack(seeds)
        q, pe, re = self.right_chain.ik(p_tw.expand(len(Q0), 3), R_tw.expand(len(Q0), 3, 3), Q0, iters=120)
        ok = (pe < 1e-4) & (re < 1e-3)
        if not ok.any():
            raise RuntimeError('no IK solution for the nominal pre-insert pose')
        margin = torch.minimum(q - lo, hi - q).min(1).values
        margin[~ok] = -1
        # prefer the solution closest to the report's (top-grasp) arm posture among good margins
        score = margin - 0.05 * (q - self.q_right_report).norm(dim=1)
        self.q_right_nominal = q[score.argmax()].clone()
        self.nominal_margin = float(margin[score.argmax()])
        # left press: report q_left (pads 3 mm above the PC top at the nominal PC pose) in the PC frame
        lp, lR = self.left_chain.fk(self.q_left_report[None])
        self.left_in_pc = (port_R.transpose(1, 2) @ (lp - port_p)[..., None])[..., 0], port_R.transpose(1, 2) @ lR

    def _pc_qpos(self, dx, dy, dyaw_deg):
        """PC free-joint qpos at nominal port + (dx, dy), yaw nominal + dyaw (runtime._sample_layout math)."""
        rt = self.runtime
        (px, py), yaw = rt.layout['port_xy'], rt.layout['yaw']
        x, y = px + dx, py + dy
        a = torch.deg2rad(yaw + dyaw_deg)
        c, s_ = torch.cos(a), torch.sin(a)
        ref = rt._pc0
        p0 = ref[:3]
        pos = torch.stack([x + c * p0[0] - s_ * p0[1], y + s_ * p0[0] + c * p0[1], p0[2].expand(len(x))], -1)
        qz = torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)
        quat = _qmul(qz, ref[3:].expand(len(x), 4))
        return torch.cat([pos, quat], 1)

    # ------------------------------------------------------------------ start pool
    def sample_starts(self, n, gen=None):
        """Start states (GPU IK); worlds whose IK fails are resampled up to 10 times."""
        c, dev, rt = self.icfg, self.device, self.runtime
        gen = gen or self.gen
        u = lambda *s: torch.rand(*s, device=dev, generator=gen) * 2 - 1
        out = None
        todo = torch.arange(n, device=dev)
        for _ in range(10):
            k = len(todo)
            if k == 0:
                break
            dx, dy, dyaw = rt.cfg.pc_pose_range
            pc = self._pc_qpos(u(k) * dx, u(k) * dy, u(k) * dyaw)
            port_p, port_R = self._port_pose(pc)
            # plug relative to the port
            lat = u(k, 2) * torch.tensor(c.lateral_m, device=dev)
            gap = c.gap_m[0] + (c.gap_m[1] - c.gap_m[0]) * (u(k) + 1) / 2
            ang = u(k, 3) * c.angle_deg * math.pi / 180
            R_rp = _rpy(ang)
            p_rp = torch.cat([lat, gap[:, None]], 1)
            R_rw = port_R @ R_rp
            p_rw = port_p + (port_R @ p_rp[..., None])[..., 0]
            R_rt, p_rt = inhand_pose(k, c, dev, gen)
            # centre the reader's mid-plane (across the width) between the coupled pads
            pad = torch.tensor([0., 0., -0.0096], device=dev)
            x_r = (R_rt.transpose(1, 2) @ (pad - p_rt)[..., None])[..., 0]
            x_r[:, 0] = 0
            cy = ((R_rt @ x_r[..., None])[..., 0] + p_rt)[:, 1]
            p_rt = p_rt - torch.stack([torch.zeros_like(cy), cy, torch.zeros_like(cy)], 1)
            R_tw = R_rw @ R_rt.transpose(1, 2)
            p_tw = p_rw - (R_tw @ p_rt[..., None])[..., 0]
            q, pe, re = self.right_chain.ik(p_tw, R_tw, self.q_right_nominal.expand(k, 6), iters=40)
            # left press pose follows the PC
            lp0, lR0 = self.left_in_pc
            l_R = port_R @ lR0
            l_hover = port_p + (port_R @ lp0[..., None].expand(k, 3, 1))[..., 0]
            down = torch.tensor([0., 0., -1.], device=dev)
            ql_h, pel, rel = self.left_chain.ik(l_hover, l_R, self.q_left_report.expand(k, 6), iters=30)
            ql_p, pep, rep = self.left_chain.ik(l_hover + down * (0.003 + c.press_m), l_R, ql_h, iters=20)
            ok = (pe < 2e-4) & (re < 2e-3) & (pel < 5e-4) & (rel < 5e-3) & (pep < 5e-4) & (rep < 5e-3)
            # fingers: pad surfaces on the reader's side faces (pad sphere radius 1.2 mm)
            h = self.body_half_width + 0.0012 + 3e-4     # 0.3 mm open: the motors close onto the faces
            fingers = torch.full((k, 2), h, device=dev)
            reader = torch.cat([p_rw, K.mat_to_quat(R_rw)], 1)
            res = dict(pc=pc, q_right=q, reader=reader, fingers=fingers, q_left_hover=ql_h, q_left_press=ql_p,
                       inhand_R=R_rt, inhand_p=p_rt, err_p=p_rp, err_ang=ang)
            if out is None:
                out = {key: torch.zeros(n, *v.shape[1:], device=dev) for key, v in res.items()}
                out['ok'] = torch.zeros(n, dtype=torch.bool, device=dev)
            good = todo[ok]
            for key, v in res.items():
                out[key][good] = v[ok]
            out['ok'][good] = True
            todo = todo[~ok]
        self.ik_failures = len(todo)
        if len(todo):     # keep the last sample (IK residual is reported); rare
            for key, v in res.items():
                out[key][todo] = v[~ok]
        return out

    def refresh_pool(self):
        P = self.icfg.pool_per_world
        self.pool = self.sample_starts(self.N * P)
        self.pool_ptr = torch.zeros(self.N, dtype=torch.long, device=self.device)

    def _take(self, ids):
        P = self.icfg.pool_per_world
        ptr = self.pool_ptr[ids] % P
        self.pool_ptr[ids] += 1
        j = ids * P + ptr
        return {k: v[j] for k, v in self.pool.items()}

    # ------------------------------------------------------------------ reset / step
    def reset(self, ids=None, starts=None):
        rt = self.runtime
        if ids is None:
            ids = torch.arange(self.N, device=self.device)
        if len(ids) == 0:
            return
        st = starts if starts is not None else self._take(ids)
        rt.reset(ids)      # dynamics, PC feet friction, contact DR, layout (overwritten below)
        q = rt.q
        qr, ql = rt.indices['right'][0], rt.indices['left'][0]
        q[ids[:, None], torch.tensor(qr, device=self.device)] = st['q_right']
        q[ids[:, None], torch.tensor(ql, device=self.device)] = st['q_left_hover']
        q[ids, rt.pc_qadr:rt.pc_qadr + 7] = st['pc']
        pa = rt.lay.peg_qadr
        q[ids, pa:pa + 7] = st['reader']
        q[ids[:, None], torch.tensor(self.finger_q, device=self.device)] = st['fingers']
        q[ids[:, None], torch.tensor(self.left_finger_q, device=self.device)] = 0.
        rt.v[ids] = 0
        act_r, act_l = rt.indices['right'][2], rt.indices['left'][2]
        rt.ctrl[ids[:, None], torch.tensor(act_r, device=self.device)] = st['q_right']
        rt.ctrl[ids[:, None], torch.tensor(act_l, device=self.device)] = st['q_left_hover']
        rt.ctrl[ids[:, None], torch.tensor(rt.indices['right'][3], device=self.device)] = rt.cfg.grip_close
        rt.ctrl[ids[:, None], torch.tensor(rt.indices['left'][3], device=self.device)] = rt.cfg.grip_close
        with wp.ScopedDevice(rt.cfg.device):
            mjw.forward(rt.m, rt.d)
        self._sample_control(ids)
        self.joint_target[ids] = self._measured_model_angles()[ids]
        self.left_press_q[ids] = st['q_left_press']
        self.steps[ids] = 0; self.seated_steps[ids] = 0; self.done[ids] = False; self.previous_action[ids] = 0
        self.sent_history[:, ids] = 0
        for k, v in st.items():
            if k not in self.start:
                self.start[k] = torch.zeros(self.N, *v.shape[1:], device=self.device, dtype=v.dtype)
            self.start[k][ids] = v
        self.pc_start[ids] = st['pc']
        self.captured[ids] = False
        self.ref_rel[ids] = self._reader_in_tcp()[ids]
        o = self._actor_obs()
        self.hist[ids] = o[ids, None].expand(-1, self.icfg.history, -1)
        self.phi[ids] = self._potential()[ids]

    def reset_all(self):
        self.reset(torch.arange(self.N, device=self.device))

    def _update_joint_targets(self, action, measured):
        proposed = super()._update_joint_targets(action, measured)
        # scripted left hold: press target (believed frame), within the tracking cap
        cap = self.cfg.target_cap_rad
        lq = torch.maximum(torch.minimum(self.left_press_q, measured[:, 1] + cap), measured[:, 1] - cap)
        proposed[:, 1] = lq
        self.joint_target.copy_(proposed)
        return proposed

    def step(self, action):
        a6 = torch.as_tensor(action, device=self.device, dtype=torch.float32).clamp(-1, 1)
        warm = self.steps < self.icfg.warmup_steps
        a6 = torch.where(warm[:, None], torch.zeros_like(a6), a6)
        a = torch.zeros(self.N, 14, device=self.device)
        a[:, :6] = a6
        a[:, 6] = -1.      # right gripper closed
        a[:, 13] = -1.     # left gripper closed
        self.done[:] = False
        _, _, terminated, truncated, info = BimanualTask.step(self, a)
        return self._post_step(terminated, truncated, info, warm)

    def _transition(self):
        """Episode status (insertion version): success after the seated hold, failure classes."""
        rt, c = self.runtime, self.icfg
        self.steps += 1
        p, R = rt.plug_in_port()
        pts = p[:, None] + self.edge[None] @ R.transpose(1, 2)
        lab, inside_xy = self._labels(pts)
        metrics = rt.insertion_metrics(self.cfg.seat_tolerance_m, self.cfg.tilt_limit_deg)
        finite = metrics['finite'] & torch.isfinite(rt.v).all(1)
        seated = metrics['geometric_seated'] & finite
        self.seated_steps = torch.where(seated, self.seated_steps + 1, torch.zeros_like(self.seated_steps))
        success = (self.seated_steps >= self.hold_steps) & finite
        # failures
        rel = self._reader_in_tcp()
        at_warm = self.steps == self.icfg.warmup_steps
        self.ref_rel[at_warm] = rel[at_warm]
        d_rel = rel[:, :3, 3] - self.ref_rel[:, :3, 3]
        ang_rel = K.rot_log(rel[:, :3, :3] @ self.ref_rel[:, :3, :3].transpose(1, 2)).norm(dim=1) * 180 / math.pi
        after_warm = self.steps > self.icfg.warmup_steps
        slip = after_warm & ((d_rel.norm(dim=1) > c.slip_m) | (ang_rel > c.slip_deg))
        pc = rt.q[:, rt.pc_qadr:rt.pc_qadr + 7]
        pc_d = (pc[:, :3] - self.pc_start[:, :3]).norm(dim=1)
        pc_tilt = K.rot_log(K.quat_to_mat(pc[:, 3:7]) @ K.quat_to_mat(self.pc_start[:, 3:7]).transpose(1, 2)).norm(dim=1) * 180 / math.pi
        pushed = (pc_d > c.pc_move_m) | (pc_tilt > c.pc_tilt_deg)
        far = p.norm(dim=1) > c.far_m
        bad = ~finite
        failure = (bad | slip | pushed | far) & ~success
        terminated = success | failure
        truncated = (self.steps >= self.max_steps) & ~terminated
        self.done = terminated | truncated
        self._last = dict(p=p, R=R, lab=lab, inside=inside_xy, seated=seated, success=success, slip=slip,
                          pushed=pushed, far=far, bad=bad, pc_d=pc_d, tilt=metrics['tilt_deg'],
                          aperture=metrics['aperture_fit'], ang_rel=ang_rel, d_rel=d_rel.norm(dim=1))
        info = {'success': success.clone(), 'nonfinite': bad, 'slip': slip, 'pushed': pushed, 'far': far,
                'geometric_seated': seated, 'steps': self.steps.clone()}
        return success.float(), terminated, truncated, info

    def _post_step(self, terminated, truncated, info, warm):
        c, L = self.icfg, self._last
        all_inside = (L['lab'] == INSIDE).all(1)
        new_capture = all_inside & ~self.captured
        self.captured |= all_inside
        phi2 = self._potential()
        gamma = 0.98
        reward = new_capture.float() + L['seated'].float() + c.shaping * (gamma * phi2 - self.phi)
        reward = torch.where(L['bad'], torch.zeros_like(reward), reward)
        reward = torch.nan_to_num(reward, 0., 0., 0.)
        self.phi = phi2
        o = self._actor_obs()
        self.hist = torch.cat([self.hist[:, 1:], o[:, None]], 1)
        crit = self.critic_obs()
        done = terminated | truncated
        info.update(timeout=truncated, captured=self.captured.clone(), valid=~warm,
                    tip_p=L['p'], tilt=L['tilt'], pc_d=L['pc_d'], aperture=L['aperture'],
                    labels=L['lab'])
        ids = done.nonzero()[:, 0]
        if len(ids):
            info['final'] = {k: v[ids].clone() for k, v in L.items() if torch.is_tensor(v)}
            info['final']['captured'] = self.captured[ids].clone()
            info['final_ids'] = ids
            self.reset(ids)
            crit[ids] = self.critic_obs()[ids]
        return self.hist, crit, reward, done, info

    # ------------------------------------------------------------------ observations
    def _labels(self, pts):
        z = pts[..., 2]
        poly = self.hole
        e = (torch.roll(poly, -1, 0) - poly)[None, None]
        r = pts[..., None, :2] - poly[None, None]
        inside = ((e[..., 0] * r[..., 1] - e[..., 1] * r[..., 0]) >= -1e-9).all(-1)
        lab = torch.full(z.shape, CONTACT, dtype=torch.long, device=self.device)
        lab[z > self.icfg.free_height] = FREE
        lab[(z < 0) & inside] = INSIDE
        return lab, inside

    def current_labels(self):
        p, R = self.runtime.plug_in_port()
        return self._labels(p[:, None] + self.edge[None] @ R.transpose(1, 2))[0]

    def _potential(self):
        """-(tip distance to the seated position) / scale, capped; in the live port frame."""
        p, _ = self.runtime.plug_in_port()
        target = torch.tensor([0., 0., -self.seat_depth], device=self.device)
        d = (p - target)
        dist = (d[:, :2].norm(dim=1) * 3 + d[:, 2].abs())    # lateral weighted (precision)
        return -(dist / self.icfg.shaping_scale_m).clamp(max=5.)

    def _reader_in_tcp(self):
        rt = self.runtime
        q = rt.q[:, rt.indices['right'][0]]
        tp, tR = self.right_chain.fk(q)
        pa = rt.lay.peg_qadr
        rp, rR = rt.q[:, pa:pa + 3], K.quat_to_mat(rt.q[:, pa + 3:pa + 7])
        T = torch.zeros(self.N, 4, 4, device=self.device)
        T[:, :3, :3] = tR.transpose(1, 2) @ rR
        T[:, :3, 3] = (tR.transpose(1, 2) @ (rp - tp)[..., None])[..., 0]
        T[:, 3, 3] = 1
        return T

    @property
    def obs_dim(self):
        return 62

    def _actor_obs(self):
        o = self.observation()
        qb = self._measured_model_angles()[:, 0]
        p, R, J = self.right_chain.fk(qb, with_jac=True)
        v = o['encoder_velocity_rad_s'].reshape(-1, 2, 6)[:, 0]
        tw = (J @ v[..., None])[..., 0]
        R0 = self._R_obs
        eul = K.euler_xyz(R0.T @ R)
        pos = p - self._p_obs
        return torch.cat([o['encoder_rad'], o['encoder_velocity_rad_s'], o['previous_action'],
                          o['target_error_rad'], pos * 10, eul, tw], 1)

    @property
    def _R_obs(self):
        if not hasattr(self, '_R_obs_'):
            p, R = self.right_chain.fk(self.q_right_nominal[None])
            self._R_obs_, self._p_obs_ = R[0], p[0]
        return self._R_obs_

    @property
    def _p_obs(self):
        self._R_obs
        return self._p_obs_

    @property
    def critic_dim(self):
        return 3 + 6 + 6 + 3 * self.n_labels + 4 + 9 + 5 + 1 + 12 + 6

    def critic_obs(self):
        rt = self.runtime
        p, R = rt.plug_in_port()
        lab = self.current_labels()
        onehot = torch.nn.functional.one_hot(lab, 3).float().reshape(self.N, -1)
        rot6 = R[:, :, :2].transpose(1, 2).reshape(self.N, 6)
        pa = rt.lay.peg_qadr
        da = rt.mjm.jnt_dofadr[rt.mjm.body('peg').jntadr[0]]
        _, port_R = self._port_pose(rt.q[:, rt.pc_qadr:rt.pc_qadr + 7])
        vel = rt.v[:, da:da + 6]
        v_port = (port_R.transpose(1, 2) @ vel[:, :3, None])[..., 0]
        w_port = (port_R.transpose(1, 2) @ K.quat_to_mat(rt.q[:, pa + 3:pa + 7]) @ vel[:, 3:6, None])[..., 0]
        pc = rt.q[:, rt.pc_qadr:rt.pc_qadr + 3] - self.pc_start[:, :3]
        pc_rot = K.rot_log(port_R @ K.quat_to_mat(self.pc_start[:, 3:7]).transpose(1, 2))[:, 2:3]
        rel = self._reader_in_tcp()
        inh = torch.cat([rel[:, :3, 3] * 100, rel[:, :3, :2].transpose(1, 2).reshape(self.N, 6)], 1)
        cp = rt.contact_params if hasattr(rt, 'contact_params') else torch.ones(self.N, 5, device=self.device)
        lat = self.latency.float()[:, None]
        enc = self.enc_error.flatten(1) * 10
        err = torch.cat([self.start['err_p'] * 100, self.start['err_ang'] * 10], 1) if 'err_p' in self.start \
            else torch.zeros(self.N, 6, device=self.device)
        return torch.nan_to_num(torch.cat([p * 100, rot6, v_port * 10, w_port, onehot, pc * 100, pc_rot * 10,
                                           inh, cp, lat, enc, err], 1), 0., 0., 0.)

    def observe(self):
        return self.hist, self.critic_obs()

    # ------------------------------------------------------------------ scripted (privileged) controller
    def scripted_action(self, gain=0.5, advance_m_s=0.01, lateral_first_m=0.0003):
        """Privileged Jacobian controller (reader/port state): aligns, then advances. For checks only."""
        rt = self.runtime
        p, R = rt.plug_in_port()
        port_p, port_R = self._port_pose(rt.q[:, rt.pc_qadr:rt.pc_qadr + 7])
        target = torch.zeros_like(p)
        lat = p[:, :2].norm(dim=1)
        ok = lat < lateral_first_m
        z_goal = torch.where(ok, torch.full_like(lat, -self.seat_depth - 0.0003), p[:, 2].clamp(min=0.001))
        step_z = (z_goal - p[:, 2]).clamp(-advance_m_s * self.dt, advance_m_s * self.dt)
        target[:, 2] = p[:, 2] + step_z
        dp_port = target - p
        dp_port[:, :2] = -p[:, :2] * gain
        rot_err = -K.rot_log(R) * gain                          # rotate the reader to identity (port frame)
        dp_w = (port_R @ dp_port[..., None])[..., 0]
        w_w = (port_R @ rot_err[..., None])[..., 0]
        q = rt.q[:, rt.indices['right'][0]]
        tp, _, J = self.right_chain.fk(q, with_jac=True)
        pa = rt.lay.peg_qadr
        tip_w = rt.q[:, pa:pa + 3]
        v_tcp = dp_w + torch.cross(w_w, tp - tip_w, dim=1)
        e = torch.cat([v_tcp, w_w], 1)
        JJt = J @ J.transpose(1, 2) + 1e-6 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]
        # accumulated targets integrate the pose error (holds the arm against gravity)
        return (dq / self.cfg.action_scale_rad).clamp(-1, 1)


def _qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def failure_class(final: dict) -> list[str]:
    """Per-episode outcome from the terminal record (``info['final']``)."""
    out = []
    for i in range(len(final['success'])):
        if bool(final['success'][i]):
            out.append('seated')
        elif bool(final['bad'][i]):
            out.append('nonfinite')
        elif bool(final['slip'][i]):
            out.append('grasp_slip')
        elif bool(final['pushed'][i]):
            out.append('pc_pushed')
        elif bool(final['far'][i]) or not bool(final['captured'][i]) and float(final['p'][i, 2]) > 0:
            out.append('lateral_miss')
        elif float(final['tilt'][i]) > 2.0:
            out.append('tilt')
        else:
            out.append('jam')
    return out
