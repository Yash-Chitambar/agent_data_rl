"""Scripted left-arm hold of the PC (policy Part 3) and a scripted bimanual harness.

The left arm holds the PC still while the right arm inserts (insertion pushes the PC along
+x world = -z of the port frame, toward the wall). Two holds from the placement search:

  press: closed fingertips on the PC top (rear half), pushing down (normal force) and
         optionally toward the panel (tangential preload against insertion, through pad friction);
  brace: closed fingers hanging behind the rear face, pushing toward the panel.

The holding force comes from the PD error: the joint target is the IK solution for the contact
point plus dq = Kp^-1 J^T F (J: linear Jacobian at the contact point), so F is the static force
the hand would exert with ideal gravity compensation and no joint friction. The identified
driver-gravity residual and Coulomb friction of the runtime make the real force differ; it is
measured from the contact forces.

`Harness` steps the GPU runtime with absolute joint targets per arm in the *believed* frame and
applies the shared control randomization (encoder-offset error +/- enc_offset_deg on joints 2-5,
action latency 0-1 step), as `BimanualTask` does: absolute targets from the model land shifted
by the offset error. Object poses used by the scripts (PC pose for the hold IK, reader pose for
the privileged insertion) are privileged unless an estimate is passed in.
"""
from dataclasses import dataclass, field
import math

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from blind_insertion.envs import kinematics as K
from blind_insertion.scenes import pc_bench as S

PAD_BELOW_TCP = -.0096        # pad centroid along grasp_site +z (+z points out of the hand)
TIP_BELOW_TCP = -.0032        # fingertip end (closed fingers) along grasp_site +z


def _profile():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[2] /
                       'data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())


def pc_box(profile=None):
    """PC case box in the port (fixture) frame: x across the panel, y up, z out of the panel."""
    from blind_insertion.scenes.pc_rear import layout
    profile = profile or _profile()
    rear = layout(profile)
    v = profile['pc']['measurements']
    w, h, dep = (float(v[k]['value']) for k in ('width', 'body_height', 'depth'))
    foot = float(profile['pc']['pad_height_derived']['value'])
    port_h = rear['port_height']
    return {'width': w, 'height': h, 'depth': dep, 'mass': float(v['mass']['value']),
            'centre': np.array([w / 2 - rear['target_from_left'], -port_h + foot + h / 2, -dep / 2]),
            'top_y': -port_h + foot + h, 'bottom_y': -port_h + foot, 'rear_z': -dep}


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)


@dataclass
class HarnessCfg:
    enc_offset_deg: float = 1.5           # absolute-target error, joints 2-5, both arms
    latency_steps: tuple = (0, 1)
    # Part 1 grasp contact (bimanual_pick.PickTaskCfg): right-pad friction and task impratio.
    # None keeps the scene values (pads 1.68, impratio 100).
    pad_mu: float | None = 1.2
    impratio: float | None = 30.


class Harness:
    """Absolute joint targets (believed model frame) for both arms with control DR."""

    def __init__(self, runtime, cfg=None):
        self.rt, self.cfg = runtime, cfg or HarnessCfg()
        n, dev = runtime.cfg.nworld, runtime.device
        self.n, self.dev = n, dev
        self.enc_error = torch.zeros(n, 2, 6, device=dev)
        self.latency = torch.zeros(n, dtype=torch.long, device=dev)
        self.hist = torch.zeros(max(self.cfg.latency_steps) + 1, n, 14, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        m = runtime.mjm
        self.chain = {side: K.Chain(m, m.site(pre + 'grasp_site').id, np.array(runtime.indices[side][0]), device=dev)
                      for side, pre in (('right', ''), ('left', 'left_'))}
        self.kp = torch.tensor([[m.actuator_gainprm[a, 0] for a in runtime.indices[s][2]] for s in ('right', 'left')],
                               device=dev, dtype=torch.float32)
        self.xfrc = wp.to_torch(runtime.d.xfrc_applied)
        if self.cfg.pad_mu is not None and hasattr(runtime, 'geom_friction'):
            pads = [g for g in range(m.ngeom) if m.geom(g).name.startswith(('left_pad_s', 'right_pad_s'))]
            runtime.geom_friction[:, pads, 0] = self.cfg.pad_mu
        if self.cfg.impratio is not None:
            wp.to_torch(runtime.m.opt.impratio_invsqrt).fill_(1. / math.sqrt(self.cfg.impratio))
        self.pc_body = m.body('fixture').id
        self._robot_geoms(m)

    def _robot_geoms(self, m):
        side = np.zeros(m.ngeom, np.int64)      # 0 other, 1 right arm/gripper, 2 left arm/gripper
        for g in range(m.ngeom):
            b = m.body(int(m.geom_bodyid[g])).name
            if b.startswith(('left_link_', 'left_left_finger', 'left_right_finger', 'left_gripper', 'left_wrist')):
                side[g] = 2
            elif b.startswith(('link_', 'left_finger', 'right_finger', 'gripper', 'wrist_camera')):
                side[g] = 1
        pc = np.array([m.geom_bodyid[g] == self.pc_body for g in range(m.ngeom)])
        reader = np.array([m.geom_bodyid[g] == m.body('peg').id for g in range(m.ngeom)])
        left_finger = np.array([m.body(int(m.geom_bodyid[g])).name in ('left_left_finger', 'left_right_finger')
                                for g in range(m.ngeom)])
        self.geom_side = torch.tensor(side, device=self.dev)
        self.geom_pc = torch.tensor(pc, device=self.dev)
        self.geom_reader = torch.tensor(reader, device=self.dev)
        self.geom_left_finger = torch.tensor(left_finger, device=self.dev)
        self.geom_table = torch.tensor([m.geom(g).name == 'table' for g in range(m.ngeom)], device=self.dev)

    def reset(self, ids=None):
        ids = torch.arange(self.n, device=self.dev) if ids is None else ids
        self.rt.reset(ids)
        k = len(ids)
        opts = torch.tensor(self.cfg.latency_steps, device=self.dev)
        self.latency[ids] = opts[torch.randint(len(opts), (k,), device=self.dev)]
        err = torch.zeros(k, 2, 6, device=self.dev)
        err[:, :, 1:5] = (torch.rand(k, 2, 4, device=self.dev) * 2 - 1) * math.radians(self.cfg.enc_offset_deg)
        self.enc_error[ids] = err
        self.fresh[ids] = True
        self.xfrc[ids] = 0

    def q(self, side):
        return self.rt.q[:, self.rt.indices[side][0]]

    def believed(self, side):
        """Model-frame joint angles as the controller believes them (true - offset error)."""
        return self.q(side) - self.enc_error[:, 0 if side == 'right' else 1]

    def step(self, target_right, grip_right, target_left, grip_left):
        a = torch.empty(self.n, 14, device=self.dev)
        a[:, :6] = target_right + self.enc_error[:, 0]
        a[:, 7:13] = target_left + self.enc_error[:, 1]
        a[:, 6] = grip_right
        a[:, 13] = grip_left
        a = torch.nan_to_num(a)              # worlds that went non-finite are flagged by the caller
        self.hist[1:] = self.hist[:-1].clone(); self.hist[0] = a
        self.hist[1:, self.fresh] = a[self.fresh]
        self.fresh[:] = False
        self.rt.step(self.hist[self.latency, torch.arange(self.n, device=self.dev)])

    # --- privileged state ---------------------------------------------------------
    def pc_pose(self):
        fa = self.rt.pc_qadr
        return self.rt.q[:, fa:fa + 3].clone(), K.quat_to_mat(self.rt.q[:, fa + 3:fa + 7])

    def reader_pose(self):
        pa = self.rt.lay.peg_qadr
        return self.rt.q[:, pa:pa + 3].clone(), K.quat_to_mat(self.rt.q[:, pa + 3:pa + 7])

    def contacts(self):
        """Per-world flags/forces from the live contact buffer."""
        d = self.rt.d
        nac = int(d.nacon.numpy()[0])
        out = {'arm_arm': torch.zeros(self.n, dtype=torch.bool, device=self.dev),
               'left_other': torch.zeros(self.n, dtype=torch.bool, device=self.dev),
               'left_pc_force': torch.zeros(self.n, 3, device=self.dev),
               'right_pc': torch.zeros(self.n, dtype=torch.bool, device=self.dev)}
        if nac == 0:
            return out
        geom = wp.to_torch(d.contact.geom)[:nac].long()
        world = wp.to_torch(d.contact.worldid)[:nac].long()
        dist = wp.to_torch(d.contact.dist)[:nac]
        s1, s2 = self.geom_side[geom[:, 0]], self.geom_side[geom[:, 1]]
        touch = dist < 0
        rr = (s1 == 1) | (s2 == 1)
        ll = (s1 == 2) | (s2 == 2)
        reader = self.geom_reader[geom[:, 0]] | self.geom_reader[geom[:, 1]]
        pc = self.geom_pc[geom[:, 0]] | self.geom_pc[geom[:, 1]]
        table = self.geom_table[geom[:, 0]] | self.geom_table[geom[:, 1]]
        arm_arm = touch & ((rr & ll) | (ll & reader))
        out['arm_arm'].index_put_((world[arm_arm],), torch.ones_like(world[arm_arm], dtype=torch.bool))
        left_other = touch & ll & ~pc & ~rr & ~reader
        out['left_other'].index_put_((world[left_other],), torch.ones_like(world[left_other], dtype=torch.bool))
        right_pc = touch & (rr | reader) & pc
        out['right_pc'].index_put_((world[right_pc],), torch.ones_like(world[right_pc], dtype=torch.bool))
        lp = ll & pc
        if lp.any():
            ids = torch.nonzero(lp)[:, 0].int()
            f = wp.zeros(len(ids), dtype=wp.spatial_vector, device=self.rt.cfg.device)
            with wp.ScopedDevice(self.rt.cfg.device):
                mjw.contact_force(self.rt.m, d, wp.from_torch(ids.contiguous(), dtype=wp.int32), True, f)
            force = wp.to_torch(f)[:, :3]
            # force on geom[1] from geom[0] along the frame; make it the force the hand puts on the PC
            sign = torch.where(self.geom_pc[geom[lp, 1]], 1., -1.)[:, None]
            out['left_pc_force'].index_add_(0, world[lp], force * sign)
        return out

    def push_pc(self, force_world, point_world):
        """External force on the PC applied at a world point (e.g. the port), held until changed."""
        com = wp.to_torch(self.rt.d.xipos)[:, self.pc_body]
        self.xfrc[:, self.pc_body, :3] = force_world
        self.xfrc[:, self.pc_body, 3:] = torch.cross(point_world - com, force_world, dim=1)

    def jac_point(self, side, q, point):
        """Linear Jacobian (B,3,6) of a world point rigidly attached to the hand."""
        p, R, J = self.chain[side].fk(q, with_jac=True)
        r = point - p
        return J[:, :3] - torch.cross(r[:, :, None].expand(-1, -1, 6), J[:, 3:], dim=1), p, R


@dataclass
class HoldCfg:
    mode: str = 'press'           # press | brace | none
    across_m: float = 0.          # contact point across the panel from the box centre (port-frame x)
    along_m: float = .03          # press: from the box centre toward the rear face
    brace_height_m: float = .02   # brace: fingertip-contact height above the case bottom
    force_n: float = 20.          # normal push into the PC (down for press, toward the panel for brace)
    preload_n: float = 0.         # press: tangential push toward the panel (against insertion)
    down_n: float = 0.            # brace: extra downward push (finger side friction)
    standoff_m: float = .015      # pre-contact clearance along the push direction
    move_s: float = 1.5           # home -> standoff
    engage_s: float = .6          # standoff -> contact target (+ force offset)
    settle_s: float = .5
    yaws_deg: tuple = tuple(range(-90, 91, 15))
    tilt_deg: float = 0.          # tool tilt about the PC-frame x axis (toward the panel), press only
    ik_iters: int = 120
    # Guarded engage (deployable: encoders only): descend from the stand-off along the push direction
    # at guard_speed until the encoder-estimated contact force along it exceeds guard_force_n (bias
    # taken at the stand-off), then latch the measured pose and apply dq = Kp^-1 J^T F from there.
    # This removes the absolute-position error (encoder offsets) from the holding force.
    guarded: bool = True
    guard_speed_m_s: float = .01
    guard_depth_m: float = .025   # search this far past the planned contact
    guard_force_n: float = 3.
    ramp_s: float = .4


class LeftHold:
    """Batched scripted hold: plan per world from a PC pose (true or estimated), then track."""

    def __init__(self, harness, cfg=None):
        self.h, self.cfg = harness, cfg or HoldCfg()
        self.box = pc_box()
        self.dev = harness.dev
        self.plan = None

    def _contact_and_frames(self, port_p, port_R):
        """World contact point, push direction (into the PC), tool rotation candidates."""
        c, b = self.cfg, self.box
        n = len(port_p)
        if c.mode == 'press':
            local = torch.tensor([b['centre'][0] + c.across_m, b['top_y'], b['centre'][2] - c.along_m], device=self.dev, dtype=torch.float32)
            normal_local = torch.tensor([0., -1., 0.], device=self.dev)            # down (port y is up)
        else:
            local = torch.tensor([b['centre'][0] + c.across_m, b['bottom_y'] + c.brace_height_m, b['rear_z']], device=self.dev, dtype=torch.float32)
            normal_local = torch.tensor([0., 0., 1.], device=self.dev)             # toward the panel
        point = port_p + (port_R @ local[:, None])[..., 0]
        normal = (port_R @ normal_local[:, None])[..., 0]
        return point, normal, local

    def solve(self, port_p, port_R, seed_q=None):
        """IK for the hold, per world. port pose (B,3),(B,3,3) as the controller believes it."""
        c, h, dev = self.cfg, self.h, self.dev
        n = len(port_p)
        point, normal, _ = self._contact_and_frames(port_p, port_R)
        toward_panel = port_R[:, :, 2]                     # port +z: out of the panel, toward the arms
        # tool z (out of the hand) points down for both holds; yaw about the vertical is free
        yaws = torch.deg2rad(torch.tensor(c.yaws_deg, device=dev, dtype=torch.float32))
        ny = len(yaws)
        pc_yaw = torch.atan2(port_R[:, 1, 0], port_R[:, 0, 0])
        a = pc_yaw[:, None] + yaws[None]
        ca, sa = torch.cos(a), torch.sin(a)
        Rt = torch.zeros(n, ny, 3, 3, device=dev)
        # columns: x = (ca, sa, 0), y = z cross x, z = (0, 0, -1)
        Rt[..., 0, 0], Rt[..., 1, 0] = ca, sa
        Rt[..., 2, 2] = -1
        Rt[..., 0, 1], Rt[..., 1, 1] = sa, -ca
        if c.tilt_deg and c.mode == 'press':
            # tilt the tool so its axis leans toward the panel (pushes the tip against insertion)
            ax = torch.cross(torch.tensor([0., 0., 1.], device=dev).expand(n, 3), toward_panel, dim=1)
            ax = ax / ax.norm(dim=1, keepdim=True)
            Rtilt = K.axis_angle_mat(ax, torch.full((n,), math.radians(c.tilt_deg), device=dev))
            Rt = Rtilt[:, None] @ Rt
        Rt = Rt.reshape(n * ny, 3, 3)
        z = Rt[:, :, 2]
        pt = point.repeat_interleave(ny, 0)
        nrm = normal.repeat_interleave(ny, 0)
        # finger support along the push direction, in the site frame (closed fingers)
        ext = self.extent((Rt.transpose(1, 2) @ nrm[..., None])[..., 0])
        if c.mode == 'press':   # lowest fingertip point on the top surface
            site = pt - nrm * ext[:, None]
        else:                   # pad centroid at the brace height, finger face on the rear face
            site = pt - nrm * ext[:, None] - z * PAD_BELOW_TCP
        q0 = (seed_q if seed_q is not None else self.nominal_seed()).expand(n * ny, 6).clone()
        ch = h.chain['left']
        q, pe, re = ch.ik(site, Rt, q0, iters=c.ik_iters)
        margin = torch.minimum(q - ch.lo, ch.hi - q).min(1).values
        ok = (pe < 1e-3) & (re < math.radians(1)) & (margin > .05)
        score = torch.where(ok, margin, torch.full_like(margin, -1.)).reshape(n, ny)
        best = score.argmax(1)
        sel = torch.arange(n, device=dev) * ny + best
        q, ok, Rsel, site_sel = q[sel], ok[sel], Rt[sel], site[sel]
        # force offset: dq = Kp^-1 J^T F at the contact point
        F = normal * c.force_n
        if c.mode == 'press' and c.preload_n:
            F = F + toward_panel * c.preload_n
        if c.mode == 'brace' and c.down_n:
            F = F + torch.tensor([0., 0., -1.], device=dev) * c.down_n
        Jl, _, _ = h.jac_point('left', q, point)
        dq = (Jl.transpose(1, 2) @ F[..., None])[..., 0] / h.kp[1]
        stand = site_sel - normal * c.standoff_m
        if c.mode == 'press':
            stand = stand + torch.tensor([0., 0., 1.], device=dev) * .0   # straight above
        qs, pes, _ = ch.ik(stand, Rsel, q, iters=60)
        qd, _, _ = ch.ik(site_sel + normal * c.guard_depth_m, Rsel, q, iters=60)
        self.latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.t_latch = torch.full((n,), float('inf'), device=dev)
        self.q_latch = q.clone()
        self.bias = None
        self.last_cmd = None
        self.plan = {'q_contact': q, 'q_force': q + dq, 'q_stand': qs, 'q_deep': qd, 'ok': ok & (pes < 2e-3),
                     'point': point, 'normal': normal, 'force_cmd': F, 'dq_force': dq,
                     'yaw_deg': torch.rad2deg(yaws[best])}
        return self.plan

    def _finger_points(self):
        """Collision-geometry points of the closed left fingers in the left grasp_site frame."""
        m = self.h.rt.mjm
        d = mujoco.MjData(m)
        for f in ('left', 'right'):
            d.qpos[m.joint('left_' + f + '_finger').qposadr] = 0.
        mujoco.mj_forward(m, d)
        site = m.site('left_grasp_site').id
        sp, sR = d.site_xpos[site], d.site_xmat[site].reshape(3, 3)
        pts = []
        for g in range(m.ngeom):
            if m.body(int(m.geom_bodyid[g])).name not in ('left_left_finger', 'left_right_finger'):
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            gR, gp = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[g]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = m.geom_size[g, 0]
                v = np.vstack([np.eye(3) * r, -np.eye(3) * r])
            else:
                continue
            pts.append((gp + v @ gR.T - sp) @ sR)
        return torch.tensor(np.vstack(pts), device=self.dev, dtype=torch.float32)

    def extent(self, dirs):
        """Support of the closed fingers along unit directions given in the site frame (B,3) -> (B,)."""
        if not hasattr(self, '_pts'):
            self._pts = self._finger_points()
        return (dirs @ self._pts.T).max(1).values

    def nominal_seed(self):
        return torch.tensor([-0.6857, 2.0911, 1.5729, -1.0604, 0.001, -0.0084], device=self.dev)

    def force_estimate(self, q_cmd):
        """Encoder-based estimate of the hand force on the PC (B,3): PD torque mapped through J^+T."""
        h, p = self.h, self.plan
        qb = h.believed('left')
        J, _, _ = h.jac_point('left', qb, p['point'])
        tau = h.kp[1] * (q_cmd - qb)
        return (torch.linalg.pinv(J.transpose(1, 2)) @ tau[..., None])[..., 0]

    def target(self, t, q_home):
        """Joint target (believed frame) at time t since the hold started (call once per control step,
        in order: the guarded engage latches per world)."""
        c, p = self.cfg, self.plan
        def blend(x):
            x = min(max(x, 0.), 1.); return x * x * (3 - 2 * x)
        if c.mode == 'none':
            return q_home
        if t < c.move_s:
            cmd = q_home + (p['q_stand'] - q_home) * blend(t / c.move_s)
        elif not c.guarded:
            u = t - c.move_s
            cmd = p['q_stand'] + (p['q_force'] - p['q_stand']) * blend(u / c.engage_s) if u < c.engage_s else p['q_force']
        elif t - c.move_s < .3:                    # dwell at the stand-off, then take the force bias
            cmd = p['q_stand']
        else:
            u = t - c.move_s - .3
            if self.bias is None:
                self.bias = self.force_estimate(self.last_cmd)
            L = c.standoff_m + c.guard_depth_m
            frac = min(1., u * c.guard_speed_m_s / L)
            descend = p['q_stand'] + (p['q_deep'] - p['q_stand']) * frac
            f = self.force_estimate(self.last_cmd) - self.bias
            fn = (f * p['normal']).sum(1)
            new = ((fn > c.guard_force_n) | (frac >= 1.)) & ~self.latched   # end of travel: latch anyway
            if new.any():
                self.q_latch[new] = self.h.believed('left')[new]
                self.t_latch[new] = t
                self.latched |= new
            r = ((t - self.t_latch) / c.ramp_s).clamp(0, 1)[:, None]
            r = r * r * (3 - 2 * r)
            held = self.q_latch + p['dq_force'] * r
            cmd = torch.where(self.latched[:, None], held, descend)
        self.last_cmd = cmd
        return cmd

    def hold_target(self):
        """Final hold command (after the engage)."""
        if self.cfg.mode == 'none':
            return None
        if not self.cfg.guarded:
            return self.plan['q_force']
        return torch.where(self.latched[:, None], self.q_latch + self.plan['dq_force'], self.last_cmd)

    @property
    def duration(self):
        c = self.cfg
        if c.mode == 'none':
            return 0.
        engage = (c.standoff_m + c.guard_depth_m) / c.guard_speed_m_s + c.ramp_s + .3 if c.guarded else c.engage_s
        return c.move_s + engage + c.settle_s


@dataclass
class InsertCfg:
    preinsert_m: float = .006     # plug tip in front of the port at the start
    gain: float = .25
    step_clip: float = .004
    cap: float = .1
    align_s: float = 3.          # align at the pre-insert gap before advancing
    insert_s: float = 3.
    lateral_gate_m: float = 5e-5  # do not advance past the panel while the tip is off-axis by more
    seat_hold_s: float = 1.
    overdrive_m: float = .0       # push the target this far past the seat (insertion force)
    # Provisional in-hand pose (blind_insertion/envs/inhand.py): confirmed 60-deg grasp, randomized
    # +/- inhand_pos_mm (reader axis, vertical) and +/- inhand_rot_deg about the closing axis.
    tilt_deg: float = 60.
    pad_z_m: float = .058
    inhand_pos_mm: float = 3.
    inhand_rot_deg: float = 8.
    # 'measured': the Part 1 post-pick in-hand poses (inhand.sample_measured; 55-deg grasp); 'uniform':
    # the provisional tilt_deg / pad_z_m pose +/- inhand_pos_mm / inhand_rot_deg.
    inhand: str = 'measured'


class RightInserter:
    """Reader teleported into the right gripper at the pre-insert pose (confirmed 60-deg grasp,
    randomized in-hand pose), then the privileged-feedback insertion of audit_bimanual_pick.py."""

    def __init__(self, harness, cfg=None):
        self.h, self.cfg = harness, cfg or InsertCfg()
        rt = harness.rt
        m = rt.mjm
        self.fingers = [int(m.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
        self.depth = float(rt.lay.seating_depth[0])
        dev = harness.dev
        base = [[0, 1.0, 1.0, -.5, 0, 0], [0, 1.6, 1.3, -.9, 0, 0], [0, 2.2, 1.8, -1.2, 0, 0],
                [0, .6, .5, -.3, 0, 0], [0, 2.6, 2.2, -1.2, 0, 0], [1.1772, 2.1079, 1.4923, -.9551, 0., 1.1772]]
        self.seeds = torch.tensor(base, device=dev, dtype=torch.float32)

    def place(self, rel_p=None, rel_R=None, resample_inhand=True, forward=True):
        """Right arm + reader teleported so the reader (tip frame) sits at (rel_R, rel_p) in the port
        frame of the current PC pose (default: on the axis, preinsert_m out), all worlds."""
        from blind_insertion.envs.inhand import sample_inhand
        h, c = self.h, self.cfg
        port_p, port_R = h.pc_pose()
        n, dev = h.n, h.dev
        if resample_inhand or not hasattr(self, 'R_sr'):
            if c.inhand == 'measured':
                from blind_insertion.envs.inhand import sample_measured
                self.R_sr, self.p_sr, self.inhand = sample_measured(n, device=dev)
            else:
                self.R_sr, self.p_sr, self.inhand = sample_inhand(n, c.tilt_deg, c.pad_z_m, c.inhand_pos_mm, c.inhand_rot_deg, device=dev)
        if rel_p is None:
            rel_p = torch.tensor([0., 0., c.preinsert_m], device=dev).expand(n, 3)
        reader_R = port_R if rel_R is None else port_R @ rel_R
        reader_p = port_p + (port_R @ rel_p[..., None])[..., 0]
        Rt = reader_R @ self.R_sr
        pt = reader_p + (reader_R @ self.p_sr[..., None])[..., 0]
        best = None
        ch = h.chain['right']
        for s in self.seeds:
            q0 = s.expand(n, 6).clone()
            q0[:, 0] = torch.atan2(pt[:, 1], pt[:, 0]) if s[0] == 0 else s[0]
            q, pe, re = ch.ik(pt, Rt, q0, iters=150)
            err = pe + .05 * re
            if best is None:
                best = [q, pe, re, err]
            else:
                better = err < best[3]
                for i, v in enumerate((q, pe, re, err)):
                    best[i] = torch.where(better[:, None] if v.dim() == 2 else better, v, best[i])
        q, pe, re, _ = best
        self.ik_ok = (pe < 5e-4) & (re < math.radians(.5))
        rt = h.rt
        qa = rt.indices['right'][0]
        rt.q[:, qa] = q
        pa = rt.lay.peg_qadr
        rt.q[:, pa:pa + 3] = reader_p
        rt.q[:, pa + 3:pa + 7] = K.mat_to_quat(reader_R)
        if not hasattr(self, '_fq'):
            self._fq = self._finger_q(rt.q[0])
        rt.q[:, self.fingers] = self._fq
        rt.v[:] = 0
        if forward:
            with wp.ScopedDevice(rt.cfg.device):
                mjw.forward(rt.m, rt.d)
        self.target = q - h.enc_error[:, 0]     # believed frame: holds the true pose
        h.fresh[:] = True                       # teleported: no stale (latency-delayed) command
        self.q_true0 = q.clone()
        return self.ik_ok

    def _finger_q(self, q0, contact=-2e-4):
        """Finger opening (both fingers) at which the pads just touch the reader (world 0 geometry)."""
        m = self.h.rt.mjm
        d = mujoco.MjData(m)
        d.qpos[:] = q0.cpu().numpy()
        reader = m.body('peg').id
        fingers = {m.body('left_finger').id, m.body('right_finger').id}
        def depth(x):
            d.qpos[self.fingers] = x
            mujoco.mj_forward(m, d)
            ds = [c.dist for c in d.contact[:d.ncon]
                  if {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} & fingers
                  and reader in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))]
            return min(ds) if ds else 1.
        lo, hi = 0., .0475
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if depth(mid) < contact else (lo, mid)
        return hi

    def tip_error(self):
        port_p, port_R = self.h.pc_pose()
        tip, Rr = self.h.reader_pose()
        return port_p, port_R, tip, Rr

    def gap(self, t):
        """Scheduled tip gap along the port axis at time t of the insertion phase (all worlds)."""
        c = self.cfg
        u = min(1., max(0., t - c.align_s) / c.insert_s); u = u * u * (3 - 2 * u)
        return torch.full((self.h.n,), c.preinsert_m + (-self.depth - c.overdrive_m - c.preinsert_m) * u, device=self.h.dev)

    @property
    def duration(self):
        return self.cfg.align_s + self.cfg.insert_s + self.cfg.seat_hold_s

    def track(self, gap_m):
        """One feedback update toward the tip at `gap_m` along the port axis (negative = inside)."""
        h, c = self.h, self.cfg
        port_p, port_R, tip, Rr = self.tip_error()
        rel = (port_R.transpose(1, 2) @ (tip - port_p)[..., None])[..., 0]
        off_axis = rel[:, :2].norm(dim=1) > c.lateral_gate_m
        gap_m = torch.where(off_axis & (rel[:, 2] > -1e-3), torch.maximum(gap_m, torch.full_like(gap_m, 5e-4)), gap_m)
        desired = port_p + port_R[:, :, 2] * gap_m[:, None]
        e_p = desired - tip
        e_r = K.rot_log(port_R @ Rr.transpose(1, 2))
        qb = h.believed('right')
        qt = h.q('right')
        p, R, J = h.chain['right'].fk(qt, with_jac=True)
        Jv = J[:, :3] - torch.cross((tip - p)[:, :, None].expand(-1, -1, 6), J[:, 3:], dim=1)
        Jw = torch.cat([Jv, .1 * J[:, 3:]], 1)
        e = torch.cat([e_p, .1 * e_r], 1)
        dq = (Jw.transpose(1, 2) @ torch.linalg.solve(Jw @ Jw.transpose(1, 2) + 1e-5 * torch.eye(6, device=h.dev), e[..., None]))[..., 0]
        self.target = self.target + torch.nan_to_num(c.gain * dq).clamp(-c.step_clip, c.step_clip)
        self.target = torch.maximum(torch.minimum(self.target, qb + c.cap), qb - c.cap)
        lim = h.chain['right']
        self.target = torch.maximum(torch.minimum(self.target, lim.hi), lim.lo)
        self.target = torch.where(torch.isfinite(self.target), self.target, qb.nan_to_num())
        return self.target
