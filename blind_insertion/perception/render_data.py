"""Labelled RGB renders for the approach estimators, generated on the GPU with the bimanual runtime
and the policy/v1 visual DR defaults (appearance, lighting, image noise, camera pose/intrinsics error
at real scale).

Each sample: a reset (PC pose +/-3 cm/+/-10 deg around the nominal, visual DR), the left arm at its
press hold (kinematic), the reader in the right hand (confirmed 60-deg grasp, in-hand pose +/-3 mm /
+/-8 deg) and teleported (IK) to a random pose relative to port 2: lateral x/y, gap along the port
axis, small rotations. Labels are in the port frame (x across the panel, y up, z out of the panel):
plug tip position (m) and the rotation vector of the reader relative to the port (rad); and the PC
port pose in the right-base frame (x, y, yaw).
"""
from dataclasses import dataclass
import math

import torch
import warp as wp
import mujoco_warp as mjw

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
from blind_insertion.envs.left_hold import Harness, HarnessCfg, HoldCfg, InsertCfg, LeftHold, RightInserter


@dataclass
class SampleCfg:
    lateral_m: float = .012       # +/- tip offset across / along the vertical of the port
    gap_m: tuple = (.003, .045)   # tip distance in front of the panel
    rot_deg: float = 6.           # +/- rotation about the port x and y axes
    roll_deg: float = 3.          # +/- about the port axis


class ApproachRenderer:
    def __init__(self, nworld=256, res=320, visual_dr=True, device='cuda:0', seed=0):
        self.rt = BimanualRuntime(BimanualCfg(nworld=nworld, device=device, render_rgb=True, visual_dr=visual_dr,
                                              camera_resolution=res, measured_camera_cadence=False))
        self.h = Harness(self.rt, HarnessCfg(enc_offset_deg=0., latency_steps=(0,)))
        self.hold = LeftHold(self.h, HoldCfg(mode='press', across_m=-.0208, along_m=-.02, yaws_deg=(-90.,), ik_iters=60))
        self.ins = RightInserter(self.h, InsertCfg())
        self.ins.seeds = self.ins.seeds[[1, 5]]
        self.n, self.dev = nworld, self.rt.device
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed)
        torch.manual_seed(seed)
        m = self.rt.mjm
        self.left_fingers = [int(m.joint('left_' + f + '_finger').qposadr[0]) for f in ('left', 'right')]

    def _u(self, lo, hi, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.gen) * (hi - lo) + lo

    def sample_top(self, nominal=False):
        """ZED frames before the arms move (reset pose, reader on the table): PC pose labels only."""
        h, n = self.h, self.n
        h.reset()
        if nominal:
            self.rt.visual.set_nominal(torch.arange(n, device=self.dev))
        with wp.ScopedDevice(self.rt.cfg.device):
            mjw.forward(self.rt.m, self.rt.d)
        port_p, port_R = h.pc_pose()
        frames = self.rt.render()
        yaw = torch.atan2(-port_R[:, 1, 2], -port_R[:, 0, 2])   # insertion direction (-port z) in the plane
        return frames['top'], {'port_xy': port_p[:, :2].clone(), 'port_yaw': yaw}

    def sample_states(self, qpos):
        """Re-render recorded states (DAgger: the in-loop distribution) under a fresh visual-DR draw.
        qpos (nworld, nq): full model state (arms, PC, reader) as recorded during closed-loop runs."""
        h, rt = self.h, self.rt
        h.reset()                                   # new visual DR (and layout, overwritten below)
        rt.q[:] = qpos
        rt.v[:] = 0
        with wp.ScopedDevice(rt.cfg.device):
            mjw.forward(rt.m, rt.d)
        frames = rt.render()
        pp, pR = h.pc_pose()
        tip, rR = h.reader_pose()
        rel_p = (pR.transpose(1, 2) @ (tip - pp)[..., None])[..., 0]
        rel_rv = K.rot_log(pR.transpose(1, 2) @ rR)
        ok = torch.isfinite(qpos).all(1) & (rel_p[:, 2] > .001) & (rel_p[:, :2].abs() < .02).all(1)
        return frames, {'rel_p': rel_p, 'rel_rv': rel_rv, 'ok': ok}

    def sample(self, cfg=None, nominal=False):
        c, n, h = cfg or SampleCfg(), self.n, self.h
        h.reset()
        if nominal:
            self.rt.visual.set_nominal(torch.arange(n, device=self.dev))
        port_p, port_R = h.pc_pose()
        plan = self.hold.solve(port_p, port_R)
        rt = self.rt
        rt.q[:, rt.indices['left'][0]] = plan['q_contact']
        rt.q[:, self.left_fingers] = 0.
        rel_p = torch.stack([self._u(-c.lateral_m, c.lateral_m, n), self._u(-c.lateral_m, c.lateral_m, n),
                             self._u(*c.gap_m, n)], 1)
        rv = torch.stack([self._u(-1, 1, n) * math.radians(c.rot_deg), self._u(-1, 1, n) * math.radians(c.rot_deg),
                          self._u(-1, 1, n) * math.radians(c.roll_deg)], 1)
        rel_R = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
        # the tip is the reader origin; keep the tip at rel_p while rotating
        ok = self.ins.place(rel_p, rel_R, resample_inhand=True)
        frames = rt.render()
        yaw = torch.atan2(-port_R[:, 1, 2], -port_R[:, 0, 2])  # insertion direction (-port z) in the plane
        lab = {'rel_p': rel_p, 'rel_rv': K.rot_log(rel_R), 'port_xy': port_p[:, :2].clone(),
               'port_yaw': yaw, 'ok': ok & plan['ok'], 'inhand': self.ins.inhand,
               'cam_err': rt.visual.params['cam_pos_err_mm'].clone() if 'cam_pos_err_mm' in rt.visual.params else None}
        return frames, lab
