"""Single-world CPU MuJoCo scenes with deterministic placement, for Phase A checks."""
from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from blind_insertion.assets import builder as B
from blind_insertion.envs import kinematics as K


class CpuScene:
    def __init__(self, robot="flexiv", variant: B.Variant | None = None, gains=None,
                 fix_p=None, fix_rpy=(0, 0, 0), grip_force=None, pad_solref=None, pad_solimp=None,
                 pad_spheres=0):
        if fix_p is None:
            fix_p = (0.65, -0.05, 0.05) if robot == "flexiv" else (0.35, 0.0, -0.20)
        variant = variant or B.sample_study_pegs(1, 0)[0]
        self.v = variant
        self.m, self.lay = B.build(robot, [variant], gains=gains, pad_spheres=pad_spheres)
        B.apply_variant_cpu(self.m, self.lay, 0)
        if pad_solref is not None:
            self.m.geom_solref[self.lay.pad_geoms] = pad_solref
        if pad_solimp is not None:
            self.m.geom_solimp[self.lay.pad_geoms, :3] = pad_solimp
        self.d = mujoco.MjData(self.m)
        self.chain = K.Chain(self.m, self.lay.tcp_site, self.lay.arm_qadr, device="cpu")
        self.home = torch.tensor(self.lay.robot.home, dtype=torch.float32)
        self.fix_p = np.asarray(fix_p, float)
        self.fix_R = K.rpy_mat(torch.tensor(fix_rpy, dtype=torch.float32)).numpy().astype(float)
        self.d.mocap_pos[0] = self.fix_p
        self.d.mocap_quat[0] = K.mat_to_quat(torch.tensor(self.fix_R)).numpy()
        self.grip = self.lay.robot.grip_force * 2.0 if grip_force is None else grip_force
        self.scale = self.lay.robot.action_scale
        self.R_nom = (K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(self.lay.robot.nom_yaw_deg)]))
                      @ K.rpy_mat(torch.tensor([math.pi, 0.0, 0.0]))).numpy()
        self.edge = self.lay.edge_points[0]

    # placement ------------------------------------------------------------------
    def place(self, xy=(0.0, 0.0), height=0.0005, tilt_deg=0.0, grasp_frac=None):
        """Peg bottom centre at fixture-frame (x, y, z_lowest=height), held aligned (or tilted).
        Grasp height: grasp_frac x peg length, default the variant's (Layout.grasp_z: 0.75 L for
        study pegs, the handle's top for USB-C / D-cell parts, as in the GPU env)."""
        lay = self.lay
        tilt = math.radians(tilt_deg)
        Rpg = K.rpy_mat(torch.tensor([math.pi, tilt, 0.0])).numpy()
        gz = lay.grasp_z[0] if grasp_frac is None else grasp_frac * self.v.peg_length
        gp = np.array([0.0, 0.0, gz])
        p_pt = -Rpg @ gp
        R_tf = self.R_nom
        R_pf = R_tf @ Rpg
        pts = np.c_[self.edge, np.zeros(64)] @ R_pf.T
        p_pf = np.array([xy[0], xy[1], height - pts[:, 2].min()])
        p_tf = p_pf - R_tf @ p_pt
        Rw = self.fix_R @ R_tf
        pw = self.fix_p + self.fix_R @ p_tf
        q, pe, re = self.chain.ik(torch.tensor(pw[None], dtype=torch.float32),
                                  torch.tensor(Rw[None], dtype=torch.float32), self.home[None],
                                  iters=150, q_rest=self.home)
        assert pe.item() < 3e-4, f"IK failed {pe.item()}"
        q = q[0].numpy().astype(float)
        mujoco.mj_resetData(self.m, self.d)
        self.d.mocap_pos[0] = self.fix_p
        self.d.mocap_quat[0] = K.mat_to_quat(torch.tensor(self.fix_R)).numpy()
        self.d.qpos[lay.arm_qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        tp = self.d.site_xpos[lay.tcp_site].copy()
        tR = self.d.site_xmat[lay.tcp_site].reshape(3, 3).copy()
        R_peg = tR @ Rpg
        p_peg = tp - R_peg @ gp
        half = lay.peg_half_short[0]
        fq = lay.robot.max_opening - half if lay.robot.name == "flexiv" else half
        self.d.qpos[lay.finger_qadr] = fq
        pa = lay.peg_qadr
        self.d.qpos[pa:pa + 3] = p_peg
        self.d.qpos[pa + 3:pa + 7] = K.mat_to_quat(torch.tensor(R_peg)).numpy()
        self.d.ctrl[lay.arm_act] = q
        self.d.ctrl[lay.grip_act] = self.grip
        mujoco.mj_forward(self.m, self.d)
        return q

    # state ------------------------------------------------------------------------
    @property
    def q(self):
        return self.d.qpos[self.lay.arm_qadr].copy()

    @property
    def qd(self):
        return self.d.qvel[self.lay.arm_dofadr].copy()

    def peg_fixture(self):
        pa = self.lay.peg_qadr
        pw = self.d.qpos[pa:pa + 3]
        Rw = np.zeros(9)
        mujoco.mju_quat2Mat(Rw, self.d.qpos[pa + 3:pa + 7])
        Rw = Rw.reshape(3, 3)
        return self.fix_R.T @ (pw - self.fix_p), self.fix_R.T @ Rw

    def peg_plate_force(self):
        """Total contact force (world) from fixture geoms on the peg."""
        fs = set(self.lay.fixture_slots.tolist())
        ps = set(self.lay.peg_slots.tolist())
        F = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 in fs and c.geom2 in ps) or (c.geom2 in fs and c.geom1 in ps):
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                F += fw if c.geom2 in ps else -fw   # force on the peg
        return F

    def step_ctrl(self, q_target, n=20):
        self.d.ctrl[self.lay.arm_act] = q_target
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    def jac(self):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.lay.tcp_site)
        return np.vstack([jp, jr])[:, self.lay.arm_dofadr]

    def tcp(self):
        return self.d.site_xpos[self.lay.tcp_site].copy(), self.d.site_xmat[self.lay.tcp_site].reshape(3, 3).copy()
