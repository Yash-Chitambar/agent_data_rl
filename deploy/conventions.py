"""Frames and maps between the real YAM and the sim models (checked by deploy/checks/conventions.py).

Joint frames
  motor  what the i2rt driver reads and takes (encoder zeros as installed).
  model  the Menagerie kinematics the sim uses: q_model = q_motor + rig.joint_offsets(side). The sim's
         "encoder frame" (PickEnv.state, ACT's state and targets) is this frame plus a residual
         error (PickCfg.phys enc_offset_deg, +-0.3 deg in the pipeline evals).
  ACT    state and absolute targets in the model frame -> send target - offsets.
  insert observation in the model frame; relative target q_motor + scale * a (offsets cancel).

Gripper
  sim    command +1 open / -1 close (constant-force finger motors, 4 N open, 9-15 N squeeze), state =
         mean finger slide (m, 0 closed .. 0.048 open; ~19 mm on the reader, ~-6 mm closed empty).
  real   i2rt gripper 0 (closed) .. 1 (open), position PD. GripperMap turns one into the other; its
         finger map is linear through (g_open, m_open) and, once measured, (g_reader, m_reader).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "scripts/real"))   # after REPO: its deploy.py must not shadow this package
import rig  # noqa: E402
from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.envs import kinematics as K  # noqa: E402
from blind_insertion.scenes import pc_bench as S  # noqa: E402


def joint_offsets(side="right"):
    return np.asarray(rig.joint_offsets(side), float)


class GripperMap:
    def __init__(self, open=1.0, close=0.0, m_open=0.048, g_reader=None, m_reader=0.019):
        self.open, self.close = open, close
        if g_reader is None:            # uncalibrated: proportional
            self.a, self.b = m_open / open, 0.0
        else:
            self.a = (m_open - m_reader) / (open - g_reader)
            self.b = m_open - self.a * open
        self.calibrated = g_reader is not None

    def command(self, sign):
        """Sim gripper command (+1 open, -1 close) -> i2rt target 0..1."""
        return self.open if sign > 0 else self.close

    def finger_m(self, g):
        """i2rt gripper reading -> the sim's mean finger opening (m)."""
        return self.a * float(g) + self.b

    def reading(self, finger_m):
        """Inverse of finger_m (the sim backend reports the fingers this way)."""
        return (finger_m - self.b) / self.a


class Robot:
    """FlexPoint YAM kinematics in the model frame (the gripper on the real arms and in the Task H
    sim). scripts/real/deploy.py's ObsBuilder loads the linear gripper, whose tcp differs: not used here."""

    def __init__(self, device="cpu"):
        spec, info = B.load_robot("yam", gripper="flexpoint")
        m = spec.compile()
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, info.tcp_site)
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in info.arm_joints]
        self.device = torch.device(device)
        self.chain = K.Chain(m, site, np.array(qadr), device=device)
        self.home = torch.tensor(info.home, dtype=torch.float32, device=device)
        self.R_base = K.rpy_mat(torch.tensor(S.PORT_BASE_RPY_DEG, device=device) * math.pi / 180)
        self.R_top = torch.tensor(S.R_TOP_TCP, device=device)

    def t(self, x):
        return torch.as_tensor(np.asarray(x, np.float32), device=self.device).reshape(1, -1)

    def fk(self, q_model, qd=None):
        """tcp position (3), rotation (3, 3), and linear and angular velocity (6) if qd is given."""
        q = self.t(q_model)
        if qd is None:
            p, R = self.chain.fk(q)
            return p[0], R[0]
        p, R, J = self.chain.fk(q, with_jac=True)
        return p[0], R[0], (J @ self.t(qd)[..., None])[0, :, 0]

    def ready(self, reader_xy=S.READER_XY):
        """PickEnv's ready pose: tcp 18 cm above the reader area, top-down, plug direction +x."""
        q, pe, _ = self.chain.ik(torch.tensor([[reader_xy[0], reader_xy[1], 0.18]], device=self.device),
                                 (self.R_base @ self.R_top)[None], self.home[None], iters=200, q_rest=self.home)
        assert pe.item() < 1e-3, "ready pose IK failed"
        return q[0].cpu().numpy().astype(float)
