"""Batched serial-chain kinematics in torch, read from an MjModel.

Used for (1) the actor's EE pose/twist observation, computed from (possibly noisy) joint
readings only, and (2) IK at reset. Validated against MuJoCo in checks/test_kinematics.py.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) wxyz, robust branch selection."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + t, min=1e-12)) / 2
    qx = torch.sqrt(torch.clamp(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=1e-12)) / 2
    qy = torch.sqrt(torch.clamp(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], min=1e-12)) / 2
    qz = torch.sqrt(torch.clamp(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], min=1e-12)) / 2
    qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    q = torch.stack([qw, qx, qy, qz], -1)
    return q / q.norm(dim=-1, keepdim=True)


def axis_angle_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis (..., 3) unit, angle (...)."""
    K = skew(axis)
    s = torch.sin(angle)[..., None, None]
    c = torch.cos(angle)[..., None, None]
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape)
    return eye + s * K + (1 - c) * (K @ K)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(v.shape[:-1] + (3, 3))


def rot_log(R: torch.Tensor) -> torch.Tensor:
    """SO(3) log map -> rotation vector (..., 3)."""
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)
    scale = torch.where(ang > 1e-4, ang / (2 * torch.sin(ang)), 0.5 + ang ** 2 / 12)
    return v * scale[..., None]


def euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """Intrinsic x-y-z Euler angles (roll, pitch, yaw) of R = Rx Ry Rz."""
    pitch = torch.asin(R[..., 0, 2].clamp(-1, 1))
    roll = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    yaw = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([roll, pitch, yaw], -1)


def rpy_mat(rpy: torch.Tensor) -> torch.Tensor:
    """R = Rz(yaw) Ry(pitch) Rx(roll) (extrinsic xyz), rpy (..., 3)."""
    r, p, y = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr], -1).reshape(rpy.shape[:-1] + (3, 3))


class Chain:
    """Serial chain from the world to a site through hinge joints."""

    def __init__(self, m: mujoco.MjModel, site: int, joint_qadr: np.ndarray, device="cuda"):
        self.device = device
        body = m.site_bodyid[site]
        bodies = []
        while body != 0:
            bodies.append(body)
            body = m.body_parentid[body]
        bodies = bodies[::-1]
        qadr = list(joint_qadr)
        self.steps = []  # (pos, R, axis or None, joint index)
        for b in bodies:
            pos = torch.tensor(m.body_pos[b], dtype=torch.float32, device=device)
            R = quat_to_mat(torch.tensor(m.body_quat[b], dtype=torch.float32, device=device))
            jidx = None
            axis = None
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if m.jnt_qposadr[j] not in qadr:
                    continue    # unmeasured joints (e.g. backlash) are invisible to the encoder FK
                assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, "chain supports hinges only"
                assert np.allclose(m.jnt_pos[j], 0), "chain assumes joint at body origin"
                jidx = qadr.index(m.jnt_qposadr[j])
                axis = torch.tensor(m.jnt_axis[j], dtype=torch.float32, device=device)
            self.steps.append((pos, R, axis, jidx))
        self.site_pos = torch.tensor(m.site_pos[site], dtype=torch.float32, device=device)
        self.site_R = quat_to_mat(torch.tensor(m.site_quat[site], dtype=torch.float32, device=device))
        self.n = len(qadr)
        self.lo = torch.tensor(m.jnt_range[[self._jid(m, a) for a in qadr], 0], dtype=torch.float32,
                               device=device)
        self.hi = torch.tensor(m.jnt_range[[self._jid(m, a) for a in qadr], 1], dtype=torch.float32,
                               device=device)

    @staticmethod
    def _jid(m, qadr):
        return int(np.where(m.jnt_qposadr == qadr)[0][0])

    def fk(self, q: torch.Tensor, with_jac: bool = False):
        """q (B, n) -> site pos (B,3), R (B,3,3), and optionally the geometric Jacobian (B,6,n)."""
        B = q.shape[0]
        p = torch.zeros(B, 3, device=q.device)
        R = torch.eye(3, device=q.device).expand(B, 3, 3).clone()
        axes, origins = [None] * self.n, [None] * self.n
        for pos, Rb, axis, j in self.steps:
            p = p + (R @ pos)
            R = R @ Rb
            if axis is not None:
                axes[j] = R @ axis
                origins[j] = p
                R = R @ axis_angle_mat(axis.expand(B, 3), q[:, j])
        ps = p + R @ self.site_pos
        Rs = R @ self.site_R
        if not with_jac:
            return ps, Rs
        A = torch.stack(axes, -1)                                # (B,3,n)
        O = torch.stack(origins, -1)                             # (B,3,n)
        Jv = torch.cross(A, ps[..., None] - O, dim=1)
        J = torch.cat([Jv, A], 1)
        return ps, Rs, J

    def ik(self, p_t: torch.Tensor, R_t: torch.Tensor, q0: torch.Tensor, *, iters: int = 80,
           damping: float = 1e-3, q_rest: torch.Tensor | None = None, null_gain: float = 0.05):
        """Damped least squares to a pose; returns q and (pos err, rot err)."""
        q = q0.clone()
        for it in range(iters):
            p, R, J = self.fk(q, with_jac=True)
            e = torch.cat([p_t - p, rot_log(R_t @ R.transpose(1, 2))], 1)   # world frame
            JJt = J @ J.transpose(1, 2) + damping * torch.eye(6, device=q.device)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]
            if q_rest is not None and self.n > 6:
                Jp = J.transpose(1, 2) @ torch.linalg.solve(JJt, J)
                N = torch.eye(self.n, device=q.device) - Jp
                dq = dq + null_gain * (N @ (q_rest - q)[..., None])[..., 0]
            step = 1.0 if it < iters - 10 else 0.5
            q = torch.max(torch.min(q + step * dq, self.hi), self.lo)
        p, R = self.fk(q)
        pe = (p_t - p).norm(dim=1)
        re = rot_log(R_t @ R.transpose(1, 2)).norm(dim=1)
        return q, pe, re
