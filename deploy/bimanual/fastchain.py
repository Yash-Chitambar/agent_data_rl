"""K.Chain with a NumPy forward kinematics for small CPU batches (the real path runs one world).

envs/kinematics.Chain.fk is ~150 tiny torch ops (1.6 ms per call on one world on mason); PickExpert.act calls it ~17
times per 50 Hz tick through Chain.ik. FastChain keeps K.Chain's IK and every other method (inherited, unchanged)
and only replaces fk() by the same product of joint transforms in float64 NumPy (Rodrigues per hinge, geometric
Jacobian from the joint axes and origins), returned as float32 tensors. deploy/checks/bimanual_conventions.py
checks it against K.Chain (FK and IK solutions).
"""
from __future__ import annotations

import numpy as np
import torch

from blind_insertion.envs import kinematics as K


class FastChain(K.Chain):
    def __init__(self, m, site, joint_qadr, device='cpu'):
        super().__init__(m, site, np.asarray(joint_qadr), device='cpu')
        f = lambda t: t.detach().cpu().numpy().astype(np.float64)  # noqa: E731
        self._steps = [(f(pos), f(Rb), None if axis is None else f(axis), j) for pos, Rb, axis, j in self.steps]
        self._site_pos, self._site_R = f(self.site_pos), f(self.site_R)

    def fk(self, q, with_jac=False):
        if q.device.type != 'cpu' or q.shape[0] > 16 or q.requires_grad:
            return super().fk(q, with_jac)
        qn = q.detach().numpy().astype(np.float64)
        B = qn.shape[0]
        p = np.zeros((B, 3))
        R = np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
        axes, origins = [None] * self.n, [None] * self.n
        for pos, Rb, axis, j in self._steps:
            p = p + R @ pos
            R = R @ Rb
            if axis is not None:
                axes[j] = R @ axis
                origins[j] = p
                x, y, z = axis
                Kx = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                s, c = np.sin(qn[:, j])[:, None, None], np.cos(qn[:, j])[:, None, None]
                R = R @ (np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx))
        ps = p + R @ self._site_pos
        Rs = R @ self._site_R
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # noqa: E731
        if not with_jac:
            return t(ps), t(Rs)
        A = np.stack(axes, -1)
        O = np.stack(origins, -1)
        Jv = np.cross(A, ps[..., None] - O, axis=1)
        return t(ps), t(Rs), t(np.concatenate([Jv, A], 1))

    def _fk_np(self, qn, with_jac):
        p, R, J = self.fk(torch.from_numpy(qn.astype(np.float32)), with_jac=True)
        return p.numpy().astype(np.float64), R.numpy().astype(np.float64), J.numpy().astype(np.float64)

    def ik(self, p_t, R_t, q0, *, iters=80, damping=1e-3, q_rest=None, null_gain=0.05):
        """K.Chain.ik (damped least squares, last 10 iterations at half step, joint-limit clamp) in NumPy for small
        CPU batches; the 6-joint arms take no null-space term (K.Chain applies it only for n > 6)."""
        if q0.device.type != 'cpu' or q0.shape[0] > 16 or self.n > 6:
            return super().ik(p_t, R_t, q0, iters=iters, damping=damping, q_rest=q_rest, null_gain=null_gain)
        f = lambda x: x.detach().cpu().numpy().astype(np.float64)  # noqa: E731
        pt, Rt, q = f(p_t), f(R_t), f(q0).copy()
        lo, hi = f(self.lo), f(self.hi)
        I6 = np.eye(6)

        def log(Rm):
            cos = np.clip((np.trace(Rm, axis1=-2, axis2=-1) - 1) / 2, -1 + 1e-7, 1 - 1e-7)
            ang = np.arccos(cos)
            v = np.stack([Rm[..., 2, 1] - Rm[..., 1, 2], Rm[..., 0, 2] - Rm[..., 2, 0], Rm[..., 1, 0] - Rm[..., 0, 1]], -1)
            sc = np.where(ang > 1e-4, ang / (2 * np.sin(ang)), 0.5 + ang ** 2 / 12)
            return v * sc[..., None]
        for it in range(iters):
            p, R, J = self._fk_np(q, True)
            e = np.concatenate([pt - p, log(Rt @ np.swapaxes(R, 1, 2))], 1)
            JJt = J @ np.swapaxes(J, 1, 2) + damping * I6
            dq = (np.swapaxes(J, 1, 2) @ np.linalg.solve(JJt, e[..., None]))[..., 0]
            step = 1.0 if it < iters - 10 else 0.5
            q = np.minimum(np.maximum(q + step * dq, lo), hi)
        p, R, _ = self._fk_np(q, True)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))  # noqa: E731
        return t(q), t(np.linalg.norm(pt - p, axis=1)), t(np.linalg.norm(log(Rt @ np.swapaxes(R, 1, 2)), axis=1))
