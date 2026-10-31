"""Identified gravity/friction residual of the real YAM arms under the i2rt driver.

The real driver adds `f_g * g_driver(q_enc)` (its own gravity model, scaled per joint) to
the PD torque. The scene bodies carry `gravcomp=1`, which cancels the scene's own gravity
exactly; the residual torque the simulator must add is therefore

    tau_res(q) = f_g * g_driver(q_enc) - g_true(q_model)

with `g_true` linear in per-link mass/first moments (`theta`, identified by
`scripts/real2sim/identify_arm_dynamics.py`). Effective Coulomb friction becomes joint
frictionloss. Both gravity terms use the same regressor: the driver chain has the scene's
joint axes and anchors (anchors differ only along the axes), so `theta_driver` is the
driver model's inertia expressed in the scene link frames.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

LINKS = ('link_2', 'link_3', 'link_4', 'link_5', 'link_6')
DRIVER_XML = Path(__file__).resolve().parents[2] / 'results/usbc/bimanual_20261029/driver_model/driver_gravity.xml'
PROFILE = Path(__file__).resolve().parents[2] / 'results/usbc/bimanual_20261029/dynamics_id/profile.json'


def _rigid_bodies(m, root):
    """Bodies rigidly attached to `root` (finger slides count as rigid at their current opening)."""
    out = [root]
    for b in range(m.nbody):
        p, ok = b, False
        while p not in (0, root):
            if m.body_jntnum[p] and any(m.jnt_type[m.body_jntadr[p] + k] == mujoco.mjtJoint.mjJNT_HINGE
                                        for k in range(m.body_jntnum[p])):
                break
            p = m.body_parentid[p]
        else:
            ok = p == root and b != root
        if ok:
            out.append(b)
    return out


class ArmRegressor:
    """CPU gravity regressor of one arm in the bimanual scene (link frames, holding-torque sign)."""

    def __init__(self, side, model=None):
        from blind_insertion.scenes import pc_bench as S
        self.m = model if model is not None else S.build_scene(bimanual=True)[0]
        self.prefix = 'left_' if side == 'left' else ''
        m, pre = self.m, self.prefix
        self.links = list(LINKS)
        self.bodies = [m.body(pre + n).id for n in LINKS]
        self.qadr = [int(m.joint(pre + f'joint{i}').qposadr[0]) for i in range(1, 7)]
        self.dof = [int(m.joint(pre + f'joint{i}').dofadr[0]) for i in range(1, 7)]
        self.P = 4 * len(LINKS)
        self.d = mujoco.MjData(m)
        mujoco.mj_forward(m, self.d)
        self.theta_scene = np.concatenate([self._link_params(b) for b in self.bodies])

    def _link_params(self, link):
        """Scene mass and first moment of everything rigid on `link`, in the link frame."""
        m, d = self.m, self.d
        R, p = d.xmat[link].reshape(3, 3), d.xpos[link]
        mass, moment = 0.0, np.zeros(3)
        for b in _rigid_bodies(m, link):
            mass += m.body_mass[b]
            moment += m.body_mass[b] * (R.T @ (d.xipos[b] - p))
        return np.r_[mass, moment]

    def set_q(self, q_model):
        self.d.qpos[self.qadr] = q_model
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def regressor(self, q_model):
        """(6, P): holding torque on the six arm joints per unit link parameter."""
        self.set_q(q_model)
        m, d = self.m, self.d
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        up = np.array([0.0, 0.0, -m.opt.gravity[2]])
        Y = np.zeros((6, self.P))
        for k, b in enumerate(self.bodies):
            R, p = d.xmat[b].reshape(3, 3), d.xpos[b].copy()
            mujoco.mj_jac(m, d, jp, jr, p, b)
            J0 = jp[:, self.dof].copy()
            Y[:, 4 * k] = J0.T @ up
            for i in range(3):
                mujoco.mj_jac(m, d, jp, jr, p + R[:, i], b)
                Y[:, 4 * k + 1 + i] = (jp[:, self.dof] - J0).T @ up
        return Y


def driver_theta(side, model=None):
    """Driver gravity model inertia in scene link frames, by least squares on random poses."""
    reg = ArmRegressor(side, model)
    dm = mujoco.MjModel.from_xml_path(str(DRIVER_XML))
    dd = mujoco.MjData(dm)
    rng = np.random.default_rng(0)
    Ys, gs = [], []
    for q in rng.uniform([-1, 0, 0, -1.5, -1.5, -2], [1, 3.5, 3.5, 1.5, 1.5, 2], (400, 6)):
        dd.qpos[:6] = q
        mujoco.mj_forward(dm, dd)
        Ys.append(reg.regressor(q)[1:]); gs.append(dd.qfrc_bias[1:6].copy())
    Y, g = np.concatenate(Ys), np.concatenate(gs)
    theta, *_ = np.linalg.lstsq(Y, g, rcond=None)
    rms = float(np.sqrt(np.mean((Y @ theta - g) ** 2)))
    return theta, rms


def load_profile(path=PROFILE):
    return json.loads(Path(path).read_text())


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class DriverGravity:
    """Batched i2rt driver gravity torque `g_driver(q_enc)` on GPU (holding-torque sign).

    Forward kinematics of the archived driver chain in its own upright base frame, as the
    driver computes it; slide fingers stay at their zero opening (as in the driver XML).
    """

    def __init__(self, device, path=DRIVER_XML):
        dm = mujoco.MjModel.from_xml_path(str(path))
        self.parent = dm.body_parentid.tolist()
        self.hinge = [int(dm.body_jntadr[b]) if dm.body_jntnum[b] and dm.jnt_type[dm.body_jntadr[b]] == 3 else -1
                      for b in range(dm.nbody)]
        f = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=torch.float32)
        self.pos, self.rot = f(dm.body_pos), f(np.stack([_quat_mat(q) for q in dm.body_quat]))
        self.ipos, self.mass = f(dm.body_ipos), f(dm.body_mass)
        self.axis = f(dm.jnt_axis)
        self.g = float(-dm.opt.gravity[2])
        self.nbody = dm.nbody

    @staticmethod
    def _axis_rot(axis, angle):
        """Rodrigues rotation, batched over angle (W,)."""
        K = torch.zeros(3, 3, device=axis.device)
        K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
        s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
        return torch.eye(3, device=axis.device) + s * K + (1 - c) * (K @ K)

    def __call__(self, q_enc):
        """q_enc (W, 6) -> holding torque (W, 6)."""
        W = q_enc.shape[0]
        R = [torch.eye(3, device=q_enc.device).expand(W, 3, 3)]
        p = [torch.zeros(W, 3, device=q_enc.device)]
        anchor, zaxis = [None] * 6, [None] * 6
        for b in range(1, self.nbody):
            Rp, pp = R[self.parent[b]], p[self.parent[b]]
            Rb, pb = Rp @ self.rot[b], pp + Rp @ self.pos[b]
            j = self.hinge[b]
            if 0 <= j < 6:
                anchor[j], zaxis[j] = pb, Rb @ self.axis[j]
                Rb = Rb @ self._axis_rot(self.axis[j], q_enc[:, j])
            R.append(Rb); p.append(pb)
        com = torch.stack([p[b] + R[b] @ self.ipos[b] for b in range(self.nbody)], 1)     # (W, B, 3)
        weight = self.mass[None, :, None] * torch.tensor([0.0, 0.0, -self.g], device=q_enc.device)
        tau = []
        for j in range(6):
            body = next(b for b in range(self.nbody) if self.hinge[b] == j)
            sub = [b for b in range(self.nbody) if self._descends(b, body)]
            lever = com[:, sub] - anchor[j][:, None]
            tau.append(-(torch.cross(zaxis[j][:, None].expand_as(lever), lever, dim=-1) * weight[:, sub]).sum((1, 2)))
        return torch.stack(tau, 1)

    def _descends(self, b, root):
        while b:
            if b == root:
                return True
            b = self.parent[b]
        return False
