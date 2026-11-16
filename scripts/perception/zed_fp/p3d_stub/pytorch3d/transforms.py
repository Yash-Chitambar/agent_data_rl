"""Minimal stand-in for pytorch3d.transforms (same formulas): FoundationPose's refiner uses so3_exp_map / rotation_6d_to_matrix."""
import torch
import torch.nn.functional as F


def _hat(v):
    z = torch.zeros_like(v[:, 0])
    return torch.stack([z, -v[:, 2], v[:, 1], v[:, 2], z, -v[:, 0], -v[:, 1], v[:, 0], z], -1).view(-1, 3, 3)


def so3_exp_map(log_rot, eps=1e-4):
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = _hat(log_rot)
    skews_square = torch.bmm(skews, skews)
    return (fac1[:, None, None] * skews + fac2[:, None, None] * skews_square
            + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None])


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _na(*a, **k):
    raise NotImplementedError('pytorch3d stub')


so3_log_map = se3_exp_map = se3_log_map = matrix_to_axis_angle = matrix_to_euler_angles = euler_angles_to_matrix = _na
