"""Plug angle from the encoders alone (docs/real/ALIGN_STAGE.md). The reader lies flat on the table when the jaws close, so its thickness axis is vertical at that moment. That fixes the reader in the hand, about the two
axes that gravity sees, per grasp: FK at the close, FK now, the port axes. Joint offsets mostly cancel: the same joints enter both poses. Not observable this way: yaw about the vertical (across the port: ry here).

Reader frame as the port frame when inserted: x across, y up, z along the plug, out of the port. Rotation vector of reader against port: rx = pitch (about across), rz = roll (about the plug), ry = yaw (about up).
"""
from __future__ import annotations

import numpy as np
import torch


def rot_between(a, b):
    """Smallest rotation matrix taking unit vector a to unit vector b."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def reader_in_site_from_close(R_site_close, R0_reader_in_site):
    """Reader in the grasp_site frame, from the site rotation at the close: nominal in-hand rotation, corrected so that the reader's y axis was vertical at the close."""
    R = np.asarray(R_site_close, float) @ np.asarray(R0_reader_in_site, float)
    flat = rot_between(R[:, 1], np.array([0., 0., 1.])) @ R
    return np.asarray(R_site_close, float).T @ flat


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def plug_rv_deg(R_site_now, R_rs, eR):
    """Reader against port as a rotation vector, deg (rx, ry, rz); ry carries the unobserved in-hand yaw and the ZED yaw."""
    return np.degrees(rotvec(np.asarray(eR, float).T @ np.asarray(R_site_now, float) @ R_rs))


def close_rotation(chain, q_believed):
    return chain.fk(torch.as_tensor(q_believed, dtype=torch.float32).reshape(1, 6))[1][0].numpy().astype(float)
