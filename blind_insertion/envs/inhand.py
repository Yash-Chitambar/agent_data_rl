"""Provisional in-hand pose of the SD reader for the confirmed grasp (2026-10-31; owner of the final
numbers: policy/pick).

Grasp: the right gripper reaches forward tipped `tilt_deg` from vertical (60: tool axis 30 deg below
horizontal), fingers closing horizontally on the reader's two side faces near the front of the body;
the reader is flat; the plug points along the reach (across the closing direction). With the reader at
yaw 0 and the plug along world +x: grasp_site y (finger opening) = world y, grasp_site z =
(sin tilt, 0, -cos tilt); the pad centroid is 9.6 mm behind the site along -z.

Reader frame (pc_bench.place_reader): origin at the plug tip, z along the part axis pointing back
from the tip (world -x at yaw 0), y up (world z), x = world -y. The body spans z 49.2..90.5 mm,
y +/-4.4 mm, x +/-17.4 mm.
"""
import math

import torch

PAD_BEHIND_SITE = .0096


def site_in_reader(tilt_deg=60., pad_z=.058, pad_y=0., device='cpu'):
    """(R, p): grasp_site orientation and position in the reader frame."""
    t = math.radians(tilt_deg)
    # world (reader yaw 0) -> reader coordinates: wx = -z_r, wy = -x_r, wz = y_r
    y = torch.tensor([-1., 0., 0.], device=device)                         # world +y
    z = torch.tensor([0., -math.cos(t), -math.sin(t)], device=device)      # (sin t, 0, -cos t) world
    x = torch.cross(y, z, dim=0)
    R = torch.stack([x, y, z], 1)
    pad = torch.tensor([0., pad_y, pad_z], device=device)
    return R, pad + z * PAD_BEHIND_SITE


def sample_inhand(n, tilt_deg=60., pad_z=.058, pos_mm=3., rot_deg=8., device='cpu', generator=None):
    """Per-world site-in-reader poses: nominal +/- pos_mm along the reader axis and vertically
    (the pads centre the reader along the closing axis) and +/- rot_deg about the closing axis
    (the reader pivots between the pads). Uniform. Returns R (n,3,3), p (n,3), offsets (n,3)."""
    from blind_insertion.envs import kinematics as K
    R0, p0 = site_in_reader(tilt_deg, pad_z, device=device)
    u = lambda *s: torch.rand(*s, device=device, generator=generator) * 2 - 1
    dpos = torch.zeros(n, 3, device=device)
    dpos[:, 1:] = u(n, 2) * pos_mm * 1e-3
    ang = u(n) * math.radians(rot_deg)
    # rotate the reader about the closing axis (reader x) through the pad centroid
    Rr = K.axis_angle_mat(torch.tensor([1., 0., 0.], device=device).expand(n, 3), ang)
    pad = (p0 - R0[:, 2] * PAD_BEHIND_SITE)
    # reader moves by (Rr, dpos) relative to the hand: site_in_reader' = Rr^T (site - pad) + pad - dpos
    R = Rr.transpose(1, 2) @ R0
    p = (Rr.transpose(1, 2) @ (p0 - pad)[:, None])[..., 0] + pad - dpos
    return R, p, torch.cat([dpos[:, 1:] * 1e3, torch.rad2deg(ang)[:, None]], 1)


# --- measured in-hand pose (Part 1, results/usbc/policy/pick/, read through reader_grasp.inhand_pose) ------
def _measured():
    import json
    from pathlib import Path
    import pandas as pd
    from blind_insertion.scenes import reader_grasp as G
    s = G.inhand_pose()
    spread = json.loads(G.SPREAD_FILE.read_text())
    src = Path(G.SPREAD_FILE).parent / spread['source']
    df = pd.read_parquet(src)
    df = df[df['success'].astype(bool)]
    cols = ['inhand_dx_mm', 'inhand_dy_mm', 'inhand_dz_mm', 'inhand_rx_deg', 'inhand_ry_deg', 'inhand_rz_deg']
    return s, df[cols].to_numpy()


def measured_mean(device='cpu'):
    """(R, p): grasp_site in the reader frame at the measured mean in-hand pose (55-deg grasp)."""
    from blind_insertion.envs import kinematics as K
    s, _ = _measured()
    R_rs = K.quat_to_mat(torch.tensor(s['quat_wxyz'], device=device, dtype=torch.float32)[None])[0]
    p_rs = torch.tensor(s['pos_m'], device=device, dtype=torch.float32)
    return R_rs.T, -R_rs.T @ p_rs


def sample_measured(n, device='cpu', generator=None):
    """Per-world site-in-reader poses drawn (with replacement) from the n = 2048 measured post-pick
    in-hand poses (expert gate, about the nominal, grasp_site frame). Returns R (n,3,3), p (n,3),
    deltas (n,6: dx, dy, dz mm, rx, ry, rz deg)."""
    from blind_insertion.envs import kinematics as K
    s, rows = _measured()
    nom = s['nominal'].get('55.0') or s['nominal']['tilt55_flip0']
    R0 = K.quat_to_mat(torch.tensor(nom['quat_wxyz'], device=device, dtype=torch.float32)[None])[0]
    p0 = torch.tensor(nom['pos_m'], device=device, dtype=torch.float32)
    rows = torch.tensor(rows, device=device, dtype=torch.float32)
    d = rows[torch.randint(len(rows), (n,), device=device, generator=generator)]
    rv = torch.deg2rad(d[:, 3:])
    dR = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
    R_rs = dR @ R0                                    # reader in site, rotated about the site frame
    p_rs = p0 + d[:, :3] * 1e-3
    R_sr = R_rs.transpose(1, 2)
    return R_sr, -(R_sr @ p_rs[..., None])[..., 0], d
