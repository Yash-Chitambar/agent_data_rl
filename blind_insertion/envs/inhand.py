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


# --- wider in-hand poses after an estimate-driven pick (chain, round 8) -------------------------------
CHAIN_PICKEST = ('results/usbc/policy/chain/pickest_1_1.5.csv', 'results/usbc/policy/chain/pickest_2_3.csv',
                 'results/usbc/policy/chain/pickest_4_5.csv')


def _rows_chain():
    import pandas as pd
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    cols = ['pick_inhand_dx_mm', 'pick_inhand_dy_mm', 'pick_inhand_dz_mm', 'pick_inhand_rx_deg', 'pick_inhand_ry_deg', 'pick_inhand_rz_deg']
    dfs = [pd.read_csv(root / f) for f in CHAIN_PICKEST if (root / f).exists()]
    d = pd.concat(dfs)
    d = d[d['pick_ok'] == 1]
    return d[cols].to_numpy()


def _site_in_reader_from_deltas(d, device):
    from blind_insertion.envs import kinematics as K
    s, _ = _measured()
    nom = s['nominal'].get('55.0') or s['nominal']['tilt55_flip0']
    R0 = K.quat_to_mat(torch.tensor(nom['quat_wxyz'], device=device, dtype=torch.float32)[None])[0]
    p0 = torch.tensor(nom['pos_m'], device=device, dtype=torch.float32)
    rv = torch.deg2rad(d[:, 3:])
    dR = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
    R_rs = dR @ R0
    p_rs = p0 + d[:, :3] * 1e-3
    R_sr = R_rs.transpose(1, 2)
    return R_sr, -(R_sr @ p_rs[..., None])[..., 0]


def sample_chain_est(n, device='cpu', generator=None, pos_mm=4., rot_deg=5., uniform_frac=0., stretch=(.8, 1.3)):
    """Wide in-hand poses for the wrist estimators (round 8): drawn from the chain's estimate-driven picks
    (pickest_*.csv, successful picks, noise levels 1-5 mm / 1.5-5 deg; settled grasps), each deviation
    from the mean scaled by U(stretch) to widen the tails, about the nominal grasp (sample_measured
    convention). uniform_frac > 0 mixes in uniform +/- pos_mm / +/- rot_deg on all axes: NOT physical for the
    parallel pads (teleported, those readers slip 10 mm in 1 s), kept off by default.
    Returns R (n,3,3), p (n,3), deltas (n,6)."""
    rows = torch.tensor(_rows_chain(), device=device, dtype=torch.float32)
    d = rows[torch.randint(len(rows), (n,), device=device, generator=generator)]
    mu = rows.mean(0)
    k = torch.rand(n, 1, device=device, generator=generator) * (stretch[1] - stretch[0]) + stretch[0]
    d = mu + (d - mu) * k
    u = (torch.rand(n, 6, device=device, generator=generator) * 2 - 1) * torch.tensor([pos_mm] * 3 + [rot_deg] * 3, device=device)
    uni = torch.rand(n, device=device, generator=generator) < uniform_frac
    d = torch.where(uni[:, None], u, d)
    R, p = _site_in_reader_from_deltas(d, device)
    return R, p, d


def nominal_reader_in_site(device='cpu'):
    """(R_rs0, p_rs0): the nominal Part 1 grasp, reader (plug-tip frame) in grasp_site (reader_grasp nominal)."""
    from blind_insertion.envs import kinematics as K
    s, _ = _measured()
    nom = s['nominal'].get('55.0') or s['nominal']['tilt55_flip0']
    R0 = K.quat_to_mat(torch.tensor(nom['quat_wxyz'], device=device, dtype=torch.float32)[None])[0]
    return R0, torch.tensor(nom['pos_m'], device=device, dtype=torch.float32)


def inhand_error6(site_p, site_R, reader_p, reader_R):
    """(n, 6) in-hand pose minus the nominal grasp, as Part 4 variant J reads it (InsertionTask.inhand_error):
    reader-in-TCP position minus nominal in mm on the TCP axes, then the rotation vector of R_rs @ R_nom^T
    in deg on the TCP axes."""
    from blind_insertion.envs import kinematics as K
    R0, p0 = nominal_reader_in_site(site_p.device)
    R_rs = site_R.transpose(1, 2) @ reader_R
    p_rs = (site_R.transpose(1, 2) @ (reader_p - site_p)[..., None])[..., 0]
    return torch.cat([(p_rs - p0) * 1e3, torch.rad2deg(K.rot_log(R_rs @ R0.T))], 1)


def sample_real_fit(n, tip_shift_tcp_mm, pos_mm=3., rot_deg=5., device='cpu', generator=None):
    """In-hand poses around the real grasp as seen in the wrist frames (results/real/wrist_nets/geometry_fit.json): the measured mean
    grasp with the plug tip moved by tip_shift_tcp_mm (TCP axes), then U(+/-pos_mm) per axis on the tip and U(+/-rot_deg) per axis
    about the tip. Returns R (n,3,3), p (n,3): grasp_site in the reader frame, deltas (n,6)."""
    from blind_insertion.envs import kinematics as K
    R0, p0 = measured_mean(device=device)
    R0, p0 = R0.reshape(3, 3).float(), p0.reshape(3).float()
    shift = torch.as_tensor(tip_shift_tcp_mm, device=device, dtype=torch.float32)
    u = lambda a, k: (torch.rand(n, k, device=device, generator=generator) * 2 - 1) * a   # noqa: E731
    d = torch.cat([shift + u(pos_mm, 3), u(rot_deg, 3)], 1)
    rv = torch.deg2rad(d[:, 3:])
    dR = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
    # R_sr = R0 dR: rotation of the reader in the TCP about the tip; tip in the TCP = -R_sr^T p_sr, so it is kept by rotating p with R_sr
    tip_in_tcp = -(R0.T @ p0) + d[:, :3] * 1e-3
    R_sr = R0 @ dR
    p_sr = -(R_sr @ tip_in_tcp[..., None])[..., 0]
    return R_sr, p_sr, d


SERVO_PICK = 'results/usbc/policy/approach/inhand_servo_pick.csv'


def sample_servo_pick(n, device='cpu', generator=None, stretch=(.8, 1.4)):
    """In-hand poses after the chain's realistic servo-descent pick (round 9): the true in-hand pose at the
    approach stop of policy/chain chain_real_servo_bf1.5_n2048 (n = 2043 successful picks; inhand_true_*,
    InsertionTask.inhand_error convention), deviations from the mean scaled by U(stretch) (widened ~x1.3)."""
    import pandas as pd
    from pathlib import Path
    rows = torch.tensor(pd.read_csv(Path(__file__).resolve().parents[2] / SERVO_PICK).to_numpy(), device=device, dtype=torch.float32)
    d = rows[torch.randint(len(rows), (n,), device=device, generator=generator)]
    mu = rows.mean(0)
    k = torch.rand(n, 1, device=device, generator=generator) * (stretch[1] - stretch[0]) + stretch[0]
    d = mu + (d - mu) * k
    R, p = _site_in_reader_from_deltas(d, device)
    return R, p, d
