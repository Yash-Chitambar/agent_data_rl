"""Real in-hand pose and real reader look (preset configs/inhand_real_<name>.json; off by default).

Measured on the calibrated right wrist camera, 13 real grasps of 2026-11-02 (scripts/real2sim/inhand_measure.py,
docs/real/INHAND_REAL.md). Same delta convention as inhand.sample_measured and InsertionTask.inhand_error: reader in the
grasp_site frame minus the Part 1 nominal grasp, mm and rotation vector deg on the TCP axes.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ROOT / 'blind_insertion/configs'
PAD_BEHIND_SITE = .0096      # pad centroid behind grasp_site along z (m)


def load(name='v1'):
    """Preset dict: a name (configs/inhand_real_<name>.json) or a path."""
    p = PRESETS / f'inhand_real_{name}.json'
    return json.loads((p if p.exists() else ROOT / name).read_text())


def sample_delta(n, preset='v1', device='cpu', generator=None, spread=1.):
    """(n, 6): deltas, the measured mean plus a uniform draw in +/- half_range x spread. spread 0: the mean."""
    c = load(preset)['inhand'] if isinstance(preset, str) else preset
    if 'rows' in c:      # one measured grasp per draw: keeps the measured coupling of x and pitch (r = -0.88 over grasps)
        rows = torch.tensor(c['rows'], device=device, dtype=torch.float32)
        mu = rows.mean(0)
        i = torch.randint(len(rows), (n,), device=device, generator=generator)
        lo, hi = c['stretch']
        k = lo + (hi - lo) * torch.rand(n, 1, device=device, generator=generator)
        z = torch.randn(n, 6, device=device, generator=generator) * torch.tensor(c['jitter_mm'] + c['jitter_deg'], device=device)
        z[:, 1] = 0
        return mu + (rows[i] - mu) * k * spread + z * spread
    mu = torch.tensor(c['mean_mm'] + c['mean_rotvec_deg'], device=device, dtype=torch.float32)
    h = torch.tensor(c['half_range_mm'] + c['half_range_deg'], device=device, dtype=torch.float32)
    return mu + (torch.rand(n, 6, device=device, generator=generator) * 2 - 1) * h * spread


def reader_in_site(d, device='cpu', lateral_mm=0.):
    """deltas (n, 6) -> (R_rs (n, 3, 3), p_rs (n, 3)) reader in grasp_site. lateral_mm: added on the closing axis after
    the centring (rendering only: the real reader looks this far off the sim's centre; not separated from a camera offset)."""
    from blind_insertion.envs import kinematics as K
    from blind_insertion.envs.inhand import nominal_reader_in_site
    R0, p0 = nominal_reader_in_site(device)
    rv = torch.deg2rad(d[:, 3:])
    dR = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
    R, p = dR @ R0, p0 + d[:, :3] * 1e-3
    # the coupled pads centre the reader: its mid-plane passes through the pad centroid (as InsertionTask starts)
    pad = torch.tensor([0., 0., -PAD_BEHIND_SITE], device=device, dtype=p.dtype)
    x_r = (R.transpose(1, 2) @ (pad - p)[..., None])[..., 0]
    x_r[:, 0] = 0
    cy = ((R @ x_r[..., None])[..., 0] + p)[:, 1]
    return R, p - torch.stack([torch.zeros_like(cy), cy - lateral_mm * 1e-3, torch.zeros_like(cy)], 1)


def sample_real(n, preset='v1', device='cpu', generator=None, spread=1.):
    """Site in reader, as inhand.sample_measured: R_sr (n, 3, 3), p_sr (n, 3), deltas (n, 6)."""
    d = sample_delta(n, preset, device, generator, spread)
    c = load(preset)['inhand'] if isinstance(preset, str) else preset
    R_rs, p_rs = reader_in_site(d, device, c.get('render_lateral_mm', 0.))
    R_sr = R_rs.transpose(1, 2)
    return R_sr, -(R_sr @ p_rs[..., None])[..., 0], d
