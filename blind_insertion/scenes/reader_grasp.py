"""Part 1 grasp of the SanDisk USB-C reader (confirmed by the user 2026-10-31, render
`results/usbc/policy/grasp/grasp_60deg_confirmed.png`).

The gripper reaches forward tipped TILT_DEG = 60 deg from vertical (tool axis 30 deg below the
horizontal), tipped about the finger-closing axis towards the plug, so the underside of the hand
faces the table. The fingers close horizontally on the reader's two side faces (8.8 mm tall) near
the front of the body, behind the neck. The reader stays flat; the plug points across the closing
direction. Reader at yaw 0 (plug along +x world): grasp_site y = world y, grasp_site z =
(sin 60, 0, -cos 60), x = y x z.

Frames (poses in the right arm's base frame, as in `pc_bench`):
  reader (part) frame: z along the part from the plug tip (z = 0) to the body end, y = thickness
      (up when flat), x = width (the closing direction). At yaw 0: z = world -x, y = world z,
      x = world -y.
  grasp_site: z = approach (out of the hand), y = finger closing axis, x = y x z.
  R_site_in_reader(tilt) = R_TOP_TCP rotated by `tilt` about the site y axis; tilt 0 is the old
  vertical top-down grasp.

Grasp point: x = 0 (centred between the side faces); height such that the lowest point of the open
fingertips is TIP_CLEARANCE above the table; along the part so that the pad spheres low enough to
touch the side face are centred AXIAL_BEHIND_NECK behind the body's front face.

`inhand_pose()` is the interface for Parts 2 and 4: the reader pose in the grasp_site frame after
the scripted pick (nominal geometry plus the measured spread in `results/usbc/policy/pick/`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from blind_insertion.scenes.pc_bench import R_TOP_TCP

TILT_DEG = 60.0               # tool axis from vertical, tipped towards the plug
TILT_FALLBACK_DEG = (55.0, 65.0, 50.0)   # tried in order where 60 deg does not reach
AXIAL_BEHIND_NECK = 0.007     # centre of the touching pad spheres behind the body's front face (m)
TIP_CLEARANCE = 0.001         # lowest fingertip point above the table at the grasp (m)
APPROACH_BACK = 0.06          # pre-grasp: back off this far along -site z ...
APPROACH_UP = 0.04            # ... and this far up (m)
LIFT_HEIGHT = 0.08            # lift: straight up (m)
SUPPORT_GAP = 1e-4            # place_reader puts the body 0.1 mm above the table
SPREAD_FILE = Path(__file__).resolve().parents[2] / 'results/usbc/policy/pick/inhand_spread.json'


def reader_dims(lay):
    """(body front z, body length, body half width, body half thickness) in the reader frame."""
    (cx, cy, cz), (hx, hy, hz) = lay.variants[0].peg_boxes[0]
    return cz - hz, 2 * hz, hx, hy


def site_rotation(tilt_deg=TILT_DEG):
    """grasp_site axes (columns) in the reader frame."""
    t = -math.radians(tilt_deg)
    Ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    # rotate the site about its own y: tilt > 0 swings site z (down) towards reader -z (the plug)
    return np.asarray(R_TOP_TCP, float) @ Ry


def hand_geometry(m, d, prefix=''):
    """Open-hand geometry in the grasp_site frame from a model and forward-kinematics data:
    tip-hull vertices (n, 3), pad-sphere centres (k, 3) and radius."""
    import mujoco  # noqa: PLC0415
    s = m.site(prefix + 'grasp_site').id
    R, p = d.site_xmat[s].reshape(3, 3), d.site_xpos[s]
    verts = []
    for f in ('left', 'right'):
        g = m.geom(prefix + f + '_tip_hull').id
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        verts.append(((V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - p) @ R)
    pads = [g for g in range(m.ngeom) if m.geom(g).name.startswith((prefix + 'left_pad_s', prefix + 'right_pad_s'))
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    return {'tip_vertices': np.concatenate(verts), 'pad_centres': (d.geom_xpos[pads] - p) @ R,
            'pad_radius': float(m.geom_size[pads[0], 0])}


def grasp_offset(lay, hand, tilt_deg=TILT_DEG, axial=AXIAL_BEHIND_NECK, tip_clearance=TIP_CLEARANCE):
    """grasp_site position in the reader frame (m) for a tilt; also returns the touching pads'
    height range above the table (m)."""
    front, _, _, ht = reader_dims(lay)
    Rs = site_rotation(tilt_deg)
    table_y = -ht - SUPPORT_GAP
    tip_h = (hand['tip_vertices'] @ Rs.T)[:, 1]
    y = table_y + tip_clearance - tip_h.min()
    pads = hand['pad_centres'] @ Rs.T
    pad_h = y + pads[:, 1]
    touch = (pad_h - hand['pad_radius'] < ht) & (pad_h + hand['pad_radius'] > -ht)
    if not touch.any():
        raise ValueError('no pad sphere reaches the side face at this tilt')
    z = front + axial - pads[touch, 2].mean()
    rng = (float(pad_h[touch].min() - table_y), float(min(pad_h[touch].max() + hand['pad_radius'], ht) - table_y))
    return np.array([0.0, y, z]), rng


def grasp_poses(p_reader, R_reader, offset, tilt_deg=TILT_DEG, back=APPROACH_BACK, up=APPROACH_UP, lift=LIFT_HEIGHT):
    """Pre-grasp, grasp and lift grasp_site poses (torch, batched). offset (N, 3) or (3,); tilt
    (N,) tensor or float. Pre-grasp is backed off along -site z and raised; lift is straight up."""
    import torch  # noqa: PLC0415
    dev, dt = p_reader.device, p_reader.dtype
    N = p_reader.shape[0]
    tilt = torch.as_tensor(tilt_deg, device=dev, dtype=dt).expand(N)
    t = -torch.deg2rad(tilt)
    c, s, z, o = torch.cos(t), torch.sin(t), torch.zeros_like(t), torch.ones_like(t)
    Ry = torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)
    Rs = torch.as_tensor(R_TOP_TCP, device=dev, dtype=dt) @ Ry
    off = torch.as_tensor(offset, device=dev, dtype=dt).expand(N, 3)
    p = p_reader + (R_reader @ off[..., None])[..., 0]
    R = R_reader @ Rs
    upv = torch.tensor([0., 0., 1.], device=dev, dtype=dt)
    return {'pregrasp': (p - back * R[:, :, 2] + up * upv, R), 'grasp': (p, R), 'lift': (p + lift * upv, R)}


def nominal_inhand(offset, tilt_deg=TILT_DEG):
    """Nominal reader pose in the grasp_site frame: (p (3,), R (3, 3))."""
    R = site_rotation(tilt_deg).T
    return -R @ np.asarray(offset), R


def _mat_to_quat(R):
    w = math.sqrt(max(0., 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0., 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0., 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0., 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def inhand_pose():
    """Reader pose in the grasp_site frame after a scripted pick, from SPREAD_FILE (written by
    `scripts/policy/pick_expert_eval.py --write_spread`): {'pos_m', 'quat_wxyz', 'nominal', 'spread'}.
    pos/quat are the measured mean; 'spread' gives mean/std/p1/p99 of the position (mm) and
    rotation vector (deg) about the nominal, in the grasp_site frame."""
    if not SPREAD_FILE.exists():
        raise FileNotFoundError(f'{SPREAD_FILE} missing: run scripts/policy/pick_expert_eval.py --write_spread')
    s = json.loads(SPREAD_FILE.read_text())
    return {'pos_m': s['mean_pos_m'], 'quat_wxyz': s['mean_quat_wxyz'], 'nominal': s['nominal'],
            'spread': s['about_nominal'], 'frame': 'reader (plug-tip origin, z to body end) in grasp_site', 'n': s['n']}
