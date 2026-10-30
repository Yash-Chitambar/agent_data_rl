"""Wall-side backdrop of the duchamp3 rig, fitted to calibrated real camera frames.

Replaces the procedural workbench dressing (pegboard, tools, monitors, generic raceway and
power strip) with the objects the real ZED and wrist cameras see: the wall, the silver
raceway along the table's back edge, the plugs in its outlets, the power strip with its
cables, the black clip-on device on the wall and the Ryobi bit case. Visual only (no
collision). Positions are in the right arm base frame (+x toward the wall, z up). The
parameters in `configs/backdrop_duchamp3.json` come from `scripts/tools/fit_backdrop.py`.
"""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PARAMS = Path(__file__).resolve().parents[1] / 'configs' / 'backdrop_duchamp3.json'
# Procedural dressing that does not exist on the real bench (or is replaced here).
REPLACED = ('detail_wall', 'detail_raceway', 'detail_pegboard', 'detail_hole', 'detail_outlet', 'detail_socket',
            'detail_power', 'detail_switch', 'detail_bit', 'detail_hammer', 'detail_monitor', 'detail_screen',
            'detail_cable')
X_AXIS = (math.sqrt(.5), 0, math.sqrt(.5), 0)          # cylinder axis z -> x
N_CABLE = 10


def load():
    return json.loads(PARAMS.read_text())


def _quat_z(yaw):
    return (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))


def _quat_x(a):
    return (math.cos(a / 2), math.sin(a / 2), 0, 0)


def _bezier(p0, p1, p2, n):
    t = np.linspace(0, 1, n + 1)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def geoms(p, table_top):
    """[(name, type, pos, size, quat, colour key)] for parameter dict `p` (metres, radians)."""
    B, C = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    col = {k: k for k in p['colors']}
    out = []
    wall_x, front = p['wall_x'], p['raceway_front_x']
    out.append(('bd_wall', B, (wall_x + .025, .31, .5), (.025, 3., 1.3), (1, 0, 0, 0), col['wall']))
    bottom = table_top - .02
    out.append(('bd_raceway', B, ((wall_x + front) / 2, .31, (p['raceway_top_z'] + bottom) / 2),
                ((wall_x - front) / 2, 1.2, (p['raceway_top_z'] - bottom) / 2), (1, 0, 0, 0), col['raceway']))
    for i, y in enumerate(p['plug_y']):
        z = p['plug_z']
        out.append((f'bd_plate{i}', B, (front - .001, y, z), (.001, .036, .024), (1, 0, 0, 0), col['plate']))
        for k, dy in enumerate((-p['plug_spacing'] / 2, p['plug_spacing'] / 2)):
            out.append((f'bd_plug{i}_{k}', C, (front - p['plug_length'] / 2, y + dy, z),
                        (p['plug_radius'], p['plug_length'] / 2, 0), X_AXIS, col['black']))
    sx, sy, yaw, (sl, sw, sh) = p['strip_x'], p['strip_y'], p['strip_yaw'], p['strip_size']
    out.append(('bd_strip', B, (sx, sy, table_top + sh / 2), (sw / 2, sl / 2, sh / 2), _quat_z(yaw), col['black']))
    end = np.array([sx, sy, table_top + sh]) + np.array([math.sin(yaw), -math.cos(yaw), 0]) * (sl / 2 - .02)
    out.append(('bd_strip_switch', B, tuple(end + [0, 0, .001]), (.012, .009, .002), _quat_z(yaw), col['switch']))
    dx, dy, dz, tilt, dlen = p['device_x'], p['device_y'], p['device_z'], p['device_tilt'], p['device_length']
    out.append(('bd_device', B, (dx, dy, dz), (.008, .012, dlen / 2), _quat_x(tilt), col['black']))
    dev_bottom = np.array([dx, dy, dz]) - np.array([0, -math.sin(tilt), math.cos(tilt)]) * dlen / 2
    along = np.array([math.sin(yaw), -math.cos(yaw), 0])
    for c, (t0, lift) in enumerate(zip(p['cable_start'], p['cable_lift'])):
        p0 = np.array([sx, sy, table_top + sh + .004]) + along * t0 * sl / 2
        p2 = dev_bottom if c == 0 else np.array([p['cable_end'][c - 1][0], p['cable_end'][c - 1][1], table_top + .004])
        p1 = (p0 + p2) / 2 + [0, 0, lift]
        pts = _bezier(p0, p1, p2, N_CABLE)
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            mid, v = (a + b) / 2, b - a
            q = np.zeros(4); mujoco.mju_quatZ2Vec(q, v / max(np.linalg.norm(v), 1e-9))
            out.append((f'bd_cable{c}_{i}', mujoco.mjtGeom.mjGEOM_CAPSULE, tuple(mid),
                        (.0035, np.linalg.norm(v) / 2, 0), tuple(q), col['black']))
    ry, rz, (rw, rh) = p['ryobi_y'], p['ryobi_z'], p['ryobi_size']
    out.append(('bd_ryobi', B, (wall_x - .015, ry, rz), (.015, rw / 2, rh / 2), (1, 0, 0, 0), col['ryobi']))
    out.append(('bd_ryobi_label', B, (wall_x - .031, ry, rz - rh * .3), (.001, rw * .35, rh * .07), (1, 0, 0, 0), col['plate']))
    return out


def apply(spec, table_top, params=None):
    """Delete the procedural dressing it replaces and add the fitted backdrop (visual only).
    One material per colour key; `emission` lets the lit white wall reach the real exposure."""
    p = params or load()
    for g in [g for g in spec.worldbody.geoms if g.name.startswith(REPLACED)]:
        spec.delete(g)
    for key, rgba in p['colors'].items():
        spec.add_material(name='bd_' + key, rgba=rgba, emission=p.get('emission', {}).get(key, 0.), specular=.1, shininess=.1)
    for name, typ, pos, size, quat, key in geoms(p, table_top):
        spec.worldbody.add_geom(name=name, type=typ, pos=pos, size=size, quat=quat, material='bd_' + key,
                                contype=0, conaffinity=0, mass=0, group=0)


def update(m, p, table_top):
    """Write parameters into a compiled model in place (same geom list as `apply`)."""
    for key, rgba in p['colors'].items():
        mat = m.material('bd_' + key).id
        m.mat_rgba[mat], m.mat_emission[mat] = rgba, p.get('emission', {}).get(key, 0.)
    for name, _, pos, size, quat, _ in geoms(p, table_top):
        g = m.geom(name).id
        m.geom_pos[g], m.geom_quat[g] = pos, quat
        m.geom_size[g] = size
        m.geom_rbound[g] = float(np.linalg.norm(size)) * 2
