"""Camera, meshes and pose conversions for the ZED FoundationPose estimator (numpy / OpenCV / trimesh only).

Frames (as the rig logs them): right base = the right arm's base frame. ZED = rectified-left optical frame (OpenCV)
of the HD1080 stream, pose `top.extrinsics_cam_in_right_base` of blind_insertion/configs/real_cameras_duchamp3.json.

PC mesh frame ('port frame', the convention of scripts/perception/fit_pc_edges.py and the ZED nets):
origin = port 2 centre (x, y) at the case top-face height, +x = into the case (the insertion direction, yaw 0 = +x of
the base), +y = toward the left panel edge, +z = up. So T_pc_in_base = [Rz(yaw) | (port_x, port_y, z_top)].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / 'blind_insertion/configs/real_cameras_duchamp3.json'

# Real case (docs/real/ZED_MAP_20261114.md "Real case geometry"; fit_pc_edges GEOMS['real']): 103.5 mm along the panel,
# 103 mm deep, port 2 19.8 mm from the case centre toward the left edge (31.95 mm from the left edge; operator
# calipers about 32 mm), top face at z 11.5 mm in the right base. Height with feet 44.5 mm (rig profile v0, calipers).
PC_REAL = dict(width=0.1035, depth=0.103, port_from_left=0.1035 / 2 - 0.0198, height=0.0445, z_top=0.0115,
               corner_r=0.006, feet=0.0033)
# Colours (RGB 0-255) sampled from real ZED frames (data/zed_gt spot1 / spot2, edge-fit pose, inner 80 % of each face).
PC_RGB = dict(top=(205, 205, 195), side=(200, 200, 192), panel=(112, 110, 110), feet=(30, 30, 30), bottom=(40, 40, 40))


def load_spec(spec=SPEC):
    return json.loads(Path(spec).read_text())


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def zed_camera(spec=None, mode='HD1080'):
    """K (3x3), T_cam_in_base (4x4, OpenCV optical frame of the rectified left image), baseline (m)."""
    s = (spec or load_spec())['top']
    k = s['intrinsics_rectified'][mode]
    K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.]])
    e = s['extrinsics_cam_in_right_base']
    T = np.eye(4); T[:3, :3] = quat_to_R(e['quat']); T[:3, 3] = e['pos']
    return K, T, s['baseline_mm'] * 1e-3


def project(P_base, K, T_cam_in_base):
    P = np.atleast_2d(P_base)
    c = (P - T_cam_in_base[:3, 3]) @ T_cam_in_base[:3, :3]
    uv = c @ K.T
    return uv[:, :2] / uv[:, 2:3]


def pose_T(x, y, yaw_deg, z=PC_REAL['z_top']):
    t = math.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def T_to_xyyaw(T):
    """(x m, y m, yaw deg, z m, tilt deg) of a port-frame pose in the base. tilt = angle of the mesh z axis from vertical."""
    R = T[:3, :3]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
    return float(T[0, 3]), float(T[1, 3]), yaw, float(T[2, 3]), tilt


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _rounded_rect(x0, x1, y0, y1, r, nseg=6):
    """CCW polygon (seen from +z) with rounded corners; returns points and a per-edge tag (the straight side it lies on)."""
    pts, tags = [], []
    corners = [((x1 - r, y0 + r), -90, 'c'), ((x1 - r, y1 - r), 0, 'c'), ((x0 + r, y1 - r), 90, 'c'), ((x0 + r, y0 + r), 180, 'c')]
    for (cx, cy), a0, _ in corners:
        for i in range(nseg + 1):
            a = math.radians(a0 + 90 * i / nseg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def pc_real_mesh(geom=None, rgb=None, nseg=6):
    """Real PC case as a closed, vertex-coloured trimesh in the port frame (see module doc). Panel face = x 0 (dark)."""
    import trimesh
    g = dict(PC_REAL, **(geom or {}))
    c = dict(PC_RGB, **(rgb or {}))
    W, D, H, r = g['width'], g['depth'], g['height'], g['corner_r']
    y_left = g['port_from_left']                 # left edge at +y
    y_right = y_left - W
    zt, zb = 0.0, -H + g['feet']                 # top face at 0; body bottom above the feet
    poly = _rounded_rect(0.0, D, y_right, y_left, r, nseg)
    n = len(poly)
    V, F, Cv = [], [], []

    def add(tri_pts, col):
        k = len(V)
        V.extend(tri_pts); F.append([k, k + 1, k + 2]); Cv.extend([col] * 3)

    ctr_t = np.array([D / 2, (y_left + y_right) / 2, zt]); ctr_b = ctr_t.copy(); ctr_b[2] = zb
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        a_t, b_t = np.array([*p, zt]), np.array([*q, zt])
        a_b, b_b = np.array([*p, zb]), np.array([*q, zb])
        add([ctr_t, a_t, b_t], c['top'])
        add([ctr_b, b_b, a_b], c['bottom'])
        mid = (p + q) / 2
        nrm = np.array([q[1] - p[1], -(q[0] - p[0])])          # outward for a CCW polygon
        col = c['panel'] if (abs(mid[0]) < 1e-6 and nrm[0] < 0) else c['side']
        add([a_b, b_b, b_t], col); add([a_b, b_t, a_t], col)
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    m.visual.vertex_colors = np.hstack([np.array(Cv, np.uint8), np.full((len(Cv), 1), 255, np.uint8)])
    return m


def mesh_mask(mesh, T_obj_in_cam, K, hw):
    """Silhouette of `mesh` at T_obj_in_cam (pinhole, no occlusion by the scene): uint8 0/1, for IoU and fallback masks."""
    import cv2
    H, W = hw
    v = mesh.vertices @ T_obj_in_cam[:3, :3].T + T_obj_in_cam[:3, 3]
    uv = v @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    out = np.zeros((H, W), np.uint8)
    tris = np.round(uv[mesh.faces] * 8).astype(np.int32)       # 3 bits of sub-pixel precision
    for t in tris:                                              # one call per triangle: fillPoly would XOR overlaps
        cv2.fillConvexPoly(out, t, 1, lineType=cv2.LINE_8, shift=3)
    return out


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float('nan')


def prior_box(mesh, T_obj_in_base, K, T_cam_in_base, margin_px=12):
    """Image box (u0, v0, u1, v1) of the mesh at a prior pose, with a margin: the SAM 3 box prompt / detection choice."""
    T = np.linalg.inv(T_cam_in_base) @ T_obj_in_base
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    uv = v @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    return (float(uv[:, 0].min() - margin_px), float(uv[:, 1].min() - margin_px),
            float(uv[:, 0].max() + margin_px), float(uv[:, 1].max() + margin_px))


def largest_cc(mask):
    """Largest 8-connected component of a 0/1 mask (render segmentation has stray pixels at far geoms)."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)
