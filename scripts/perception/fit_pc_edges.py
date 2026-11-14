"""PC pose (port 2 x, y in the right base, yaw) from the case top-face outline in a raw ZED HD1080 frame.

Classical fit, no net. It replaces the rig-laptop script of 2026-11-01 (not in git) that made data/zed_gt.
Docs: docs/real/ZED_MAP_20261114.md.

Model: the PC case top face, a rectangle at z 11.5 mm in the right base (sim table top -33 mm + 44.5 mm; the left
hand's press contact on the real top reads 11.5 mm). Two geometries:
  * 'real' (default): 103.5 mm along the panel x 103 mm deep, fitted on the real ZED frames (edge residuals ~0 px,
    docs/real/ZED_MAP_20261114.md); port 2 19.8 mm from the case centre toward the left (+y) edge, from the
    wrist-camera panel model (deploy/config/port_vision_model.json: L/R silhouettes at -28.8 / +68.5 mm).
    Output = the physical port 2 centre.
  * 'sim': the sim scene case (pc_bench.build_scene, measured profile v0): 102.5 x 97 mm, port 30.4 mm from the
    left edge. Output = the sim label convention (the ZED nets' labels).
Pose convention: fixture frame = port 2 frame, case behind the panel (+x world at yaw 0); yaw = the insertion
direction, the label of the ZED nets. Camera: the deploy calibration (blind_insertion/configs/real_cameras_duchamp3.json,
top, HD1080 rectified intrinsics, extrinsics_cam_in_right_base).

Method: for each of the 4 top edges, sample 15-85 % of its length (the corners are rounded), search along the
projected normal for the strongest bright-inside / dark-outside step on the smoothed image, then solve
(x, y, yaw) by robust least squares on point-to-line distances (pixels). Five rounds, search radius 40 -> 10 px.
Edge points in a mask (e.g. the left gripper) can be excluded with --mask (u0, v0, u1, v1 boxes).

    PYTHONPATH=. python scripts/perception/fit_pc_edges.py <frame.jpg|png> --init 0.229 0.395 4.5 [--geom real|sim] [--overlay out.jpg]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'blind_insertion/configs/real_cameras_duchamp3.json'
# fixture (port 2) frame extent of the case, from the sim scene (pc_bench.build_scene, measured profile v0):
A0, A1, DEPTH, Z_TOP = -0.0304019, 0.0720981, 0.097, 0.0115      # sim geometry
GEOMS = {'sim': dict(a0=A0, a1=A1, depth=DEPTH, z=Z_TOP),
         'real': dict(a0=-(0.1035 / 2 - 0.0198), a1=0.1035 / 2 + 0.0198, depth=0.103, z=Z_TOP)}
EDGES = ('panel', 'right', 'back', 'left')     # panel = port side (faces the bases); right = world -y side (image right)


def camera(spec=SPEC):
    s = json.loads(Path(spec).read_text())['top']
    k = s['intrinsics_rectified']['HD1080']
    K = np.array([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.]])
    e = s['extrinsics_cam_in_right_base']
    w, x, y, z = np.asarray(e['quat'], float) / np.linalg.norm(e['quat'])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return K, R, np.asarray(e['pos'], float)


def project(P, cam):
    K, R, t = cam
    c = (np.atleast_2d(P) - t) @ R            # world -> optical frame (R maps optical to world)
    uv = c @ K.T
    return uv[:, :2] / uv[:, 2:3]


def corners(pose, a0=A0, a1=A1, depth=DEPTH, z=Z_TOP):
    """Top-face corners (world, 4 x 3) in edge order: panel (a0->a1), right (a1, 0->depth), back, left."""
    x, y, th = pose
    s, c = math.sin(th), math.cos(th)
    ea, ed = np.array([s, -c]), np.array([c, s])     # along the panel (fixture x), into the case (-fixture z)
    q = [(a0, 0), (a1, 0), (a1, depth), (a0, depth)]
    return np.array([[*(np.array([x, y]) + a * ea + d * ed), z] for a, d in q])


def edge_samples(pose, cam, n=40, lo=.15, hi=.85, geom=None):
    """Per edge: world sample points, image points and outward unit normals (image)."""
    C = corners(pose, **(geom or GEOMS['real']))
    ctr = C.mean(0)
    out = []
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        t = np.linspace(lo, hi, n)[:, None]
        P = a + t * (b - a)
        mid = (a + b) / 2
        nrm = mid - ctr; nrm[2] = 0; nrm /= np.linalg.norm(nrm)
        uv = project(P, cam)
        uv2 = project(P + .002 * nrm, cam)
        nv = uv2 - uv; nv /= np.linalg.norm(nv, axis=1, keepdims=True)
        out.append((P, uv, nv))
    return out


def detect(gray, uv, nv, radius, step=.5, min_grad=4.):
    """Strongest bright-inside step along each normal; returns pixel points (nan where none)."""
    ts = np.arange(-radius, radius + step, step)
    pts = uv[:, None, :] + ts[None, :, None] * nv[:, None, :]
    vals = cv2.remap(gray, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    der = -(np.gradient(vals, step, axis=1))                     # + where intensity drops going outward
    k = der.argmax(1)
    ok = der[np.arange(len(k)), k] > min_grad
    kk = np.clip(k, 1, len(ts) - 2)
    y0, y1, y2 = (der[np.arange(len(k)), kk + j] for j in (-1, 0, 1))
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-9, .5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0.)
    t = ts[kk] + np.clip(off, -1, 1) * step
    p = uv + t[:, None] * nv
    p[~ok] = np.nan
    return p, der[np.arange(len(k)), k]


def fit(img, init, cam=None, geom=None, mask=(), use=EDGES, rounds=((40, 3.), (25, 2.), (15, 1.5), (10, 1.), (10, 1.)), n=40):
    """img: BGR or gray 1920x1080 rectified-left frame. init: (x m, y m, yaw deg). Returns a dict."""
    cam = cam or camera()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    pose = np.array([init[0], init[1], math.radians(init[2])], float)
    for radius, fscale in rounds:
        es = edge_samples(pose, cam, n=n, geom=geom)
        obs = []
        for name, (P, uv, nv) in zip(EDGES, es):
            p, g = detect(gray, uv, nv, radius)
            good = np.isfinite(p[:, 0])
            for (u0, v0, u1, v1) in mask:
                good &= ~((p[:, 0] >= u0) & (p[:, 0] <= u1) & (p[:, 1] >= v0) & (p[:, 1] <= v1))
            if name not in use:
                good[:] = False
            obs.append((name, p, good))

        def res(q):
            r = []
            for (name, p, good), (P, uv, nv) in zip(obs, edge_samples(q, cam, n=n, geom=geom)):
                if good.any():
                    r.append(np.sum((p[good] - uv[good]) * nv[good], 1))   # signed normal distance (the line is straight)
            return np.concatenate(r) if r else np.zeros(3)
        sol = least_squares(res, pose, loss='soft_l1', f_scale=fscale, x_scale=[.001, .001, .01])
        pose = sol.x
    r_all = res(pose)
    per = {}
    k = 0
    for name, p, good in obs:
        m = int(good.sum())
        rr = r_all[k:k + m]; k += m
        per[name] = dict(n=m, inl=int((np.abs(rr) < 3).sum()), rms_px=float(np.sqrt(np.mean(rr ** 2))) if m else None,
                         med_px=float(np.median(rr)) if m else None)
    J = sol.jac
    inl = np.abs(r_all) < 3
    s2 = float(np.mean(r_all[inl] ** 2)) if inl.any() else float('nan')
    try:
        cov = np.linalg.inv(J[inl].T @ J[inl]) * s2
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(3, np.nan)
    return dict(pose=[float(pose[0]), float(pose[1]), math.degrees(pose[2])], sd_stat=[float(sd[0]), float(sd[1]), math.degrees(sd[2])],
                edges=per, inlier_rms_px=math.sqrt(s2) if s2 == s2 else None, obs=obs)


def overlay(img, result, cam=None, geom=None, crop=None, scale=1.0, label=''):
    cam = cam or camera()
    im = img.copy()
    pose = np.array([result['pose'][0], result['pose'][1], math.radians(result['pose'][2])])
    C = project(corners(pose, **(geom or GEOMS['real'])), cam)
    cv2.polylines(im, [np.round(C).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1, cv2.LINE_AA)
    port = project(np.array([[pose[0], pose[1], Z_TOP]]), cam)[0]
    cv2.drawMarker(im, tuple(int(v) for v in np.round(port)), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
    for name, p, good in result['obs']:
        for q, g in zip(p, good):
            if np.isfinite(q[0]):
                cv2.circle(im, (int(round(q[0])), int(round(q[1]))), 2, (255, 200, 0) if g else (0, 0, 255), -1)
    if crop is None:
        c = C.mean(0)
        crop = (int(c[0]) - 230, int(c[1]) - 170, int(c[0]) + 230, int(c[1]) + 190)
    u0, v0, u1, v1 = crop
    im = im[max(v0, 0):v1, max(u0, 0):u1]
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(im, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frame')
    ap.add_argument('--init', nargs=3, type=float, required=True, help='port x m, y m, yaw deg')
    ap.add_argument('--geom', default='real', choices=sorted(GEOMS))
    ap.add_argument('--z-top', type=float, default=None)
    ap.add_argument('--overlay')
    a = ap.parse_args()
    img = cv2.imread(a.frame)
    g = dict(GEOMS[a.geom], **({} if a.z_top is None else {'z': a.z_top}))
    r = fit(img, a.init, geom=g)
    print(json.dumps({k: v for k, v in r.items() if k != 'obs'}, indent=1))
    if a.overlay:
        cv2.imwrite(a.overlay, overlay(img, r, geom=g), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
