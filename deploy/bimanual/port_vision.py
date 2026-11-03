"""Classical port and plug measurement from one raw right wrist frame. No net, no encoders, no in-hand estimate.

    from deploy.bimanual.port_vision import measure
    m = measure(raw_bgr)            # 1280x1024 BGR frame of the right wrist camera
    if m.ok: m.across_mm, m.up_mm, m.gap_mm

Axes and signs match `p_mm` of data/real_training/port_v1/manifest.jsonl: plug tip relative to port 2 centre on the panel,
x across (right in the wrist view), y up, z gap (out of the panel). Method and limits: docs/real/PORT_VISION.md.

Pixels to mm: the port row gives the camera pose against the panel by PnP on a stored 3D model of the row (PANEL_MM; a long
baseline across the row sets the scale, port 2 itself is not needed). The plug tip is a ray through the shell's tip edge,
at the depth given by the shell's width in pixels (8.25 mm USB-C shell). The gap is the tip's distance to the panel plane.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

SPEC = Path(__file__).resolve().parents[2] / 'blind_insertion/configs/real_cameras_duchamp3.json'
MODEL = Path(__file__).resolve().parent.parent / 'config' / 'port_vision_model.json'

SHELL_W_MM = 8.25           # USB-C plug shell width (USB Type-C specification)
SHELL_T_MM = 2.40           # USB-C plug shell thickness

# Panel feature model: (x across, y up) mm on the panel, origin at port 2 centre. First estimate; the stored file
# deploy/config/port_vision_model.json (fitted by scripts/real/port_vision_fit.py) replaces it. wh = bright blob size.
PANEL_INIT = {
    'P1': dict(xy=(-14.0, 0.0), wh=(8.8, 2.4)),
    'P2': dict(xy=(0.0, 0.0), wh=(8.8, 2.4)),
    'S':  dict(xy=(-9.5, -5.0), wh=(6.5, 1.6)),
    'A':  dict(xy=(12.5, -5.0), wh=(6.4, 7.0)),
    'RJ': dict(xy=(26.0, -5.0), wh=(17.0, 8.0)),
    'H':  dict(xy=(47.0, -3.5), wh=(15.5, 4.0)),
    'U':  dict(xy=(42.5, -8.0), wh=(15.5, 4.5)),
}
# What a good fit requires. Conservative on purpose: a miss is cheap, a wrong number is not.
MIN_FEATURES = 4
PLUG_Z_MM = 118.5           # shell depth from the camera: median of its width over the collect sweep (8.25 mm shell). One grasp type.
PLUG_Z_TOL_MM = 5.0         # a shell further from this depth is another grasp: refuse
TIP_ABOVE_END_PX = 32.0     # shell tip edge above the sleeve end in the image (rigid; median of the collect sweep)
SIGMA_MM = (0.4, 1.0, 0.8)  # residual SD after a constant, held-out collect blocks; see docs/real/PORT_VISION.md
POSE_ITERS = 2
POSE_FEATURES = ('P1', 'P2', 'S')     # clean slots; the frames of the larger ports only confirm the pattern
MAX_REPROJ_PX = 3.0
SCALE_PX_MM = (1.8, 9.0)    # plausible image scale at the panel
PLUG_ROI = (575, 720, 470, 565)     # u0, u1, v0, v1: undistorted pixels. The plug is rigid in the hand.
PLUG_W_PX = (40.0, 60.0)
PLUG_H_PX = (22.0, 46.0)
MIN_EDGE = 10.0             # smallest edge step in the mean profile (grey levels)


@dataclasses.dataclass
class PortMeasure:
    ok: bool
    reason: str = ''
    conf: float = 0.0
    port_px: tuple | None = None      # port 2 centre, undistorted pixels
    plug_px: tuple | None = None      # plug tip centre, undistorted pixels
    across_mm: float | None = None
    up_mm: float | None = None
    gap_mm: float | None = None
    sigma_mm: tuple | None = None     # estimated 1-sigma (across, up, gap)
    debug: dict = dataclasses.field(default_factory=dict)


def _camera(side):
    w = json.loads(SPEC.read_text())['wrist'][side]
    i = w['intrinsics']
    K = np.array([[i['fx'], 0, i['cx']], [0, i['fy'], i['cy']], [0, 0, 1.]])
    return K, np.array(w['distortion'], float)


_MAPS = {}


def undistort(raw, side='right'):
    """Full frame undistorted with the calibrated intrinsics, same K. Returns (image, K)."""
    if side not in _MAPS:
        K, D = _camera(side)
        _MAPS[side] = (K, *cv2.initUndistortRectifyMap(K, D, None, K, (1280, 1024), cv2.CV_16SC2))
    K, a, b = _MAPS[side]
    return cv2.remap(raw, a, b, cv2.INTER_LINEAR), K


_MODEL = {}


def load_model(reload=False):
    if reload or not _MODEL:
        _MODEL.clear()
        if MODEL.exists():
            _MODEL.update(json.loads(MODEL.read_text()))
        else:
            _MODEL.update(panel={k: dict(xyz=[*v['xy'], 0.0], wh=list(v['wh'])) for k, v in PANEL_INIT.items()},
                          R_view=[[1, 0, 0], [0, -1, 0], [0, 0, -1]], plug={}, lines=dict(band=dict(y=24.0, z=0.0), bottom=dict(y=-18.0, z=0.5), L=dict(x=-30.0, z=0.0), R=dict(x=73.0, z=0.0)))
    return _MODEL


def _view_prior():
    return load_model()['R_view']


# ---------------------------------------------------------------------------------- plug
def _edge(profile, lo, hi, sign):
    """Strongest step of `sign` in profile[lo:hi] with parabola refinement. Returns (position, step size)."""
    g = np.convolve(profile, [1, 1, 0, -1, -1], mode="same") / 2.0     # rising step > 0: levels per step width
    lo, hi = max(lo, 3), min(hi, len(g) - 3)
    if hi <= lo:
        return None, 0.0
    i = lo + int(np.argmax(sign * g[lo:hi]))
    a, b, c = sign * g[i - 1], sign * g[i], sign * g[i + 1]
    d = a - 2 * b + c
    return i + (0.5 * (a - c) / d if abs(d) > 1e-6 else 0.0), float(b)


def detect_plug(und):
    """Metal shell in its fixed image window. Returns dict(ul, ur, vt, ve, ...) in undistorted pixels, or (None, reason)."""
    u0, u1, v0, v1 = PLUG_ROI
    g = cv2.GaussianBlur(cv2.cvtColor(und[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    # sleeve end: bright shell over the dark sleeve, the strongest falling step on the middle columns
    c0, c1 = 55, 90
    prof = g[:, c0:c1].mean(1)
    ve, se = _edge(prof, 40, v1 - v0 - 8, -1)
    if ve is None or se < MIN_EDGE:
        return None, f'no shell end edge (step {se:.0f})'
    # side edges on the rows above the sleeve end
    r1 = int(ve) - 5
    r0 = r1 - 18
    if r0 < 0:
        return None, 'shell end too high in the window'
    side = g[r0:r1].mean(0)
    ul, sl = _edge(side, 15, 70, +1)
    ur, sr = _edge(side, 70, u1 - u0 - 15, -1)
    if ul is None or ur is None or min(sl, sr) < MIN_EDGE:
        return None, f'no shell side edges (steps {sl:.0f}, {sr:.0f})'
    w = ur - ul
    if not PLUG_W_PX[0] <= w <= PLUG_W_PX[1]:
        return None, f'shell width {w:.1f} px out of range'
    # tip: rising step going down, above the sleeve end, on the rims and the middle
    a, b = int(ul) + 4, int(ur) - 3
    top = g[:, a:b].mean(1)
    vt, st = _edge(top, max(int(ve) - int(PLUG_H_PX[1]), 3), int(ve) - int(PLUG_H_PX[0]) + 1, +1)
    if vt is None or st < 5.0:
        return None, f'no shell tip edge (step {st:.0f})'
    return dict(ul=ul + u0, ur=ur + u0, vt=vt + v0, ve=ve + v0, w=w, steps=(se, sl, sr, st)), ''


def detect(und, K, panel, lines=None, R_prior=None):
    """All image features of one undistorted frame. row: {feature: (u, v, w, h)} or None."""
    plug, why = detect_plug(und)
    box = None
    if plug is not None:
        box = (int(plug['ul']) - 3, int(plug['vt']) - 3, int(plug['ur']) + 4, int(plug['ve']) + 2)
    blobs = bright_blobs(und, box)
    m, score = match_row(blobs, panel)
    row = {k: blobs[i][:2] + blobs[i][2:4] for k, i in m.items()} if m else None
    lp = {}
    if row is not None and lines and R_prior is not None:
        t0 = t_given_R(row, K, R_prior, panel)
        lp = detect_lines(und, K, R_prior, t0, lines, box)
    return dict(K=K, plug=plug, plug_why=why, blobs=blobs, row=row, score=score, lines=lp, box=box)


# ---------------------------------------------------------------------------------- port row blobs
def bright_blobs(und, plug_box=None):
    """Bright compact blobs (slots, port frames) on the panel, blue channel so gold fingers and red labels stay out."""
    b = und[:, :, 0]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    th = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, ker)
    m = (th > 55).astype(np.uint8)
    if plug_box is not None:
        u0, v0, u1, v1 = plug_box
        m[v0:v1, u0:u1] = 0
    n, _, st, c = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 40 or w < 12 or y + h > 800 or w > 180 or h > 60:
            continue
        out.append((float(x + (w - 1) / 2), float(y + (h - 1) / 2), float(w), float(h), float(a)))
    return out


def _match_xy(panel):
    """Model positions as the camera sees them at the nominal view: (x right, y down) mm, rotated by the stored view prior."""
    names = list(panel)
    X = np.array([panel[k]['xyz'] for k in names], float)
    R = np.array(_view_prior(), float)
    return names, (R @ X.T).T[:, :2]


def match_row(blobs, panel):
    """Pattern match: blobs <-> panel features under a 2D similarity from any pair, then an affine refit.
    Returns {feature: blob index} or None."""
    names, M = _match_xy(panel)
    WH = np.array([panel[k]['wh'] for k in names], float)
    if len(blobs) < MIN_FEATURES:
        return None, 0.0
    C = np.array([b[:2] for b in blobs])
    S = np.array([b[2:4] for b in blobs])
    best, best_score = None, 0.0
    anchors = [(i, j) for i in range(len(names)) for j in range(len(names)) if abs(M[i, 0] - M[j, 0]) > 10 and i < j]
    for a in range(len(blobs)):
        for b in range(len(blobs)):
            if a == b:
                continue
            for i, j in anchors:
                dm = complex(*(M[j] - M[i]))
                dc = complex(*(C[b] - C[a]))
                z = dc / dm
                s = abs(z)
                if not SCALE_PX_MM[0] <= s <= SCALE_PX_MM[1] or abs(np.angle(z)) > 0.45:
                    continue
                if not (0.55 < S[a, 0] / (WH[i, 0] * s) < 1.6 and 0.55 < S[b, 0] / (WH[j, 0] * s) < 1.6):
                    continue
                w = ((M[:, 0] - M[i, 0]) + 1j * (M[:, 1] - M[i, 1])) * z
                pred = np.c_[C[a, 0] + w.real, C[a, 1] + w.imag]
                d = np.linalg.norm(pred[:, None] - C[None], axis=2)
                ratio = S[None, :, 0] / (WH[:, None, 0] * s)
                d[(ratio < 0.55) | (ratio > 1.6)] = 1e9
                k = d.argmin(1)
                good = d[np.arange(len(names)), k] < max(4.0, 0.9 * s)
                score = good.sum() - d[np.arange(len(names)), k][good].sum() / (10 * s)
                if good.sum() >= MIN_FEATURES and score > best_score and len(set(k[good])) == good.sum():
                    best_score, best = score, {names[n]: int(k[n]) for n in np.flatnonzero(good)}
    return best, best_score


# ---------------------------------------------------------------------------------- case edges
LINE_XS = np.arange(-20.0, 56.0, 5.0)       # panel x of the column samples (mm)
LINE_HALF = 16                               # search half height (px, scaled with the image scale)
SIDE_YS = (-6.0, -2.0, 2.0, 6.0)             # panel y of the side edge samples (mm)


def t_given_R(row, K, R, panel):
    """Camera translation for a known rotation, linear least squares on the matched features."""
    A, b = [], []
    for k, (u, v, *_) in row.items():
        if k not in panel:
            continue
        Xc = R @ np.array(panel[k]['xyz'], float)
        x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        A += [[1, 0, -x], [0, 1, -y]]
        b += [x * Xc[2] - Xc[0], y * Xc[2] - Xc[1]]
    return np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]


def _proj(K, R, t, X):
    c = R @ np.asarray(X, float) + t
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2], c[2]


def _ransac_line(a, b, tol=1.0, min_in=4, min_span=30.0):
    """Straight b(a) by pairs. Returns (inlier mask, polynomial) or (None, None)."""
    best, bk = None, 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if abs(a[j] - a[i]) < min_span:
                continue
            m = (b[j] - b[i]) / (a[j] - a[i])
            inl = np.abs(b - b[i] - m * (a - a[i])) < tol
            if inl.sum() > bk:
                best, bk = inl, int(inl.sum())
    if best is None or bk < min_in:
        return None, None
    return best, np.polyfit(a[best], b[best], 1)


def line_point(L, name, c):
    """3D point of a case line at running coordinate c (x for band/bottom, y for L/R)."""
    a, b = L.get('a', 0.0), L.get('b', 0.0)
    if name in ('band', 'bottom'):
        return [c, L['y'] + a * c, L['z'] + b * c]
    return [L['x'] + a * c, c, L['z'] + b * c]


def detect_lines(und, K, R, t, lines, plug_box=None):
    """Case edges from a coarse pose (R, t). h lines run along panel x: 'band' (chrome lower edge: the first strong falling step
    going up from the port row), 'bottom' (case bottom edge against the table). v lines run along panel y: 'L', 'R' (case side
    silhouettes). Returns {name: [(X, Y, Z, u, v), ...]}: samples on a straight fit, ready for PnP."""
    b = cv2.GaussianBlur(und[:, :, 0], (3, 3), 0).astype(np.float32)
    h, w = b.shape
    out = {}
    for name, L in lines.items():
        pts = []
        if name in ('band', 'bottom'):
            for x in LINE_XS:
                X3 = line_point(L, name, x)
                u, v, Zc = _proj(K, R, t, X3)
                sc = 118.0 / Zc
                iu = int(round(u))
                half = int(max(10, LINE_HALF * sc))
                if not 6 <= iu < w - 6:
                    continue
                if name == 'band':
                    vr = _proj(K, R, t, [x, 0.0, 0.0])[1]
                    v1, v0 = int(vr - 14 * sc), int(vr - 95 * sc)
                    if v0 < 4 or v1 >= h - 4:
                        continue
                    prof = b[v0:v1, iu - 4:iu + 5].mean(1)
                    g = np.convolve(prof, [1, 1, 0, -1, -1], mode='same') / 2.0
                    idx = np.flatnonzero(-g[3:-3] > 30.0) + 3
                    if not len(idx):
                        continue
                    i1 = idx[-1]
                    i = max(i1 - 8, 3) + int(np.argmin(g[max(i1 - 8, 3):i1 + 1]))
                    d = g[i - 1] - 2 * g[i] + g[i + 1]
                    pts.append((*X3, u, v0 + i + (0.5 * (g[i - 1] - g[i + 1]) / d if abs(d) > 1e-6 else 0.0)))
                else:
                    iv = int(round(v))
                    if not half + 8 <= iv < h - half - 8:
                        continue
                    if plug_box is not None and plug_box[0] - 8 <= iu <= plug_box[2] + 8 and iv >= plug_box[1] - half:
                        continue
                    prof = b[iv - half - 3:iv + half + 4, iu - 4:iu + 5].mean(1)
                    vv, step = _edge(prof, 3, len(prof) - 3, -1)
                    if vv is None or step < 14.0:
                        continue
                    pts.append((*X3, u, iv - half - 3 + vv))
            if len(pts) < 5:
                continue
            P = np.array(pts)
            inl, c = _ransac_line(P[:, 3], P[:, 4], min_in=5, min_span=40.0)
            if inl is None:
                continue
            Q = P[inl]
            Q = Q[np.linspace(0, len(Q) - 1, 3).astype(int)]
            Q[:, 4] = np.polyval(c, Q[:, 3])
            out[name] = [tuple(map(float, q)) for q in Q]
        else:
            sign = +1 if name == 'L' else -1
            for y in SIDE_YS:
                X3 = line_point(L, name, y)
                u, v, Zc = _proj(K, R, t, X3)
                half = int(max(14, 26 * 118.0 / Zc))
                iu, iv = int(round(u)), int(round(v))
                if not (half + 6 <= iu < w - half - 6 and 6 <= iv < h - 6):
                    continue
                prof = b[iv - 3:iv + 4, iu - half - 3:iu + half + 4].mean(0)
                uu, step = _edge(prof, 3, len(prof) - 3, sign)
                if uu is None or step < 10.0:
                    continue
                pts.append((*X3, iu - half - 3 + uu, v))
            if len(pts) < 3:
                continue
            P = np.array(pts)
            inl, c = _ransac_line(P[:, 4], P[:, 3], tol=1.5, min_in=3, min_span=8.0)
            if inl is None:
                continue
            Q = P[inl]
            Q = Q[[0, -1]]
            Q[:, 3] = np.polyval(c, Q[:, 4])
            out[name] = [tuple(map(float, q)) for q in Q]
    return out


# ---------------------------------------------------------------------------------- pose and measurement
def solve_panel(row, K, panel, R_prior, line_pts=None, lines=None):
    """Camera against the panel by PnP on the matched features, worst feature dropped once if the fit is poor.
    Returns (R_cp, t_cp, rms px, feature names used) or None. X_cam = R_cp X_panel + t_cp (mm)."""
    names = [k for k in row if k in panel and k in POSE_FEATURES]
    if len(names) + sum(len(v) for v in (line_pts or {}).values()) < 6:
        return None
    for _ in range(2):
        X = np.array([panel[k]['xyz'] for k in names], float)
        x = np.array([row[k][:2] for k in names], float)
        if line_pts:
            XL = np.array([p[:3] for ps in line_pts.values() for p in ps], float)
            xL = np.array([p[3:5] for ps in line_pts.values() for p in ps], float)
        else:
            XL, xL = np.zeros((0, 3)), np.zeros((0, 2))
        rv0, _ = cv2.Rodrigues(R_prior)
        t0 = t_given_R({k: row[k] for k in names} if len(names) >= 3 else row, K, R_prior, panel)
        ok, rv, t = cv2.solvePnP(np.vstack([X, XL]), np.vstack([x, xL]), K, None, rv0, t0.reshape(3, 1), True, cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(t).all():
            return None
        p, _ = cv2.projectPoints(X, rv, t, K, None)
        e = np.linalg.norm(p.reshape(-1, 2) - x, axis=1)
        if len(XL):
            pL, _ = cv2.projectPoints(XL, rv, t, K, None)
            eL = np.linalg.norm(pL.reshape(-1, 2) - xL, axis=1)
        else:
            eL = np.zeros(1)
        if e.max() < MAX_REPROJ_PX or len(names) <= 1:
            break
        names.pop(int(e.argmax()))
    R, _ = cv2.Rodrigues(rv)
    allr = np.r_[e, eL]
    return R, t.ravel(), float(np.sqrt((allr ** 2).mean())), float(allr.max()), names


def measure(raw_bgr, side='right', **cfg):
    """Plug tip relative to port 2 from one raw frame. ok=False with a reason rather than a poor number."""
    if side != 'right':
        return PortMeasure(False, 'only the right wrist camera has a model')
    if raw_bgr is None or raw_bgr.shape[:2] != (1024, 1280):
        return PortMeasure(False, f'frame must be 1280x1024 raw, got {None if raw_bgr is None else raw_bgr.shape[:2]}')
    model = load_model()
    panel = model['panel']
    und, K = undistort(raw_bgr, side)
    R_prior = np.array(model['R_view'], float)
    lines = model.get('lines')
    d = detect(und, K, panel, lines, R_prior)
    dbg = dict(row=d['row'], plug=d['plug'], score=d['score'], lines=d['lines'])
    if d['plug'] is None:
        return PortMeasure(False, 'plug: ' + d['plug_why'], debug=dbg)
    if d['row'] is None:
        return PortMeasure(False, 'port row pattern not found', debug=dbg)
    sol = solve_panel(d['row'], K, panel, R_prior, d['lines'], lines)
    if sol is None:
        return PortMeasure(False, 'panel pose failed', debug=dbg)
    for _ in range(POSE_ITERS):       # edges sampled again from the better pose: along-line errors vanish
        lp = detect_lines(und, K, sol[0], sol[1], lines, d['box'])
        s2 = solve_panel(d['row'], K, panel, sol[0], lp, lines)
        if s2 is None:
            break
        sol = s2
        dbg['lines'] = lp
    R, t, rms, emax, used = sol
    dbg.update(R=R, t=t, rms=rms, emax=emax, used=used)
    if emax > MAX_REPROJ_PX:
        return PortMeasure(False, f'row fit residual {emax:.1f} px', debug=dbg)
    pl = d['plug']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = 0.5 * (pl['ul'] + pl['ur']), pl['ve'] - TIP_ABOVE_END_PX
    Z_w = 0.5 * (fx + fy) * SHELL_W_MM / pl['w']
    if abs(Z_w - PLUG_Z_MM) > PLUG_Z_TOL_MM:
        return PortMeasure(False, f'plug depth {Z_w:.1f} mm from its width, expected {PLUG_Z_MM:.1f}', debug=dbg)
    Z = PLUG_Z_MM
    tip_top = np.array([(u - cx) / fx * Z, (v - cy) / fy * Z, Z])
    tip = tip_top + R @ np.array([0.0, -SHELL_T_MM / 2, 0.0])      # rim top -> centre, plug about parallel to the panel axes
    X = R.T @ (tip - t)
    pp = (K @ t)
    dbg.update(tip_cam=tip, Z=Z)
    return PortMeasure(True, '', 1.0, (float(pp[0] / pp[2]), float(pp[1] / pp[2])), (float(u), float(v)), float(X[0]), float(X[1]), float(X[2]), SIGMA_MM, dbg)
