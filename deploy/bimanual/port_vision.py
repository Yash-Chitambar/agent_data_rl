"""Classical port and plug measurement from one raw right wrist frame. No net, no encoders, no in-hand estimate.

    from deploy.bimanual.port_vision import measure
    m = measure(raw_bgr)            # 1280x1024 BGR frame of the right wrist camera
    if m.ok: m.across_mm, m.up_mm, m.gap_mm

Axes and signs match `p_mm` of data/real_training/port_v1/manifest.jsonl: plug tip relative to port 2 centre on the panel,
x across (right in the wrist view), y up, z gap (out of the panel). Method and limits: docs/real/PORT_VISION.md.

Pixels to mm: the camera pose against the panel comes from PnP on a stored 3D model of the case edges and slots (config/port_vision_model.json;
the baseline across the case sets the scale, port 2 itself is not needed). The plug tip is a ray through the shell, at the depth given by
the shell's apparent width in pixels (SHELL_W_EFF_MM). The gap is the tip's distance to the panel plane.
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
SHELL_W_EFF_MM = 8.15       # apparent width between the rim edges: from frames with the plug in the port mouth (docs/real/PORT_VISION.md)
TIP_DEEPER_MM = 1.5         # the tip is this much further from the camera than the shell's middle rows (shell along the view, 45% of 3.3 mm)

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
MIN_FEATURES = 3
SLOT_REJECT_PX = 2.5
MATCH_PAIRS = (('P1', 'RJ'), ('P2', 'RJ'), ('S', 'RJ'), ('P1', 'P2'), ('A', 'RJ'), ('RJ', 'H'), ('P1', 'A'), ('RJ', 'U'))
MATCH_VSCALE = (1.0, 0.75, 1.3)   # the row's height in the image changes with the viewing angle
PLUG_Z_RANGE = (108.0, 145.0)   # plausible shell depth from the camera (mm), from its width
TIP_ABOVE_END_PX = 32.0     # shell tip edge above the sleeve end in the image at depth Z0_TIP (rigid; median of the collect sweep)
Z0_TIP = 117.2
PORT_CENTER_MM = (0.3, 0.5, 0.0)   # port 2 mouth centre in the model frame (origin: bright lip blob); from frames with the plug in the mouth
# Residual SD after a constant, held-out collect blocks, gap under 15 mm and over (docs/real/PORT_VISION.md). Constant offsets are extra: 0.5 mm guess.
SIGMA_NEAR_MM = (0.4, 1.2, 0.7)
SIGMA_NEAR_GRASP_MM = (0.4, 0.8, 0.5)
SIGMA_FAR_MM = (1.2, 1.2, 0.9)
POSE_ITERS = 2
POSE_FEATURES = ('P1', 'P2', 'S')     # clean slots; the frames of the larger ports only confirm the pattern
MAX_REPROJ_PX = 5.0         # tuned on the tune blocks: errors do not grow up to 5 px, detection rises
MAX_ROT_FROM_NOMINAL_DEG = 25.0
SCALE_PX_MM = (1.8, 9.0)    # plausible image scale at the panel
PLUG_ROI = (575, 715, 450, 590)     # u0, u1, v0, v1: undistorted pixels. The plug is rigid in the hand.
PLUG_W_PX = (38.0, 56.0)
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
    """Metal shell in its window. Returns dict(ul, ur, vt, ve, w, ...) in undistorted pixels, or (None, reason).
    The sleeve end is the shell's lower edge (bright over dark). The width comes from all shell rows at once."""
    u0, u1, v0, v1 = PLUG_ROI
    g = cv2.GaussianBlur(cv2.cvtColor(und[v0:v1, u0:u1], cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    # columns inside the shell for any plausible position
    mid = (u1 - u0) // 2
    c0, c1 = mid - 12, mid + 12
    # candidates: rising then falling on the mid columns: tip edge above, sleeve end below, at the shell's height in the image
    prof = g[:, c0:c1].mean(1)
    gr = np.convolve(prof, [1, 1, 0, -1, -1], mode='same') / 2.0
    best, best_s = None, 0.0
    fall = [i for i in range(4, len(gr) - 4) if -gr[i] > MIN_EDGE and -gr[i] >= -gr[i - 1] and -gr[i] >= -gr[i + 1]]
    why = 'no shell end edge'
    for i in fall:
        if i < 27:
            continue
        below = prof[i + 4:i + 12].mean() if i + 12 < len(prof) else 255.0
        above = prof[i - 18:i - 4].mean()
        if above - below < 15.0:
            continue
        side = g[max(i - 26, 0):i - 5].mean(0)
        ul, sl = _edge(side, 8, mid - 8, +1)
        ur, sr = _edge(side, mid + 8, len(side) - 8, -1)
        if ul is None or ur is None or min(sl, sr) < MIN_EDGE:
            why = f'no shell side edges (steps {sl:.0f}, {sr:.0f})'
            continue
        w = ur - ul
        if not PLUG_W_PX[0] <= w <= PLUG_W_PX[1]:
            why = f'shell width {w:.1f} px out of range'
            continue
        # the shell's centre in the window should sit on the mid columns; bright rims on both sides
        score = min(sl, sr) + 0.3 * -gr[i]
        if score > best_s:
            best, best_s = (i, ul, ur, sl, sr, w), score
    if best is None:
        return None, why
    ve, ul, ur, sl, sr, w = best
    a_, b_, c_ = -gr[ve - 1], -gr[ve], -gr[ve + 1]
    d_ = a_ - 2 * b_ + c_
    vei = ve + (0.5 * (a_ - c_) / d_ if abs(d_) > 1e-6 else 0.0)
    a, b = int(ul) + 4, int(ur) - 3
    top = g[:, a:b].mean(1)
    vt, st = _edge(top, max(ve - int(PLUG_H_PX[1]), 3), ve - int(PLUG_H_PX[0]) + 1, +1)
    return dict(ul=ul + u0, ur=ur + u0, vt=(vt + v0) if vt is not None else None, ve=vei + v0, w=w, steps=(best_s, sl, sr, st)), ''


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
    names, M0 = _match_xy(panel)
    WH = np.array([panel[k]['wh'] for k in names], float)
    if len(blobs) < MIN_FEATURES:
        return None, 0.0
    C = np.array([b[:2] for b in blobs])
    S = np.array([b[2:4] for b in blobs])
    best, best_score = None, 0.0
    idx = {k: i for i, k in enumerate(names)}
    pairs = [(idx[a], idx[b]) for a, b in MATCH_PAIRS if a in idx and b in idx]
    n = len(names)
    for c in MATCH_VSCALE:
        M = M0 * [1.0, c]
        for i, j in pairs:
            dm = complex(*(M[j] - M[i]))
            for a in range(len(blobs)):
                for b in range(len(blobs)):
                    if a == b or C[b, 0] <= C[a, 0]:
                        continue
                    z = complex(*(C[b] - C[a])) / dm
                    s = abs(z)
                    if not SCALE_PX_MM[0] <= s <= SCALE_PX_MM[1] or abs(np.angle(z)) > 0.3:
                        continue
                    if not (0.55 < S[a, 0] / (WH[i, 0] * s) < 1.6 and 0.55 < S[b, 0] / (WH[j, 0] * s) < 1.6):
                        continue
                    w = ((M[:, 0] - M[i, 0]) + 1j * (M[:, 1] - M[i, 1])) * z
                    pred = np.c_[C[a, 0] + w.real, C[a, 1] + w.imag]
                    d = np.linalg.norm(pred[:, None] - C[None], axis=2)
                    ratio = S[None, :, 0] / (WH[:, None, 0] * s)
                    d[(ratio < 0.55) | (ratio > 1.6)] = 1e9
                    k = d.argmin(1)
                    dm_ = d[np.arange(n), k]
                    good = dm_ < max(4.0, 0.9 * s)
                    ng = int(good.sum())
                    if ng < MIN_FEATURES or len(set(k[good])) != ng:
                        continue
                    score = ng - dm_[good].sum() / (10 * s)
                    if score > best_score:
                        best_score, best = score, {names[q]: int(k[q]) for q in np.flatnonzero(good)}
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
def solve_panel(row, K, panel, R_prior, line_pts=None, t_init=None):
    """Camera against the panel by PnP on case edge samples and slots. Slots with a poor fit are dropped: the edges carry the pose.
    Returns (R_cp, t_cp, rms px, max px, slots used) or None. X_cam = R_cp X_panel + t_cp (mm)."""
    names = [k for k in row if k in panel and k in POSE_FEATURES]
    XL = np.array([p[:3] for ps in (line_pts or {}).values() for p in ps], float).reshape(-1, 3)
    xL = np.array([p[3:5] for ps in (line_pts or {}).values() for p in ps], float).reshape(-1, 2)
    if len(names) + len(XL) < 6:
        return None
    rv0, _ = cv2.Rodrigues(R_prior)
    t0 = t_given_R(row, K, R_prior, panel) if t_init is None else t_init
    while True:
        X = np.vstack([np.array([panel[k]['xyz'] for k in names], float).reshape(-1, 3), XL])
        x = np.vstack([np.array([row[k][:2] for k in names], float).reshape(-1, 2), xL])
        ok, rv, t = cv2.solvePnP(X, x, K, None, rv0.copy(), np.asarray(t0, float).reshape(3, 1).copy(), True, cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(t).all():
            return None
        p, _ = cv2.projectPoints(X, rv, t, K, None)
        e = np.linalg.norm(p.reshape(-1, 2) - x, axis=1)
        ns = len(names)
        if ns and e[:ns].max() > SLOT_REJECT_PX:
            names.pop(int(e[:ns].argmax()))
            if len(names) + len(XL) < 6:
                return None
            continue
        break
    R, _ = cv2.Rodrigues(rv)
    return R, t.ravel(), float(np.sqrt((e ** 2).mean())), float(e.max()), list(names)


def panel_pose(und, K, box=None, max_reproj_px=MAX_REPROJ_PX, model=None):
    """Camera against the panel from one undistorted frame. Returns (dict(R, t, rms, emax, used, lines, row), '') or (None, reason)."""
    model = model or load_model()
    panel, lines = model['panel'], model['lines']
    R_prior = np.array(model['R_view'], float)
    blobs = bright_blobs(und, box)
    m, _ = match_row(blobs, panel)
    if not m:
        return None, 'port row pattern not found'
    row = {k: blobs[i][:2] + blobs[i][2:4] for k, i in m.items()}
    lp = detect_lines(und, K, R_prior, t_given_R(row, K, R_prior, panel), lines, box)
    sol = solve_panel(row, K, panel, R_prior, lp)
    if sol is None:
        return None, 'panel pose failed'
    for _ in range(POSE_ITERS):       # edges sampled again from the better pose: along-line errors vanish
        lp2 = detect_lines(und, K, sol[0], sol[1], lines, box)
        s2 = solve_panel(row, K, panel, sol[0], lp2, sol[1])
        if s2 is None:
            break
        sol, lp = s2, lp2
    if not all(n in lp for n in ('band', 'bottom')) or not ('L' in lp or 'R' in lp):
        return None, f'case edges missing: band, bottom and one side needed, have {sorted(lp)}'
    R, t, rms, emax, used = sol
    if emax > max_reproj_px and 'L' in lp and 'R' in lp:      # a side silhouette can be off (corner radius, far views)
        errs = {}
        for n, ps in lp.items():
            P = np.array(ps, float)
            p, _ = cv2.projectPoints(P[:, :3].reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t.reshape(3, 1), K, None)
            errs[n] = float(np.linalg.norm(p.reshape(-1, 2) - P[:, 3:5], axis=1).max())
        worst = max(errs, key=errs.get)
        if worst in ('L', 'R'):
            lp2 = {k: v for k, v in lp.items() if k != worst}
            s2 = solve_panel(row, K, panel, R, lp2, t)
            if s2 is not None:
                R, t, rms, emax, used = s2
                lp = lp2
    dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ R_prior.T)[0]))
    if dR > MAX_ROT_FROM_NOMINAL_DEG:
        return None, f'panel rotation {dR:.1f} deg from nominal'
    if emax > max_reproj_px:
        return None, f'row fit residual {emax:.1f} px'
    return dict(R=R, t=t, rms=rms, emax=emax, used=used, lines=lp, row=row), ''


def plug_box(pl):
    """Image box of the shell (u0, v0, u1, v1): hidden from the panel search."""
    return (int(pl['ul']) - 3, int(pl['ve'] - TIP_ABOVE_END_PX) - 3, int(pl['ur']) + 4, int(pl['ve']) + 2)


def measure(raw_bgr, side='right', **cfg):
    """Plug tip relative to port 2 from one raw frame. ok=False with a reason rather than a poor number.

    cfg: plug_z_mm: the shell's depth for this grasp (see plug_depth); default the depth from this frame's shell width (noisier).
         max_reproj_px: pose fit gate."""
    if side != 'right':
        return PortMeasure(False, 'only the right wrist camera has a model')
    if raw_bgr is None or raw_bgr.shape[:2] != (1024, 1280):
        return PortMeasure(False, f'frame must be 1280x1024 raw, got {None if raw_bgr is None else raw_bgr.shape[:2]}')
    model = load_model()
    und, K = undistort(raw_bgr, side)
    pl, why = detect_plug(und)
    dbg = dict(plug=pl)
    pose, why_p = panel_pose(und, K, plug_box(pl) if pl else None, float(cfg.get('max_reproj_px', MAX_REPROJ_PX)), model)
    port_px = None
    if pose is not None:
        dbg.update(pose)
        pp = K @ pose['t']
        port_px = (float(pp[0] / pp[2]), float(pp[1] / pp[2]))
    if pl is None:
        return PortMeasure(False, 'plug: ' + why, port_px=port_px, debug=dbg)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z_w = 0.5 * (fx + fy) * SHELL_W_EFF_MM / pl['w']     # depth of the shell's middle rows
    Z = float(cfg.get('plug_z_mm') or Z_w)
    u, v = 0.5 * (pl['ul'] + pl['ur']), pl['ve'] - TIP_ABOVE_END_PX * Z0_TIP / Z
    plug_px = (float(u), float(v))
    dbg['Z_w'] = Z_w
    if not PLUG_Z_RANGE[0] <= Z_w <= PLUG_Z_RANGE[1]:
        return PortMeasure(False, f'plug depth {Z_w:.1f} mm from its width: another grasp or not the shell', port_px=port_px, plug_px=plug_px, debug=dbg)
    if pose is None:
        return PortMeasure(False, why_p, plug_px=plug_px, debug=dbg)
    R, t = pose['R'], pose['t']
    Zt = Z + TIP_DEEPER_MM
    tip_top = np.array([(u - cx) / fx * Zt, (v - cy) / fy * Zt, Zt])
    tip = tip_top + R @ np.array([0.0, -SHELL_T_MM / 2, 0.0])      # rim top -> centre, plug about parallel to the panel axes
    X = R.T @ (tip - t) - np.array(model.get('port_center_in_model_mm', PORT_CENTER_MM), float)
    dbg.update(tip_cam=tip, Z=Z)
    conf = float(np.clip(1.0 - pose['emax'] / MAX_REPROJ_PX * 0.5, 0.0, 1.0))
    sigma = SIGMA_FAR_MM if X[2] > 15.0 else (SIGMA_NEAR_GRASP_MM if cfg.get('plug_z_mm') else SIGMA_NEAR_MM)
    return PortMeasure(True, '', conf, port_px, plug_px, float(X[0]), float(X[1]), float(X[2]), sigma, dbg)


def plug_depth(frames, side='right'):
    """Shell depth (mm) for one grasp: median over raw frames of the same grasp, for measure(..., plug_z_mm=). None if too few shells."""
    z = []
    for f in frames:
        pl, _ = detect_plug(undistort(f, side)[0])
        if pl is not None:
            z.append(pl['w'])
    if len(z) < 3:
        return None
    K = undistort(frames[0], side)[1]
    return float(0.5 * (K[0, 0] + K[1, 1]) * SHELL_W_EFF_MM / np.median(z))
