"""Measure the real in-hand reader pose from the calibrated right wrist camera (docs/real/INHAND_REAL.md).

Features in an undistorted frame (extended canvas: the silver's back edge lies past the raw frame's undistorted bottom):
silver plate edges (4 fitted lines -> 4 corners), plug tip and plug base lines. A least-squares pose fit (6 dof) of the
reader model gives the reader in the camera, hence in grasp_site (the camera to site transform is rigid: no joints).

    PYTHONPATH=. MUJOCO_GL=egl python scripts/real2sim/inhand_measure.py --out results/real/inhand_real/poses.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / 'blind_insertion/configs/real_cameras_duchamp3.json').read_text())
W = SPEC['wrist']['right']
K0 = W['intrinsics']
KM = np.array([[K0['fx'], 0, K0['cx']], [0, K0['fy'], K0['cy']], [0, 0, 1.]])
DIST = np.array(W['distortion'])
OFF = np.array(SPEC['arms']['right']['joint_offsets_rad'])
RUNS = ROOT / 'data/real_runs/2026-11-02'
CANVAS = (1280, 1300)       # undistorted canvas: same K, taller, so the silver's back edge stays in view
MAP = cv2.initUndistortRectifyMap(KM, DIST, None, KM, CANVAS, cv2.CV_32FC1)


def undist(raw):
    return cv2.remap(raw, MAP[0], MAP[1], cv2.INTER_LINEAR)


# Reader model (reader frame: origin plug tip, z to the body end, y up, x width; mm). Silver sizes: wrist hover frames
# (docs/real/INHAND_REAL.md "Reader"); the caliper 29.6 is the width and 34.6 the length.
SILVER = dict(w=29.6, l=34.6, z0=49.2, y=4.4)
PLUG = dict(w=8.25, t=2.4, base_z=5.8, neck_top=3.5)


def model_points(s=SILVER, p=PLUG, halo=0.):
    """Corners, in the fit's order: silver front L/R, back L/R; plug tip L/R, plug base L/R (the neck's front top edge).
    Image left is reader -x: y up towards the camera, z towards the image bottom, x = y x z = image right."""
    w, l, z0, y = s['w'] + 2 * halo, s['l'] + 2 * halo, s['z0'] - halo, s['y']
    x = w / 2
    pw = p['w'] / 2
    return np.array([[-x, y, z0], [x, y, z0], [-x, y, z0 + l], [x, y, z0 + l],
                     [-pw, p['t'] / 2, 0], [pw, p['t'] / 2, 0], [-pw, p['neck_top'], p['base_z']], [pw, p['neck_top'], p['base_z']]])


def edge(profile, centre, half, sign=0):
    """Sub-pixel position of the largest gradient of a 1-D profile near `centre` (+- half). sign: +1 rising, -1 falling."""
    p = cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (7, 1), 0).ravel()
    g = np.gradient(p)
    a, b = max(1, int(centre - half)), min(len(p) - 2, int(centre + half))
    if b <= a:
        return None
    s = g[a:b] * sign if sign else np.abs(g[a:b])
    i = a + int(np.argmax(s))
    v = g * sign if sign else np.abs(g)
    d = v[i - 1] - 2 * v[i] + v[i + 1]
    return i + (.5 * (v[i - 1] - v[i + 1]) / d if abs(d) > 1e-6 else 0.)


def fit_line(a, b):
    """Robust line b = m a + c."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 6:
        return None
    m, c = np.polyfit(a, b, 1)
    for _ in range(3):
        r = b - (m * a + c)
        k = np.abs(r) < max(1., 2.5 * np.median(np.abs(r)))
        if k.sum() < 6:
            break
        m, c = np.polyfit(a[k], b[k], 1)
    return m, c


def _isect(v, h):
    """v: x = m y + c; h: y = m x + c."""
    mv, cv_ = v
    mh, ch = h
    y = (mh * cv_ + ch) / (1 - mh * mv)
    return np.array([mv * y + cv_, y])


def detect(und, debug=None, silver_only=False, dark=85, min_v=130):
    """Feature pixels of a held reader: dict with 'pts' (8, 2) in model_points order, or None and a reason."""
    g = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    hsv = cv2.cvtColor(und, cv2.COLOR_BGR2HSV)
    x0, x1, y0, y1 = 430, 860, 700, CANVAS[1]
    mask = ((hsv[..., 2] > min_v) & (hsv[..., 1] < 55))[y0:y1, x0:x1].astype(np.uint8)   # the silver is bright and grey; the tips are yellow
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _c = cv2.connectedComponentsWithStats(mask)
    cand = [i for i in range(1, n) if st[i, 4] > 12000 and abs(st[i, 0] + st[i, 2] / 2 + x0 - 640) < 110]
    if not cand:
        return None, 'no silver blob'
    i = max(cand, key=lambda a: st[a, 4])
    bx, by, bw, bh = st[i, 0] + x0, st[i, 1] + y0, st[i, 2], st[i, 3]
    if bw < 120 or bh < 120:
        return None, 'silver too small'
    M = lab == i
    # edges: walk out of the silver to the black housing gap (the silver reflects the yellow tips, so a colour or plain
    # gradient test stops too early), then take the largest gradient just before the gap
    def walk(p, c, step, dark=dark):
        k, n_ = int(c), len(p)
        while 0 < k < n_ - 1 and p[k] > dark:
            k += step
        if not 0 < k < n_ - 1:
            return np.nan
        return edge(p, k - 2 * step, 5, -step)

    ys = np.arange(by + int(.15 * bh), by + int(.85 * bh))
    cx_ = bx + bw // 2
    L = fit_line(ys, [walk(g[y], cx_, -1) for y in ys])
    R = fit_line(ys, [walk(g[y], cx_, +1) for y in ys])
    xs = np.arange(bx + int(.3 * bw), bx + int(.7 * bw))
    cy_ = by + bh // 2
    T = fit_line(xs, [walk(g[:, x], cy_, -1) for x in xs])
    B = fit_line(xs, [walk(g[:, x], cy_, +1) for x in xs])
    if None in (L, R, T):
        return None, 'silver edge fit'
    cTL, cTR = _isect(L, T), _isect(R, T)
    if silver_only:
        pts = np.full((8, 2), np.nan)
        pts[:2] = cTL, cTR
        if B:
            pts[2:4] = _isect(L, B), _isect(R, B)
        return {'pts': pts, 'silver_px_w': float(cTR[0] - cTL[0]), 'plug_px_w': float('nan')}, ''
    # plug: bright along the central column above the silver
    xc = (cTL[0] + cTR[0]) / 2
    col = g[:, int(xc) - 12:int(xc) + 13].mean(1)
    top = int(cTL[1]) - 80
    lo = max(0, top - 480)
    seg = col[lo:top]
    if len(seg) < 30:
        return None, 'silver too high'
    bg = np.median(seg)
    if seg.max() - bg < 25:
        return None, 'no plug'
    hi = np.max(seg)
    idx = np.where(seg > bg + .5 * (hi - bg))[0]
    # last contiguous bright run nearest the neck: the plug is the bright run closest to the silver
    runs = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    run = max(runs, key=lambda r_: (len(r_) >= 8, r_[-1]))
    tip_y, base_y = lo + run[0], lo + run[-1]
    if base_y - tip_y < 12:
        return None, 'plug too short'
    rows = np.arange(tip_y + 3, base_y - 2)
    # plug sides from the central column outwards
    midrow = g[int((tip_y + base_y) / 2)]
    x_l = edge(midrow, xc - 25, 30, +1)
    x_r = edge(midrow, xc + 25, 30, -1)
    if x_l is None or x_r is None:
        return None, 'plug sides'
    PL = fit_line(rows, [edge(g[y], x_l, 8, +1) for y in rows])
    PR = fit_line(rows, [edge(g[y], x_r, 8, -1) for y in rows])
    pxs = np.arange(int(x_l) + 4, int(x_r) - 3)
    PT = fit_line(pxs, [edge(g[:, x], tip_y, 7, +1) for x in pxs])
    PB = fit_line(pxs, [edge(g[:, x], base_y, 7, -1) for x in pxs])
    if None in (PL, PR, PT, PB):
        return None, 'plug edge fit'
    pts = np.array([cTL, cTR, _isect(L, B) if B else [np.nan] * 2, _isect(R, B) if B else [np.nan] * 2,
                    _isect(PL, PT), _isect(PR, PT), _isect(PL, PB), _isect(PR, PB)], float)
    if debug is not None:
        im = und.copy()
        for k, q in enumerate(pts):
            if np.isfinite(q[0]):
                cv2.circle(im, (int(q[0]), int(q[1])), 4, (0, 0, 255) if k < 4 else (255, 0, 0), 1)
        cv2.imwrite(debug, im)
    return {'pts': pts, 'silver_px_w': float(cTR[0] - cTL[0]), 'plug_px_w': float(pts[5, 0] - pts[4, 0])}, ''


def project(P, rv, t):
    Pc = Rot.from_rotvec(rv).apply(P) + t
    return np.c_[KM[0, 0] * Pc[:, 0] / Pc[:, 2] + KM[0, 2], KM[1, 1] * Pc[:, 1] / Pc[:, 2] + KM[1, 2]]


def fit_pose(pts, x0=None, halo=0., silver=SILVER, plug=PLUG, sigma=None):
    """Reader in the camera: (R (3, 3), t mm, rms px). pts as detect(); NaN corners are skipped."""
    P = model_points(silver, plug, halo)
    ok = np.isfinite(pts[:, 0])
    sig = np.ones(len(P)) if sigma is None else sigma
    if x0 is None:
        # reader roughly flat under the camera, the plug away from it (camera +y down, +z forward): reader z = -cam y
        R0 = np.array([[-1, 0, 0], [0, 0, -1.], [0, -1, 0]], float)
        R0 = np.c_[R0[:, 0], R0[:, 1], np.cross(R0[:, 0], R0[:, 1])]
        x0 = np.r_[Rot.from_matrix(R0).as_rotvec(), 0, 60, 140]

    def res(x):
        return ((project(P[ok], x[:3], x[3:]) - pts[ok]) / sig[ok][:, None]).ravel()

    s = least_squares(res, x0, loss='soft_l1', f_scale=3.)
    r = project(P[ok], s.x[:3], s.x[3:]) - pts[ok]
    return Rot.from_rotvec(s.x[:3]).as_matrix(), s.x[3:], float(np.sqrt((r ** 2).sum(1).mean())), r


class Hand:
    """Rigid camera -> grasp_site, from the sim model and the calibrated mount."""

    def __init__(self):
        from deploy.bimanual.scene import SceneModel
        self.sc = SceneModel('cpu')
        m = self.m = self.sc.mjm
        self.d = mujoco.MjData(m)
        q = np.array(W['pose_in_link6']['quat'], float)
        M = np.zeros(9)
        mujoco.mju_quat2Mat(M, q / np.linalg.norm(q))
        self.R_c6, self.p_c6 = M.reshape(3, 3), np.array(W['pose_in_link6']['pos'])
        self.pose(np.zeros(6))
        Rs, ps = self.site()
        Rc, pc = self.cam()
        self.R_cs = Rs.T @ Rc           # camera in the site frame
        self.p_cs = Rs.T @ (pc - ps)
        sp = json.loads((ROOT / 'results/usbc/policy/pick/inhand_spread.json').read_text())['nominal']['tilt55_flip0']
        w, x, y, z = sp['quat_wxyz']
        self.R_nom, self.p_nom = Rot.from_quat([x, y, z, w]).as_matrix(), np.array(sp['pos_m'])

    def camera_shift_site(self):
        """The runtime's wrist camera shift (BimanualCfg.tip_offset_target='camera') on the site axes (m). The sim image
        uses the shifted camera, so a reader that looks right in the sim is this far from the hand-eye value."""
        fit = json.loads((ROOT / 'results/usbc/gripper_sweep/sweep_20261030/fit.json').read_text())
        off = np.array(list(fit['arms']['right']['fitted_offset_mm'].values())) * 1e-3
        M = np.zeros(9)
        mujoco.mju_quat2Mat(M, self.m.body('gripper').quat)
        self.pose(np.zeros(6))
        Rs, _ = self.site()
        R6 = self.d.xmat[self.m.body('link_6').id].reshape(3, 3)
        return Rs.T @ R6 @ (-M.reshape(3, 3) @ off)

    def x0(self):
        """Initial guess for fit_pose: the nominal in-hand pose seen from the camera."""
        R = self.R_cs.T @ self.R_nom
        return np.r_[Rot.from_matrix(R).as_rotvec(), self.R_cs.T @ (self.p_nom - self.p_cs) * 1e3]

    def delta(self, Rs, ps):
        """(mm in the site frame, rotation vector deg in the site frame) of a reader pose about the nominal."""
        return (ps - self.p_nom) * 1e3, Rot.from_matrix(Rs @ self.R_nom.T).as_rotvec(degrees=True)

    def pose(self, q_b):
        for j in range(6):
            self.d.qpos[self.m.joint(f'joint{j + 1}').qposadr] = q_b[j]
        mujoco.mj_kinematics(self.m, self.d)

    def site(self):
        s = self.m.site('grasp_site').id
        return self.d.site_xmat[s].reshape(3, 3).copy(), self.d.site_xpos[s].copy()

    def cam(self):
        b = self.m.body('link_6').id
        R6, p6 = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
        return R6 @ self.R_c6, R6 @ self.p_c6 + p6

    def reader_in_site(self, R_rc, t_rc_mm):
        """reader in camera -> reader in site (R, p m)."""
        return self.R_cs @ R_rc, self.R_cs @ (t_rc_mm * 1e-3) + self.p_cs


def held_frames(run, n=12):
    """Raw wrist frames of a held reader in free space: after the pick's end_hold and before the next stage."""
    ev = [json.loads(x) for x in open(run / 'events.jsonl')]
    t0 = next((e['t'] for e in ev if e.get('event') == 'end_hold'), None)
    if t0 is None:
        t0 = next((e['t'] for e in ev if e.get('event') == 'lift'), None)
    t1 = next((e['t'] for e in ev if e.get('event') == 'done'), None)
    if t1 is None:
        t1 = t0 + 1.
    fr = [json.loads(x) for x in open(run / 'frames.jsonl')]
    fr = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw' and t0 - .5 <= f['t_capture'] <= t1 + 3]
    return fr[:n]


def all_runs():
    return sorted([Path(p) for p in glob.glob(str(RUNS / 'pick_*')) + glob.glob(str(RUNS / 'full_*')) + glob.glob(str(RUNS / 'collect_*'))], key=lambda p: p.name)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/real/inhand_real/poses.csv')
    ap.add_argument('--debug', default='')
    ap.add_argument('--write_preset', default='', help='write this configs/inhand_real_<name>.json from the fit')
    a = ap.parse_args()
    import pandas as pd
    hand = Hand()
    rows = []
    for run in all_runs():
        for f in held_frames(run):
            und = undist(cv2.imread(str(run / f['file'])))
            dbg = os.path.join(a.debug, f"{run.name}_{Path(f['file']).stem}.jpg") if a.debug else None
            r, why = detect(und, dbg)
            if r is None:
                print(run.name, f['file'], why)
                continue
            R, t, rms, _res = fit_pose(r['pts'], hand.x0())
            Rs, ps = hand.reader_in_site(R, t)
            dp, dr = hand.delta(Rs, ps)
            # the rotation on the reader's own axes: pitch about the closing axis, yaw about its thickness axis, roll about its length
            ax = hand.R_nom.T @ dr
            rows.append(dict(run=run.name, file=f['file'], rms_px=rms, tx=t[0], ty=t[1], tz=t[2], dx=dp[0], dy=dp[1], dz=dp[2], rx=dr[0], ry=dr[1], rz=dr[2],
                             pitch=ax[0], yaw=ax[1], roll=ax[2],
                             sil_px=r['silver_px_w'], plug_px=r['plug_px_w']))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    good = df[df.rms_px < 5]
    g = good.groupby('run')[['dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'pitch', 'yaw', 'roll', 'rms_px']].median()
    print(g.round(2).to_string())
    print(good.groupby('run').size().to_dict(), len(df))
    print(g.agg(['mean', 'std', 'min', 'max']).round(2).to_string())
    print('within-run SD', good.groupby('run')[['dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'pitch', 'yaw', 'roll']].std().mean().round(2).to_dict())
    if a.write_preset:
        sh = hand.camera_shift_site() * 1e3
        med = g.median()
        sd = g.std()
        p = ROOT / f'blind_insertion/configs/inhand_real_{a.write_preset}.json'
        cfg = json.loads(p.read_text()) if p.exists() else {}
        cfg.pop('inhand', None)
        cfg['_about'] = ('Real grasp, right wrist camera, 13 grasps 2026-11-02 (docs/real/INHAND_REAL.md). Deltas about the Part 1 nominal on the TCP axes, '
                         'the median over grasps; roll about the reader axis set to 0 (flat reader, level hand, below what this camera resolves); y follows from the '
                         'pads centring the reader (envs/inhand_real.reader_in_site). Position in the runtime camera frame '
                         '(tip_offset_target=camera): image-true. half_range = 2 SD over grasps.')
        rv = hand.R_nom @ np.array([med.pitch, med.yaw, 0.])     # no roll: below what this camera resolves
        cfg['inhand'] = dict(mean_mm=[round(float(med.dx + sh[0]), 2), 0., round(float(med.dz + sh[2]), 2)], mean_rotvec_deg=[round(float(x), 2) for x in rv],
                             half_range_mm=[round(float(2 * sd.dx), 2), 0., round(float(2 * sd.dz), 2)], half_range_deg=[1., round(float(2 * sd.ry), 2), 1.],
                             hand_eye_mean_mm=[round(float(med.dx), 2), round(float(med.dy), 2), round(float(med.dz), 2)], render_lateral_mm=1.45, n_grasps=int(len(g)))
        cfg.setdefault('look', dict(silver_width_m=.0296, silver_length_m=.0346, silver_front_z_m=.0492, shoulder_length_m=.005, shoulder_front_width_m=.0135))
        p.write_text(json.dumps(cfg, indent=1) + '\n')
        print('wrote', p, cfg['inhand'])
