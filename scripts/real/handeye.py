"""Wrist-camera hand-eye (T_link6_cam) into the camera spec, with a residual estimate; synthetic self-test.

The wrist-camera mount error is the accuracy floor of the pick servo: the sim randomizes it up to 5.1 mm / 1.2 deg
(right) and nothing in the image reveals it. scripts/real/calib.py collects board views from many arm poses and
solves AX = XB (Park-Martin + reprojection refinement); its `solve` fits the encoder offsets together with X and
writes data/calib/<run>/solve_<side>.json only. This tool:

  apply     refits X with the spec's joint offsets FIXED (the frame every net and the runner use: model = encoder +
            rig.joint_offsets), on calib.py's views and intrinsics; residuals: reprojection (all views, 5-fold
            hold-out), fold-to-fold spread of X (mm / deg: the metric uncertainty), the change against the spec and
            against the sim DR range. --write puts pos/quat into wrist.<side>.pose_in_link6 (and wrist.sim for the
            right camera), the run, date and residuals into wrist.<side>, bumps calibration_version, keeps a backup.
  selftest  synthetic board views from random arm poses with a known X, pixel noise and joint noise: the solver
            must return X within the expected error (no hardware).

    # rig, bases clamped, estop.py running, board flat on the table (calib.py docstring):
    .venv/bin/python scripts/real/calib.py collect --side right --n 40        # MOVES
    .venv/bin/python scripts/real/calib.py solve --side right --run data/calib/run_NNN
    .venv/bin/python scripts/real/handeye.py apply --side right --run data/calib/run_NNN [--write]
    .venv/bin/python scripts/real/handeye.py selftest                         # offline

Note: writing the spec changes the sim's wrist camera (dry runs, future training), not the trained nets; the nets
were trained around the previous pose with mount DR. A change beyond the DR range means the servo nets see a
camera they were not trained for: re-render / retrain before trusting the pick.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import calib as C  # noqa: E402
import rig  # noqa: E402

DR_RANGE = {'right': (5.1, 1.2), 'left': (3.4, 1.2)}     # mm, deg: 2x the real mount error used by the sim DR


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def T_of(pose):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(pose['quat'], float))
    return C.T_from(R.reshape(3, 3), pose['pos'])


def diff(Ta, Tb):
    D = C.T_inv(Ta) @ Tb
    return float(np.linalg.norm(D[:3, 3]) * 1e3), float(np.degrees(np.linalg.norm(cv2.Rodrigues(D[:3, :3])[0])))


def solve_fixed_offsets(Fs, corners, K, dist, folds=5):
    """X, Y, residuals and the fold-to-fold spread of X, joint offsets fixed (Fs already include them)."""
    Cs = [C.board_pose(c, K, dist)[0] for c in corners]
    X0 = C.park_martin(Fs, Cs)
    Y0 = np.mean([F @ X0 @ Cb for F, Cb in zip(Fs, Cs)], axis=0)
    U, _, Vt = np.linalg.svd(Y0[:3, :3])
    Y0[:3, :3] = U @ Vt
    X, Y = C.refine(X0, Y0, Fs, corners, K, dist)
    e = np.linalg.norm(C.reproj_residual(X, Y, Fs, corners, K, dist), axis=1)
    n = len(Fs)
    idx = np.random.default_rng(0).permutation(n)
    ho, spread = [], []
    for f in range(folds):
        te, tr = idx[f::folds], np.setdiff1d(idx, idx[f::folds])
        Xf, Yf = C.refine(X, Y, [Fs[i] for i in tr], [corners[i] for i in tr], K, dist)
        ho.append(np.linalg.norm(C.reproj_residual(Xf, Yf, [Fs[i] for i in te], [corners[i] for i in te], K, dist), axis=1))
        spread.append(diff(X, Xf))
    sp = np.array(spread)
    return X, Y, {'n_views': n, 'reproj_mean_px': float(e.mean()), 'reproj_p95_px': float(np.percentile(e, 95)),
                  'holdout_mean_px': float(np.concatenate(ho).mean()),
                  'fold_spread_mm': float(sp[:, 0].max()), 'fold_spread_deg': float(sp[:, 1].max())}


def cmd_apply(a):
    run = Path(a.run)
    sol = json.loads((run / f'solve_{a.side}.json').read_text())
    K, dist = np.array(sol['K']), np.array(sol['dist'])
    S = [json.loads(p.read_text()) for p in sorted((run / a.side).glob('[0-9]*.json'))]
    S = [s for s in S if s.get('wrist_corners') is not None]
    corners = [np.array(s['wrist_corners'], np.float32) for s in S]
    kin = C.Kin()
    off = np.asarray(rig.joint_offsets(a.side))
    Fs = [kin.fk(np.array(s['q'][:6]) + off) for s in S]
    X, Y, q = solve_fixed_offsets(Fs, corners, K, dist)
    spec_path = rig.SPEC
    spec = json.loads(spec_path.read_text())
    w = spec['wrist'][a.side]
    old = T_of(w['pose_in_link6'])
    dmm, ddeg = diff(old, X)
    dr = DR_RANGE[a.side]
    q.update(change_vs_spec_mm=dmm, change_vs_spec_deg=ddeg, sim_dr_range_mm_deg=list(dr),
             within_dr=bool(dmm <= dr[0] and ddeg <= dr[1]), joint_offsets_rad=off.tolist())
    pose = {'pos': X[:3, 3].round(6).tolist(), 'quat': quat_of(X[:3, :3]).round(6).tolist()}
    print(json.dumps({'side': a.side, 'run': str(run), 'pose_in_link6': pose, 'previous': w['pose_in_link6'], 'quality': q}, indent=1))
    if not q['within_dr']:
        print(f'WARNING: the mount differs from the spec by {dmm:.1f} mm / {ddeg:.2f} deg, beyond the sim DR range '
              f'{dr[0]} mm / {dr[1]} deg: the servo nets were not trained for this camera.')
    if a.write:
        bak = spec_path.with_suffix(f'.json.bak{int(time.time())}')
        shutil.copy2(spec_path, bak)
        stamp = time.strftime('%Y-%m-%d')
        w['pose_in_link6'] = pose
        w['calibrated'] = f'{stamp}, {run} (scripts/real/handeye.py apply, joint offsets fixed)'
        w.setdefault('quality', {})['handeye_fixed_offsets'] = q
        if a.side == spec['wrist']['sim'].get('source', 'right'):
            spec['wrist']['sim']['pose_in_link6'] = pose
        spec['calibration_version'] = f'{stamp} wrist_{a.side} {run.name}'
        spec_path.write_text(json.dumps(spec, indent=1) + '\n')
        print(f'wrote {spec_path} (backup {bak.name}); commit it with the run summary')


def cmd_selftest(a):
    """Known X (the spec's right mount + an offset), arm poses looking at a table board, noisy pixels and joints."""
    rng = np.random.default_rng(a.seed)
    kin = C.Kin()
    spec = json.loads(rig.SPEC.read_text())
    X_true = T_of(spec['wrist']['right']['pose_in_link6']) @ C.T_from(cv2.Rodrigues(np.radians([0.8, -0.5, 0.6]))[0], [0.003, -0.002, 0.004])
    Y = C.T_from(np.eye(3), [0.30, 0.0, 0.0])                       # board on the table, 30 cm in front
    K = np.array([[683.6, 0, 640], [0, 683.6, 512], [0, 0, 1]])
    dist = np.zeros(5)
    Fs, corners, q_used = [], [], []
    tries = 0
    while len(Fs) < a.views and tries < 20000:
        tries += 1
        q = rng.uniform(kin.lo, kin.hi)
        F = kin.fk(q)
        Cb = C.T_inv(X_true) @ C.T_inv(F) @ Y                    # board in the camera
        if Cb[2, 3] < 0.12 or Cb[2, 3] > 0.45:
            continue
        rv, tv = C.rt_from_T(Cb)
        pr, _ = cv2.projectPoints(C.OBJ, rv, tv, K, dist)
        pr = pr.reshape(-1, 2)
        if (pr < 20).any() or (pr[:, 0] > 1260).any() or (pr[:, 1] > 1004).any() or Cb[:3, 2] @ np.array([0, 0, 1]) > -0.5:
            continue
        corners.append((pr + rng.normal(0, a.px_noise, pr.shape)).astype(np.float32))
        q_used.append(q)
        Fs.append(kin.fk(q + rng.normal(0, np.radians(a.joint_noise_deg), 6)))   # encoder noise / residual offsets
    X, _, qual = solve_fixed_offsets(Fs, corners, K, dist)
    emm, edeg = diff(X_true, X)
    ok = emm < a.tol_mm and edeg < a.tol_deg
    print(json.dumps({'views': len(Fs), 'px_noise': a.px_noise, 'joint_noise_deg': a.joint_noise_deg,
                      'error_mm': emm, 'error_deg': edeg, 'estimated_spread_mm': qual['fold_spread_mm'],
                      'estimated_spread_deg': qual['fold_spread_deg'], 'reproj_mean_px': qual['reproj_mean_px'],
                      'pass': ok}, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('apply')
    p.add_argument('--side', required=True, choices=('right', 'left'))
    p.add_argument('--run', required=True)
    p.add_argument('--write', action='store_true')
    p = sub.add_parser('selftest')
    p.add_argument('--views', type=int, default=36)
    p.add_argument('--px-noise', type=float, default=0.5)
    p.add_argument('--joint-noise-deg', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tol-mm', type=float, default=1.0)
    p.add_argument('--tol-deg', type=float, default=0.3)
    a = ap.parse_args()
    {'apply': cmd_apply, 'selftest': cmd_selftest}[a.cmd](a)
