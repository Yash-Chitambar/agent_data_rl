"""Joint backlash of the real YAM: encoder (motor side) vs wrist camera on a fixed board (link side).

The encoders sit on the motor side, so they cannot see gear play. Here the arm parks its wrist
camera over a checkerboard lying flat and still on the table, then steps one joint at a time
up and down in small stairs. At each stair it records the encoder and the camera's rotation
from the board (solvePnP, median of several frames). A joint with backlash shows a hysteresis
loop: after each reversal the motor turns through the play before the link (camera) follows.

    # MOVES the arm (small steps only: default 3 mrad, at most +/-9 mrad from the viewing pose)
    .venv/bin/python scripts/real/measure_backlash.py run --side right --out data/backlash/right_20261030
    .venv/bin/python scripts/real/measure_backlash.py run --side left  --out data/backlash/left_20261030
    # offline, any machine with the repo
    .venv/bin/python scripts/real/measure_backlash.py analyze data/backlash/right_20261030

Setup: bases clamped; the calibration board (calib.py BOARD, 15 mm squares) flat on the table
in front of the arm, taped down so it cannot move; rig console (rigctl.py) or estop.py running.
`--hint x y` (m, this arm's base frame) is where the board centre roughly is.

Joints 2-3 carry gravity load in one direction, which keeps their gears on one flank: their
loop width is the play that survives that preload (expected small). Joints 1, 4, 5, 6 carry
little static load and show the free play.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

ARCMIN = 180 / np.pi * 60


def stair(step, levels, cycles):
    """Offsets: preload downward, then up/down stairs; every reversal is approached the same way."""
    up = [k * step for k in range(-levels, levels + 1)]
    seq = [-levels * step]
    for _ in range(cycles):
        seq += up[1:] + up[::-1][1:]
    return seq + [0.0]


def cmd_run(a):
    import calib
    import estop
    import safety
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty: choose a new name')
    out.mkdir(parents=True, exist_ok=True)
    kin, checker = calib.Kin(), calib.calib_checker(calib.PLAN_CLEARANCE)
    X, K, dist = calib.cam_prior(kin, a.side)
    wcam = calib.WristCam(a.side)
    record = {'side': a.side, 'step_rad': a.step, 'levels': a.levels, 'cycles': a.cycles, 'settle_s': a.settle,
              'frames_per_stair': a.frames, 'K': K.tolist(), 'dist': None if dist is None else dist.tolist(),
              'board': {'inner_corners': calib.BOARD, 'square_m': calib.SQUARE}, 'joints': {}, 'result': 'incomplete'}

    def observe(mover):
        rot, reproj = [], []
        for _ in range(a.frames):
            c = calib.detect(cv2.cvtColor(wcam.grab(), cv2.COLOR_BGR2GRAY))
            if c is None:
                continue
            T, err = calib.board_pose(c, K, dist)
            rot.append(cv2.Rodrigues(T[:3, :3])[0].ravel().tolist()); reproj.append(err)
        return mover.joints().tolist(), rot, reproj

    def body(mover, q_rest):
        R_ref = (kin.fk(q_rest[:6]) @ X)[:3, :3]
        q_view = None
        for k, ((x, y), T_cam) in enumerate(calib.search_poses(a.side, kin, R_ref, a.hint)[:a.max_tries]):
            q = kin.ik(T_cam @ calib.T_inv(X), seeds=[q_rest[:6], [0, 1.0, 1.0, -0.5, 0, 0], [0, 1.6, 1.3, -0.9, 0, 0]])
            if q is None or checker.violation(q) or not mover.lateral_ok(q):
                continue
            if mover.path_ok(mover.arm.read()[:6], q, check=mover.path_check):
                continue
            print(f'candidate {k}: looking at ({x:.2f}, {y:+.2f})', flush=True)
            mover.move_to(q)
            if calib.detect(cv2.cvtColor(wcam.grab(), cv2.COLOR_BGR2GRAY)) is not None:
                q_view = mover.joints()[:6]
                break
        if q_view is None:
            raise SystemExit('board not seen from any safe candidate pose; check --hint')
        record['q_view'] = q_view.tolist()
        excursion = a.step * a.levels
        for j in a.joints:
            rows = []
            for off in stair(a.step, a.levels, a.cycles):
                if estop.stopped():
                    raise safety.SafetyStop('software stop')
                q = q_view.copy(); q[j - 1] += off
                mover.move_to(q, max_vel=0.05)
                time.sleep(a.settle)
                enc, rot, reproj = observe(mover)
                rows.append({'offset_cmd_rad': off, 'q_encoder': enc, 'board_rotvec_in_cam': rot, 'reproj_px': reproj})
                if not rot:
                    print(f'  joint {j}: board lost at offset {off:+.4f}; skipping the rest of this joint', flush=True)
                    break
            record['joints'][str(j)] = rows
            mover.move_to(q_view, max_vel=0.05)
            print(f'joint {j}: {len(rows)} stairs (+/-{excursion * 1e3:.0f} mrad)', flush=True)
            (out / 'backlash.json').write_text(json.dumps(record, indent=1) + '\n')
        record['result'] = 'completed'

    try:
        calib.supervised(a.side, kin, checker, body)
    finally:
        wcam.close()
        (out / 'backlash.json').write_text(json.dumps(record, indent=1) + '\n')
        print(f"{record['result']}: {out / 'backlash.json'}", flush=True)


def analyze_joint(rows, j):
    """Link angle (camera rotation about the dominant axis) vs motor angle (encoder), with the
    direction of the last motor move; the loop width (motor units) is the backlash."""
    rows = [r for r in rows if r['board_rotvec_in_cam']]
    if len(rows) < 6:
        return None
    R0 = cv2.Rodrigues(np.median(rows[0]['board_rotvec_in_cam'], axis=0))[0]
    rel, noise = [], []
    for r in rows:
        vs = [cv2.Rodrigues((cv2.Rodrigues(np.array(v))[0] @ R0.T))[0].ravel() for v in r['board_rotvec_in_cam']]
        rel.append(np.median(vs, axis=0)); noise.append(np.std(vs, axis=0).max() if len(vs) > 1 else np.nan)
    rel = np.array(rel)
    axis = np.linalg.svd(rel - rel.mean(0))[2][0]
    motor = np.array([r['q_encoder'][j - 1] for r in rows])
    motor -= motor[0]
    link = rel @ axis
    if np.corrcoef(link, motor)[0, 1] < 0:
        link = -link
    move = np.diff(motor, prepend=motor[0])
    direction = np.zeros_like(motor)
    last = 0.0
    for i, m in enumerate(move):
        if abs(m) > 2e-4:
            last = np.sign(m)
        direction[i] = last
    ok = direction != 0
    A = np.column_stack([motor[ok], np.ones(ok.sum()), direction[ok]])
    (slope, icpt, half), *_ = np.linalg.lstsq(A, link[ok], rcond=None)
    resid = link[ok] - A @ np.array([slope, icpt, half])
    width = 2 * abs(half) / max(abs(slope), 1e-6)       # motor travel inside the play
    return {'joint': j, 'stairs': len(rows), 'link_per_motor_slope': round(float(slope), 4),
            'backlash_arcmin': round(float(width * ARCMIN), 2), 'backlash_mrad': round(float(width * 1e3), 3),
            'fit_residual_arcmin': round(float(np.std(resid) * ARCMIN), 2),
            'frame_noise_arcmin': round(float(np.nanmedian(noise) * ARCMIN), 2),
            'curve': {'motor_rad': motor.tolist(), 'link_rad': link.tolist(), 'direction': direction.tolist()}}


def cmd_analyze(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = Path(a.dir)
    rec = json.loads((d / 'backlash.json').read_text())
    res = {'side': rec['side'], 'result': rec['result'], 'joints': {}}
    js = sorted(rec['joints'], key=int)
    fig, axes = plt.subplots(1, max(1, len(js)), figsize=(4 * max(1, len(js)), 3.6), squeeze=False)
    for ax, j in zip(axes[0], js):
        r = analyze_joint(rec['joints'][j], int(j))
        if r is None:
            ax.set_title(f'joint {j}: too few stairs'); continue
        c = r.pop('curve'); res['joints'][j] = r
        m, l, dr = (np.array(c[k]) * 1e3 for k in ('motor_rad', 'link_rad', 'direction'))
        ax.plot(m, l, '-', color='0.7', lw=.8)
        for s, col, lab in ((1e3, 'C0', 'after up-move'), (-1e3, 'C3', 'after down-move')):
            ax.plot(m[dr == s], l[dr == s], 'o', ms=3, color=col, label=lab)
        ax.set(title=f'joint {j}: {r["backlash_arcmin"]} arcmin', xlabel='motor (encoder) mrad', ylabel='link (camera) mrad')
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{rec["side"]} arm backlash (noise per stair ~{np.nanmedian([v["frame_noise_arcmin"] for v in res["joints"].values()] or [np.nan]):.2f} arcmin)')
    fig.tight_layout(); fig.savefig(d / 'backlash.png', dpi=120); plt.close(fig)
    (d / 'analysis.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--side', choices=['left', 'right'], required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--joints', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    r.add_argument('--step', type=float, default=0.003, help='stair height (rad)')
    r.add_argument('--levels', type=int, default=3, help='stairs each side of the viewing pose')
    r.add_argument('--cycles', type=int, default=2)
    r.add_argument('--settle', type=float, default=0.8)
    r.add_argument('--frames', type=int, default=5)
    r.add_argument('--hint', type=float, nargs=2, help='board centre x y (m) in this arm base frame')
    r.add_argument('--max-tries', type=int, default=12)
    an = sub.add_parser('analyze')
    an.add_argument('dir')
    a = ap.parse_args()
    if a.cmd == 'run':
        if not 0 < a.step <= 0.005 or not 1 <= a.levels <= 4:
            raise SystemExit('keep --step <= 0.005 rad and --levels in 1..4 (max excursion 20 mrad)')
        cmd_run(a)
    else:
        cmd_analyze(a)


if __name__ == '__main__':
    main()
