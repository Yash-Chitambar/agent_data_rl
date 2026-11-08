"""Plug roll against the panel from one wrist image, without the camera mount or the in-hand model:
the shell's sleeve-end edge (a line across the shell) against the panel's across axis, both in the same undistorted frame.
Per frame: subpixel edge row per column across the shell -> line angle; prediction of that angle for a plug square to the panel at the
measured tip (panel pose R, t from port_vision) -> difference; divided by d(angle)/d(roll) of the same geometry -> roll (deg).
Compared with the align stage's logged rv_deg z (panel pose x fixed mount x mean grasp).
  PYTHONPATH=. python scripts/real/relvis_roll.py"""
import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
from deploy.bimanual import port_vision as PV

PROT, W, T = 5.8, PV.SHELL_W_EFF_MM, PV.SHELL_T_MM


def edge_line(und, pl):
    g = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    ve = pl['ve']; pts = []
    for u in range(int(pl['ul']) + 6, int(pl['ur']) - 5):
        col = g[int(ve) - 8:int(ve) + 9, u - 1:u + 2].mean(1)
        gr = -np.gradient(col)
        i = int(np.argmax(gr))
        if 1 <= i < len(gr) - 1 and gr[i] > PV.MIN_EDGE * 0.5:
            a, b, c = gr[i - 1], gr[i], gr[i + 1]; den = a - 2 * b + c
            pts.append((u, int(ve) - 8 + i + (0.5 * (a - c) / den if abs(den) > 1e-6 else 0)))
    if len(pts) < 15:
        return None
    p = np.array(pts, float)
    for _ in range(2):                       # fit, drop outliers, refit
        k, c0 = np.polyfit(p[:, 0], p[:, 1], 1)
        r = p[:, 1] - (k * p[:, 0] + c0)
        p = p[np.abs(r) < max(2.5 * r.std(), 0.3)]
    k, c0 = np.polyfit(p[:, 0], p[:, 1], 1)
    return np.degrees(np.arctan(k)), float(np.std(p[:, 1] - (k * p[:, 0] + c0))), len(p)


def pred_angle(K, R, tip_cam, roll_deg):
    Rp = R @ Rot.from_euler('z', roll_deg, degrees=True).as_matrix()
    top_end = tip_cam + Rp @ np.array([0, T / 2, PROT])
    a, b = top_end + Rp @ np.array([-W / 2, 0, 0]), top_end + Rp @ np.array([W / 2, 0, 0])
    pa, pb = (K @ a)[:2] / (K @ a)[2], (K @ b)[:2] / (K @ b)[2]
    return np.degrees(np.arctan2(pb[1] - pa[1], pb[0] - pa[0]))


def main():
    out = {}
    for d in sorted(glob.glob('data/real_runs/2026-11-0[67]/full_*')):
        fr = [json.loads(l) for l in open(f'{d}/frames.jsonl')]
        fr = [r for r in fr if r['cam'] == 'wrist_right' and r['kind'] == 'raw' and r['tag'].startswith('align_') and r['tag'][6:9].isdigit()]
        if not fr:
            continue
        ev = [json.loads(l) for l in open(f'{d}/events.jsonl')]
        rvz = [e['rv_deg'][2] for e in ev if e['kind'] == 'align_measure' and e.get('rv_deg') is not None]
        rolls = []
        for r in fr[::2]:
            raw = cv2.imread(f"{d}/{r['file']}")
            m = PV.measure(raw)
            if not m.ok:
                continue
            und, K = PV.undistort(raw)
            el = edge_line(und, m.debug['plug'])
            if el is None:
                continue
            R, tip_cam = np.asarray(m.debug['R']), np.asarray(m.debug['tip_cam'])
            a0 = pred_angle(K, R, tip_cam, 0.0)
            gain = (pred_angle(K, R, tip_cam, 1.0) - pred_angle(K, R, tip_cam, -1.0)) / 2
            rolls.append(((el[0] - a0) / gain, el[1], gain))
        if not rolls:
            continue
        a = np.array(rolls)
        last = a[-6:, 0]
        rec = dict(n=len(a), roll_img_deg_median=float(np.median(a[:, 0])), roll_img_sd=float(a[:, 0].std()), roll_img_last6_mean=float(last.mean()), roll_img_last6_sd=float(last.std()),
                   fit_resid_px_median=float(np.median(a[:, 1])), deg_image_per_deg_roll=float(np.median(a[:, 2])),
                   align_rv_z_median=float(np.median(rvz)) if rvz else None, align_rv_z_last=rvz[-1] if rvz else None, n_align=len(rvz))
        out[Path(d).name] = rec
        print(Path(d).name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()})
    D = np.array([[v['roll_img_last6_mean'], v['align_rv_z_last']] for v in out.values() if v['align_rv_z_last'] is not None])
    if len(D):
        diff = D[:, 0] - D[:, 1]
        out['summary'] = dict(n_runs=len(D), img_minus_align_mean=float(diff.mean()), img_minus_align_sd=float(diff.std(ddof=1)), img_mean=float(D[:, 0].mean()), img_sd_runs=float(D[:, 0].std(ddof=1)),
                              align_mean=float(D[:, 1].mean()), align_sd_runs=float(D[:, 1].std(ddof=1)), corr=float(np.corrcoef(D.T)[0, 1]) if len(D) > 2 else None)
        print(out['summary'])
    Path('results/real_align/relvis/roll_image.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
