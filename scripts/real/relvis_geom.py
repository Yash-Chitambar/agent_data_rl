"""Geometric sensitivity of port_vision's direct plug-tip-to-port measurement to the grasp (no image detector: its arithmetic only).
Camera, panel pose and nominal tip from real hand-over frames (port_vision.measure debug). Shell model from the measured reader
(protrusion 5.8 mm, data/real_yam/usbc_lab_20261027/measurements.csv) and the USB-C spec shell (8.25 x 2.4 mm; effective rim width 8.15).
The plug turns about the neck end (the grasp side) by pitch / yaw / roll and shifts by dx, dy, dz in panel axes; the 'detector' sees
the projected rim edges and the sleeve end; measure()'s arithmetic gives (across, up, gap). Error = output - true tip centre, minus the
nominal case's output - truth (the empirical constants absorb the nominal bias).
Compared with 'port only + in-hand estimate': its error = the in-hand estimate error itself (1:1 in position).
  PYTHONPATH=. python scripts/real/relvis_geom.py"""
import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
from deploy.bimanual import port_vision as PV

PROT = 5.8                        # measured: neck edge -> tip
W, T = PV.SHELL_W_EFF_MM, PV.SHELL_T_MM


def proj(K, X):
    x = K @ X
    return x[:2] / x[2]


def shell_points(tip, Rp):
    """Rp columns: plug x (across), y (up), z (axis, tip -> neck). Rim edges at the top face: left/right lines; sleeve end edge."""
    top = tip + Rp @ np.array([0, T / 2, 0])
    pts = {}
    for s, k in ((-1, 'l'), (1, 'r')):
        pts[k] = [top + Rp @ np.array([s * W / 2, 0, a]) for a in np.linspace(0.3 * PROT, 0.9 * PROT, 5)]   # rows the detector averages (near the sleeve end)
    pts['end'] = top + Rp @ np.array([0, 0, PROT])      # sleeve end, top face centre
    pts['end_low'] = tip + Rp @ np.array([0, -T / 2, PROT])
    return pts


def emulate(K, R, t, tip_true, Rp, model_pc):
    p = shell_points(tip_true, Rp)
    ul = np.mean([proj(K, X)[0] for X in p['l']]); ur = np.mean([proj(K, X)[0] for X in p['r']])
    w = ur - ul
    ve = max(proj(K, p['end'])[1], proj(K, p['end_low'])[1])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = 0.5 * (fx + fy) * W / w
    u, v = 0.5 * (ul + ur), ve - PV.TIP_ABOVE_END_PX * PV.Z0_TIP / Z
    Zt = Z + PV.TIP_DEEPER_MM
    tip_top = np.array([(u - cx) / fx * Zt, (v - cy) / fy * Zt, Zt])
    tip = tip_top + R @ np.array([0.0, -T / 2, 0.0])
    return R.T @ (tip - t) - model_pc, Z


rows = []
for d in sorted(glob.glob('data/real_runs/2026-11-0[67]/full_*')):
    fr = [json.loads(l) for l in open(f'{d}/frames.jsonl')]
    fr = [r for r in fr if r['cam'] == 'wrist_right' and r['kind'] == 'raw' and r['tag'].startswith('align_') and r['tag'][6:9].isdigit()]
    if not fr:
        continue
    raw = cv2.imread(f"{d}/{fr[-1]['file']}")
    m = PV.measure(raw)
    if not m.ok:
        continue
    _, K = PV.undistort(raw)
    rows.append((Path(d).name, K, np.asarray(m.debug['R']), np.asarray(m.debug['t']), np.asarray(m.debug['tip_cam']), np.array([m.across_mm, m.up_mm, m.gap_mm])))
model_pc = np.asarray(PV.load_model().get('port_center_in_model_mm', PV.PORT_CENTER_MM), float)

PERT = {'pitch': ('x', 'deg'), 'yaw': ('y', 'deg'), 'roll': ('z', 'deg'), 'dx': (0, 'mm'), 'dy': (1, 'mm'), 'dz': (2, 'mm')}
out = {}
for name, K, R, t, tip_cam, meas in rows:
    tip_true0 = R.T @ (tip_cam - t) - model_pc          # nominal truth := this frame's tip (panel frame, port-centred)
    Rp0 = R                                                # plug axes = panel axes (plug square to the panel)
    def run(dR=np.eye(3), dp=np.zeros(3)):
        neck = tip_cam + Rp0 @ np.array([0, 0, PROT])
        Rp = R @ dR @ R.T @ Rp0
        tip_c = neck + Rp @ np.array([0, 0, -PROT]) + R @ dp
        truth = R.T @ (tip_c - t) - model_pc
        X, Z = emulate(K, R, t, tip_c, Rp, model_pc)
        return X - truth
    e0 = run()
    res = {}
    for k, (ax, unit) in PERT.items():
        for a in (-5.0, 5.0) if unit == 'deg' else (-3.0, 3.0):
            if unit == 'deg':
                e = run(dR=Rot.from_euler(ax, a, degrees=True).as_matrix())
            else:
                dp = np.zeros(3); dp[ax] = a; e = run(dp=dp)
            res[f'{k}{a:+.0f}'] = (e - e0).round(3).tolist()
    out[name] = dict(nominal_bias_mm=e0.round(3).tolist(), sens=res)
agg = {}
for k in out[rows[0][0]]['sens']:
    a = np.array([out[n]['sens'][k] for n in out])
    agg[k] = dict(max_abs_mm=np.abs(a).max(0).round(3).tolist())
print('runs', len(out))
for k, v in agg.items():
    print(f'{k:10s} max |error| across, up, gap (mm): {v["max_abs_mm"]}')
Path('results/real_align/relvis/geom_sensitivity.json').write_text(json.dumps(dict(per_run=out, agg=agg, n_runs=len(out)), indent=1))
