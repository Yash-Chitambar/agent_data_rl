"""Scale and sign of the image roll (relvis_roll.py) against the FK roll labels of the collect sweep (one grasp: label changes are true
roll changes of the plug against the panel). Frames not moving, gap under 15 mm. Slope, residual SD, and the same for the align stage's
mount-based roll (convert of port_vision R) on the same frames.
  PYTHONPATH=. python scripts/real/relvis_roll_collect.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))
import cv2, numpy as np
from deploy.bimanual import port_vision as PV
from relvis_roll import edge_line, pred_angle

rows = [json.loads(l) for l in open('data/real_training/port_v1/manifest.jsonl')]
rows = [r for r in rows if r['source'] == 'collect' and not r['moving'] and r['p_mm'][2] < 15]
X = []
for r in rows:
    raw = cv2.imread(r['raw'])
    m = PV.measure(raw)
    if not m.ok:
        continue
    und, K = PV.undistort(raw)
    el = edge_line(und, m.debug['plug'])
    if el is None:
        continue
    R, tip = np.asarray(m.debug['R']), np.asarray(m.debug['tip_cam'])
    g = (pred_angle(K, R, tip, 1.0) - pred_angle(K, R, tip, -1.0)) / 2
    from scipy.spatial.transform import Rotation as Rot
    panel_rz = Rot.from_matrix(R).as_rotvec(degrees=True)[2]
    X.append([r['rv_deg'][2], (el[0] - pred_angle(K, R, tip, 0.0)) / g, panel_rz, el[1]])
X = np.array(X)
lab = X[:, 0] - X[:, 0].mean()
out = dict(n=len(X), label_range_deg=[float(X[:, 0].min()), float(X[:, 0].max())], label_sd=float(X[:, 0].std()))
for i, name in ((1, 'image_roll'), (2, 'panel_rv_z')):
    y = X[:, i]
    k, c = np.polyfit(lab, y, 1)
    res = y - (k * lab + c)
    # bootstrap 95 % CI of the slope
    bs = [np.polyfit(lab[j], y[j], 1)[0] for j in [np.random.default_rng(s).integers(0, len(X), len(X)) for s in range(1000)]]
    out[name] = dict(slope=float(k), slope_ci95=np.percentile(bs, [2.5, 97.5]).round(3).tolist(), resid_sd=float(res.std(ddof=2)), corr=float(np.corrcoef(lab, y)[0, 1]))
print(json.dumps(out, indent=1))
Path('results/real_align/relvis/roll_collect.json').write_text(json.dumps(out, indent=1))
