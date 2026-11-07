"""Plug roll from the image alone, no grasp model (docs/real/K2_PROGRESS.md, "Real-to-sim replay and roll").

In each raw right-wrist align frame: the tilt of the shell's sleeve end (its lower edge, sub-pixel row per column across
the shell, line fit) against the tilt of the panel's across axis projected at port 2 (port_vision panel pose). Both are
the "across" direction when the plug has no roll, so their difference (image deg) is a roll measure that does not use the
mean-grasp / mount rotation that the align stage's rv_deg relies on. Per run: median difference and the run's logged rz.
Across runs: the relation between the two (slope, offset). The image scale of roll (deg of image tilt per deg of roll)
is not calibrated here: read the sign and the across-run relation, not the absolute value.

  python scripts/real/roll_image_check.py --out results/real_align/roll/roll_image.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = pathlib.Path('/home/yashchitambar/peg_test/data/real_runs')


def sleeve_tilt(und, pl):
    """Tilt (deg, image, + = right end lower) of the shell's lower edge from per-column gradient peaks."""
    u0, u1 = int(round(pl['ul'])) + 5, int(round(pl['ur'])) - 5
    ve = pl['ve']
    v0, v1 = int(ve) - 7, int(ve) + 8
    g = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    us, vs = [], []
    for u in range(u0, u1):
        col = g[v0:v1, u]
        d = col[:-2] - col[2:]                   # bright over dark going down: positive
        i = int(np.argmax(d))
        if d[i] < 8 or i == 0 or i == len(d) - 1:
            continue
        a, b, c = d[i - 1], d[i], d[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0
        us.append(u); vs.append(v0 + i + 1 + off)
    if len(us) < 15:
        return None
    us, vs = np.array(us, float), np.array(vs, float)
    for _ in range(2):                           # robust refit
        p = np.polyfit(us, vs, 1)
        r = vs - np.polyval(p, us)
        k = np.abs(r) < max(2.5 * np.std(r), 0.3)
        us, vs = us[k], vs[k]
    p = np.polyfit(us, vs, 1)
    return float(np.degrees(np.arctan(p[0]))), len(us)


def panel_axis_tilt(K, R, t, model_center):
    c = np.asarray(model_center, float)
    pts = np.stack([c - [5, 0, 0], c + [5, 0, 0]])           # panel model mm, across axis
    X = (R @ pts.T).T + t
    uv = (K @ X.T).T
    uv = uv[:, :2] / uv[:, 2:]
    du, dv = uv[1] - uv[0]
    return float(np.degrees(np.arctan2(dv, du)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results/real_align/roll/roll_image.json'))
    ap.add_argument('--max-frames', type=int, default=40)
    a = ap.parse_args()
    from deploy.bimanual import port_vision as PV
    model = PV.load_model()
    center = model.get('port_center_in_model_mm', PV.PORT_CENTER_MM)
    rows = []
    for r in sorted(glob.glob(str(DATA / '2026-11-0*' / 'full_*'))):
        r = pathlib.Path(r)
        if 'dry' in r.name:
            continue
        ev = [json.loads(l) for l in open(r / 'events.jsonl')]
        ms = [e for e in ev if e['kind'] == 'align_measure' and e.get('rv_deg') is not None]
        if not ms:
            continue
        fr = [json.loads(l) for l in open(r / 'frames.jsonl')]
        fr = [f for f in fr if f.get('cam') == 'wrist_right' and f.get('kind') == 'raw' and f['tag'].startswith('align_') and not f['tag'].startswith('align_depth')]
        fr = fr[-a.max_frames:]
        diffs = []
        for f in fr:
            img = cv2.imread(str(r / f['file']))
            if img is None:
                continue
            und, K = PV.undistort(img, 'right')
            pl, _ = PV.detect_plug(und)
            if pl is None:
                continue
            pose, _ = PV.panel_pose(und, K, PV.plug_box(pl), PV.MAX_REPROJ_PX, model)
            if pose is None:
                continue
            st = sleeve_tilt(und, pl)
            if st is None:
                continue
            diffs.append(st[0] - panel_axis_tilt(K, pose['R'], pose['t'], center))
        if len(diffs) < 5:
            continue
        rz = np.array([e['rv_deg'][2] for e in ms[-8:]], float)
        d = np.array(diffs)
        row = dict(run=r.name, day=r.parent.name, n_frames=len(d), image_diff_deg_median=float(np.median(d)), image_diff_sd=float(d.std()),
                   rz_log_median=float(np.median(rz)))
        rows.append(row)
        print(f"{row['day']} {row['run']:28s} n {row['n_frames']:2d} image plug-minus-panel tilt {row['image_diff_deg_median']:+.2f} (sd {row['image_diff_sd']:.2f})  logged rz {row['rz_log_median']:+.2f}")
    if len(rows) >= 3:
        x = np.array([r['rz_log_median'] for r in rows]); y = np.array([r['image_diff_deg_median'] for r in rows])
        p = np.polyfit(x, y, 1)
        cc = float(np.corrcoef(x, y)[0, 1])
        summ = dict(n_runs=len(rows), slope=float(p[0]), offset=float(p[1]), corr=cc, image_diff_mean=float(y.mean()), image_diff_sd=float(y.std()), rz_mean=float(x.mean()))
        print('across runs:', json.dumps(summ))
    else:
        summ = {}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(dict(runs=rows, summary=summ), indent=1))


if __name__ == '__main__':
    main()
