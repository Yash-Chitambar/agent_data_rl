"""Contact sheet: the last align frame (hand-over) of each real align run, undistorted, with port_vision's plug tip (green),
port 2 (red), shell box (yellow) and the plug ROI (blue); plus a 3x zoom of the plug region (neck and shell).
  PYTHONPATH=. python scripts/real/relvis_sheet.py --out results/real_align/relvis/sheet_handover.jpg"""
import argparse, json, glob
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2, numpy as np
from deploy.bimanual import port_vision as PV

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); A = ap.parse_args()
tiles = []
for d in sorted(glob.glob('data/real_runs/2026-11-0[67]/full_*')):
    rows = [json.loads(l) for l in open(f'{d}/frames.jsonl')]
    rows = [r for r in rows if r['cam'] == 'wrist_right' and r['kind'] == 'raw' and r['tag'].startswith('align_') and r['tag'][6:9].isdigit()]
    if not rows:
        continue
    r = rows[-1]
    raw = cv2.imread(f"{d}/{r['file']}")
    und, K = PV.undistort(raw)
    m = PV.measure(raw)
    img = und.copy()
    u0, u1, v0, v1 = PV.PLUG_ROI
    cv2.rectangle(img, (u0, v0), (u1, v1), (255, 0, 0), 2)
    pl = (m.debug or {}).get('plug')
    if pl:
        b = PV.plug_box(pl); cv2.rectangle(img, b[:2], b[2:], (0, 255, 255), 1)
    for p, c in ((m.plug_px, (0, 255, 0)), (m.port_px, (0, 0, 255))):
        if p is not None:
            cv2.circle(img, (int(p[0]), int(p[1])), 6, c, 2)
    zoom = cv2.resize(und[v0 - 60:v1 + 60, u0 - 80:u1 + 80], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(img, (640, 512))
    zoom = cv2.resize(zoom, (int(zoom.shape[1] * 512 / zoom.shape[0]), 512))
    t = np.hstack([small, zoom])
    txt = f"{Path(d).name} {r['tag']} ok={m.ok} " + (f"x {m.across_mm:.1f} y {m.up_mm:.1f} z {m.gap_mm:.1f}" if m.ok else m.reason[:40])
    cv2.putText(t, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(t)
    print(txt, 'shell w px', None if not pl else round(pl['w'], 1))
w = max(t.shape[1] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
sheet = np.vstack(tiles[:8])
cv2.imwrite(A.out, cv2.resize(sheet, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
