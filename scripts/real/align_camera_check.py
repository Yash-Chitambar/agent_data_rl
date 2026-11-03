"""Right wrist camera and port_vision only. Connects no arm: nothing can move. Prints the measurement of the newest raw frames and saves them.

    PYTHONPATH=. .venv/bin/python scripts/real/align_camera_check.py --n 30 --out data/real_runs/$(date +%F)/camera_check

The arm does not move in this tool, so the port is in view only if the operator holds the PC or a spare panel in view (or the arm rests where it sees it). It checks the camera, the frame format and the speed of
the measurement. Numbers mean something only with the port row in view at about 10-40 mm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--out', default=None)
    ap.add_argument('--image', action='append', default=[], help='offline: raw jpg files instead of the camera')
    a = ap.parse_args()
    from deploy.bimanual.port_vision import measure
    cam = None
    if not a.image:
        from deploy.cameras import WristCam
        cam = WristCam('right')
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for i in range(a.image and len(a.image) or a.n):
            if cam is None:
                f = cv2.imread(a.image[i])
            else:
                f = None
                t0 = time.monotonic()
                while f is None and time.monotonic() - t0 < 3.:
                    time.sleep(.05)
                    f = cam.latest()[0]
                if f is None:
                    raise SystemExit('no frame from the right wrist camera')
            t0 = time.perf_counter()
            m = measure(f, side='right')
            ms = (time.perf_counter() - t0) * 1e3
            rows.append(dict(i=i, ok=m.ok, reason=m.reason, conf=m.conf, v=[m.across_mm, m.up_mm, m.gap_mm], sigma=m.sigma_mm, ms=ms))
            print(f'{i:3d} ok {m.ok} conf {m.conf:.2f} across {m.across_mm} up {m.up_mm} gap {m.gap_mm} ({ms:.0f} ms) {m.reason}', flush=True)
            if out and i % 5 == 0:
                cv2.imwrite(str(out / f'{i:04d}.jpg'), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        if cam is not None:
            cam.close()
    good = [r for r in rows if r['ok']]
    s = dict(n=len(rows), ok=len(good), ms_p50=float(np.median([r['ms'] for r in rows])) if rows else None)
    if good:
        V = np.array([r['v'] for r in good], float)
        s.update(median=np.median(V, 0).round(2).tolist(), spread=(V.max(0) - V.min(0)).round(2).tolist())
    print(json.dumps(s))
    if out:
        (out / 'summary.json').write_text(json.dumps(dict(summary=s, rows=rows), indent=1))


if __name__ == '__main__':
    main()
