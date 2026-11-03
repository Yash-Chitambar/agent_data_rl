"""Annotated contact sheets of the port vision on real frames: results/real/port_vision/sheet_{success,fail}.jpg. Green: panel features. Red dot: port 2 centre.
Blue dot: plug tip. Numbers: vision and (if any) label, x across, y up, z gap, mm."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual import port_vision as PV   # noqa: E402

OUT = ROOT / 'results/real/port_vision'


def tile(r):
    raw = cv2.imread(r.raw)
    m = PV.measure(raw)
    und, _ = PV.undistort(raw)
    d = m.debug
    if d.get('lines'):
        for ps in d['lines'].values():
            for *_, u, v in ps:
                cv2.circle(und, (int(u), int(v)), 3, (0, 255, 0), 1)
    if m.port_px:
        cv2.circle(und, (int(m.port_px[0]), int(m.port_px[1])), 4, (0, 0, 255), -1)
    if m.plug_px:
        cv2.circle(und, (int(m.plug_px[0]), int(m.plug_px[1])), 4, (255, 128, 0), -1)
    c = und[200:640, 300:1000]
    tag = f'{r.run[-8:]} ' + (f'L {r.lx:.1f} {r.ly:.1f} {r.lz:.1f}' if pd.notna(r.lz) else 'no label')
    res = f'V {m.across_mm:.1f} {m.up_mm:.1f} {m.gap_mm:.1f}' if m.ok else 'FAIL ' + m.reason[:44]
    cv2.putText(c, tag, (6, 22), 0, .6, (0, 255, 255), 2)
    cv2.putText(c, res, (6, 46), 0, .6, (0, 255, 255) if m.ok else (0, 128, 255), 2)
    return cv2.resize(c, (420, 264), interpolation=cv2.INTER_AREA)


def grid(rows, name, ncol=4):
    t = [tile(r) for r in rows]
    while len(t) % ncol:
        t.append(np.zeros_like(t[0]))
    cv2.imwrite(str(OUT / name), np.vstack([np.hstack(t[i:i + ncol]) for i in range(0, len(t), ncol)]), [cv2.IMWRITE_JPEG_QUALITY, 75])


def main():
    d = pd.read_csv(OUT / 'eval_frames_full.csv')
    g = np.random.default_rng(7)
    good = []
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 25), (25, 500)):
        b = d[(d.ok == 1) & (d.lz >= lo) & (d.lz < hi)]
        good += [b.iloc[i] for i in g.choice(len(b), min(4, len(b)), replace=False)]
    un = d[(d.ok == 1) & (d.source == 'unlabeled')]
    good += [un.iloc[i] for i in g.choice(len(un), 4, replace=False)]
    grid(good, 'sheet_success.jpg')
    bad = d[(d.ok == 0) & (d.source != 'negative')]
    picks = []
    for reason, b in bad.groupby(bad.reason.str.replace(r'[\d\.\-]+', '#', regex=True).str[:24]):
        picks += [b.iloc[i] for i in g.choice(len(b), min(4, len(b)), replace=False)]
    n = d[d.source == 'negative']
    picks += [n.iloc[i] for i in g.choice(len(n), 4, replace=False)]
    grid(picks[:24], 'sheet_fail.jpg')


if __name__ == '__main__':
    main()
