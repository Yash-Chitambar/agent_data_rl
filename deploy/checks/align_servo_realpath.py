"""The servo's robot-only vision path on real frames, no arm: PortVisionSource.prepare (shell depth from 9 frames) through a fake chain that serves
logged raw wrist frames of a real run, then each frame through the child-process worker (vision_thread.ProcessWorker + PortVisionSource.convert)
against the direct inline call (PortVisionSource.measure_raw). They must agree exactly (same function, same depth), including the plug angles.

  PYTHONPATH=. python -m deploy.checks.align_servo_realpath [--run /home/yashchitambar/peg_test/data/real_runs/2026-11-06/full_173941_align_full9]
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np

from deploy.bimanual.align import PortMeasure, PortVisionSource
from deploy.bimanual.vision_thread import ProcessWorker


class FakeIO:
    is_sim = False

    def __init__(self, files):
        self.files, self.i = files, 0

    def fresh_frames(self, names=('wrist_right',)):
        f = self.files[self.i % len(self.files)]
        self.i += 1
        raw = cv2.imread(f)
        return {'wrist_right': (None, raw, time.monotonic(), self.i)}

    frames = fresh_frames


class FakeChain:
    def __init__(self, files):
        self.io, self.logger, self.k, self.state = FakeIO(files), None, 0, {'t': 0.}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/home/yashchitambar/peg_test/data/real_runs/2026-11-06/full_173941_align_full9')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--out')
    a = ap.parse_args()
    depth = sorted(glob.glob(f'{a.run}/frames/wrist_right/*_align_depth_*.jpg'))
    meas = sorted(glob.glob(f'{a.run}/frames/wrist_right/*_align_0*.jpg'))[:a.n]
    ch = FakeChain(depth)
    src = PortVisionSource(ch, {'kind': 'port_vision'})
    pr = src.prepare(ch, lambda: None)
    logged = [json.loads(x) for x in open(f'{a.run}/events.jsonl') if '"align_prepare"' in x]
    out = dict(prepare=pr, logged_prepare=logged[0] if logged else None, frames=[])
    w = ProcessWorker(src.kw, lambda: src.plug_z, src.convert, lambda e: PortMeasure(False, str(e)))
    worst = 0.
    for i, f in enumerate(meas):
        raw = cv2.imread(f)
        m0 = src.measure_raw(raw)
        w.submit(raw, i, 0., f)
        res = None
        t0 = time.time()
        while res is None and time.time() - t0 < 10:
            res = w.poll()
            time.sleep(.005)
        m1 = res.m
        d = None
        if m0.ok and m1.ok:
            d = float(np.abs(m0.vec() - m1.vec()).max())
            if m0.rv_deg is not None and m1.rv_deg is not None:
                d = max(d, float(np.abs(np.asarray(m0.rv_deg) - np.asarray(m1.rv_deg)).max()))
            worst = max(worst, d)
        out['frames'].append(dict(file=Path(f).name, inline_ok=m0.ok, proc_ok=m1.ok, same_ok=m0.ok == m1.ok, max_abs_diff=d, v=None if not m1.ok else m1.vec().round(3).tolist(),
                                  rv_deg=None if m1.rv_deg is None else np.round(m1.rv_deg, 3).tolist(), proc_ms=round(res.ms, 1)))
    w.close()
    out['all_same_ok'] = all(x['same_ok'] for x in out['frames'])
    out['worst_abs_diff'] = worst
    out['pass'] = out['all_same_ok'] and worst < 1e-9 and bool(pr.get('accepted'))
    print(json.dumps({k: out[k] for k in ('prepare', 'all_same_ok', 'worst_abs_diff', 'pass')}))
    for x in out['frames']:
        print(json.dumps(x))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
