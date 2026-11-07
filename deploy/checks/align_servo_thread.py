"""Vision worker timing for the servo align mode (deploy/bimanual/vision_thread.py): a 50 Hz loop with per-tick torch work (as the chain's FK,
Jacobian and logging) runs beside the worker; the worker measures one frame at a time. Reports the loop period (p50 / p99 / max), late ticks
(> 25 ms), frames per second and frame age at use. Backends: sleep (fake 82 ms, GIL free), cpu (fake 82 ms of numpy work), real
(deploy/bimanual/port_vision.measure on a real raw wrist frame of 2026-11-06 align_full9, its shell depth 119.47 mm) in a thread and in a
child process (vision_thread.ProcessWorker, the servo's default). No arm.

  PYTHONPATH=. python -m deploy.checks.align_servo_thread [--seconds 10] [--out results/real_align/servo_deploy/thread_timing.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from deploy.bimanual.vision_thread import ThreadWorker

ROOT = Path(__file__).resolve().parents[2]
FRAME = Path('/home/yashchitambar/peg_test/data/real_runs/2026-11-06/full_173941_align_full9/frames/wrist_right/000172_align_000_0.jpg')


def fake_sleep(raw):
    time.sleep(.082)
    return dict(ok=True)


def fake_cpu(raw):
    t0 = time.perf_counter()
    x = np.random.rand(256, 256)
    while time.perf_counter() - t0 < .082:
        x = x @ x.T / 256.
    return dict(ok=True)


def run(fn, seconds, raw, worker=None):
    w = worker or ThreadWorker(fn, lambda e: dict(ok=False, err=str(e)))
    A = torch.randn(64, 6, 6)
    periods, ages, n_res, late = [], [], 0, 0
    t_next = time.monotonic()
    t_prev = None
    last_cap = None
    k = 0
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        _ = torch.linalg.solve(A @ A.transpose(1, 2) + torch.eye(6), torch.randn(64, 6, 1))      # per-tick numeric work
        res = w.poll()
        if res is not None:
            n_res += 1
            last_cap = res.t_capture
        if last_cap is not None:
            ages.append(time.monotonic() - last_cap)
        if not w.busy:
            w.submit(raw, k, time.monotonic(), f'f{k}')
        t_next += .02
        now = time.monotonic()
        if t_next > now:
            time.sleep(t_next - now)
        else:
            late += 1
            t_next = now
        t = time.monotonic()
        if t_prev is not None:
            periods.append(t - t_prev)
        t_prev = t
        k += 1
    w.close()
    p = np.asarray(periods) * 1e3
    a = np.asarray(ages) * 1e3
    return dict(ticks=k, period_ms_p50_p99_max=[round(float(np.percentile(p, 50)), 2), round(float(np.percentile(p, 99)), 2), round(float(p.max()), 2)],
                ticks_over_25ms=int((p > 25).sum()), late=late, frames_per_s=round(n_res / seconds, 1),
                frame_age_ms_p50_p99_max=[round(float(np.percentile(a, 50)), 1), round(float(np.percentile(a, 99)), 1), round(float(a.max()), 1)] if len(a) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=10.)
    ap.add_argument('--out')
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = {}
    out['sleep'] = run(fake_sleep, a.seconds, None)
    out['cpu'] = run(fake_cpu, a.seconds, None)
    if FRAME.exists():
        import cv2
        from deploy.bimanual import port_vision as PV
        raw = cv2.imread(str(FRAME))
        t0 = time.perf_counter()
        m = PV.measure(raw, side='right', plug_z_mm=119.47)
        out['real_single_call'] = dict(ms=round((time.perf_counter() - t0) * 1e3, 1), ok=bool(m.ok), v=[m.across_mm, m.up_mm, m.gap_mm])
        out['real_thread'] = run(lambda r: PV.measure(r, side='right', plug_z_mm=119.47), a.seconds, raw)
        from deploy.bimanual.vision_thread import ProcessWorker
        pw = ProcessWorker({}, lambda: 119.47, lambda m: m, lambda e: dict(ok=False, err=str(e)))
        out['real_process'] = run(None, a.seconds, raw, worker=pw)
        out['real_process']['last_ok'] = None
        out['baseline_no_worker'] = run(lambda r: dict(ok=True), a.seconds, None)
    for k, v in out.items():
        print(k, json.dumps(v))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
