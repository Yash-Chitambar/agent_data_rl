"""Timing on this machine: ACT and insertion inference per 50 Hz tick, and (--cameras, real rig only)
camera frame rate and the age of the frame the runner would use.

Budget: one tick is 20 ms. The sim queries ACT every tick (temporal ensembling); the measured loop
latency without ACT is ~4 ms (FINDINGS 2026-10-25). The ACT demos assume a 30 fps frame clock and
30-100 ms frame age (deploy/cameras.py: the logged age is arrival -> use, roughly true age - 30 ms).

    .venv/bin/python -m deploy.checks.timing [--device cuda:0|cpu] [--cameras --seconds 10]
"""
import argparse
import time

import numpy as np
import torch

from deploy import conventions as C
from deploy.runner import load_config
from deploy.stages import ActStage, InsertStage


def stats(x):
    x = np.asarray(x) * 1e3
    return f"p50 {np.median(x):6.2f} ms, p99 {np.percentile(x, 99):6.2f} ms, max {x.max():6.2f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--cameras", action="store_true")
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(2)
    robot = C.Robot("cpu")
    act = ActStage(cfg["models"]["act"], a.device, cfg["act"]["ensemble"])
    act.reset()
    q = np.array(cfg["act"]["start_q_model"])
    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)  # noqa: E731
    ts = []
    for k in range(a.ticks + 20):
        top, wrist = img(), img()
        t0 = time.perf_counter()
        out = act.step(q, 0.03, top, wrist)
        ts.append(time.perf_counter() - t0)
        act.sent(out[:6])
    print(f"ACT step ({a.device}, ensembled, incl. upload): {stats(ts[20:])}")
    ins = InsertStage(cfg["models"]["insert"], cfg["models"]["insert_config"], robot)
    ins.start(q, np.zeros(6))
    ts = []
    for k in range(a.ticks):
        t0 = time.perf_counter()
        x = ins.act()
        ins.push(x, q, np.zeros(6))
        ts.append(time.perf_counter() - t0)
    print(f"insertion step (cpu, obs + forward): {stats(ts)}")
    if a.cameras:
        from deploy.cameras import Cameras
        cams = Cameras(cfg["side"], cfg["cameras"]["res"], max_age_s=10.0)
        try:
            ages, n0, t0 = {"top": [], "wrist": []}, None, time.monotonic()
            pre = []
            while time.monotonic() - t0 < a.seconds:
                t1 = time.perf_counter()
                _, age, n = cams.frames()
                pre.append(time.perf_counter() - t1)
                n0 = n0 or n
                for c in ages:
                    ages[c].append(age[c])
                time.sleep(cfg["control_dt"])
            dt = time.monotonic() - t0
            for c in ages:
                print(f"{c}: {(n[c] - n0[c]) / dt:5.1f} fps, frame age at use {stats(ages[c])}")
            print(f"preprocessing both frames: {stats(pre)}")
        finally:
            cams.close()


if __name__ == "__main__":
    main()
