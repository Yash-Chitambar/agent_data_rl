"""Phase E E2 (capture radius) and E3 (clearance sweep) for one USB-C checkpoint.

E2: start offsets uniform in a +-6 mm square (--halfwidth; 8,192 episodes, clearance 50 um), binned by the
actual radial start distance; logistic fit success ~ sigmoid(a - b r), r50 = a / b, bootstrap CI.
E3: clearance 0.02, 0.04, 0.05, 0.07, 0.1, 0.2 mm at a 2 mm median start offset (2,048 each);
0.04 and 0.07 mm are the plan's validity check (50 um -10 / +20 um: success must change smoothly).

python scripts/eval/usbc_capture.py --run E-usbc-proprio-currsph-s0 --tag sph_s0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.analysis import stats as S  # noqa: E402

OUT = ROOT / "results" / "E" / "capture"
CF = "blind_insertion/configs/phase_e_usbc_spheres.json"


def ev(run, out, clearance, halfwidth, episodes, extra=()):
    if out.exists():
        return
    cmd = [str(ROOT / ".venv/bin/python"), "-m", "blind_insertion.eval", "--policy", f"runs/{run}/last.pt",
           "--env_file", CF, "--family", "usbc", "--n_variants", "320", "--episodes", str(episodes),
           "--clearance", clearance, "--peg_offset", str(halfwidth), "--out", str(out), *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT), "MUJOCO_GL": "egl"})
    print([ln for ln in r.stdout.splitlines() if "success" in ln][-1:] or r.stderr[-800:], flush=True)


def logistic_fit(r, y, iters=200):
    """Max-likelihood logistic regression y ~ sigmoid(a - b r) by Newton's method."""
    X = np.c_[np.ones_like(r), -r]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - p)
        Hm = -(X * (p * (1 - p))[:, None]).T @ X - 1e-9 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="E-usbc-proprio-currsph-s0")
    ap.add_argument("--tag", default="sph_s0")
    ap.add_argument("--halfwidth", type=float, default=6e-3, help="E2 start offsets: uniform per axis (m)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # E2: capture radius
    e2 = OUT / f"{a.tag}_E2.parquet"
    ev(a.run, e2, "5e-5", a.halfwidth, 8192)
    d = pd.read_parquet(e2)
    r = 1000 * d.start_dist.values
    y = (d.success > 0).values.astype(float)
    w = logistic_fit(r, y)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(r), len(r))
        wb = logistic_fit(r[i], y[i])
        boots.append(wb[0] / wb[1])
    r50 = w[0] / w[1]
    edges = np.arange(0, 1000 * a.halfwidth * 1.42 + 0.5, 0.5)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() >= 20:
            k, n = int(y[m].sum()), int(m.sum())
            lo_, hi_ = S.wilson(k, n)
            fail = d[m & (y == 0)].failure.value_counts().to_dict()
            bins.append(dict(r_lo=lo, r_hi=hi, n=n, k=k, rate=k / n, ci=(lo_, hi_), failures=fail))
    e2_res = dict(clearance_mm=0.05, episodes=int(len(d)), a=w[0], b=w[1], r50_mm=r50,
                  r50_ci=(float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))), bins=bins)
    print(f"E2: r50 = {r50:.2f} mm (95% CI {e2_res['r50_ci'][0]:.2f}-{e2_res['r50_ci'][1]:.2f}), n = {len(d)}")

    # E3: clearance sweep at a 2 mm median offset (half-width 2.614 mm)
    e3 = []
    for c in (2e-5, 4e-5, 5e-5, 7e-5, 1e-4, 2e-4):
        p = OUT / f"{a.tag}_E3_c{int(round(c * 1e6))}um.parquet"
        ev(a.run, p, f"{c}", 2.614e-3, 2048)
        dd = pd.read_parquet(p)
        k, n = int(dd.success.sum()), len(dd)
        lo_, hi_ = S.wilson(k, n)
        e3.append(dict(clearance_um=c * 1e6, n=n, k=k, rate=k / n, ci=(lo_, hi_),
                       failures=dd[dd.success == 0].failure.value_counts().to_dict(),
                       median_time_s=float(dd[dd.success > 0].time.median())))
        print(f"E3: clearance {c * 1e6:.0f} um: {k / n:.3f}")
    json.dump(dict(E2=e2_res, E3=e3), open(OUT / f"{a.tag}_capture.json", "w"), indent=1, default=float)
    print("wrote", OUT / f"{a.tag}_capture.json")


if __name__ == "__main__":
    main()
