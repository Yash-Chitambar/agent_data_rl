"""Fit the real YAM's command latency (and gain scale) from joint step tests against the sim.

Protocol and CSV format: docs/REAL_YAM_LATENCY_BACKLASH.md. Each real log is one CSV with one row
per encoder reading: t (s, host monotonic clock), q_cmd_1..q_cmd_6 (rad, the target last SENT),
q_1..q_6 (rad, measured), and optionally t_cmd (s, send time of q_cmd; better than the read time).

python scripts/real/yam_latency_fit.py fit data/real_yam/steps_*.csv [--out results/C/real_latency.json]
python scripts/real/yam_latency_fit.py selftest      # fake "real" data from the sim, known answer
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.assets import builder as B  # noqa: E402

KP = np.array([80, 80, 80, 30, 30, 30], float)   # sim gains (builder._load_robot "yam")
KV = np.array([6, 6, 6, 2, 2, 2], float)
SCALES = [0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
CONTROL_DT = 0.020
T_RESP = 0.4                                      # s of response used in the fit


def sim_step(q0, joint, dq, scale=1.0, T=T_RESP, delay=0.0):
    """Sim response of `joint` (0-based) to a target step dq at t = delay, from rest at q0.
    Returns t (1 ms grid from 0) and (q - q0[joint]) / dq."""
    spec, _ = B.load_robot("yam", gains=(KP * scale, KV * scale))
    spec.option.timestep = 0.001
    m = spec.compile()
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 7)]
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}") for i in range(1, 7)]
    d.qpos[qadr] = q0
    d.ctrl[act] = q0
    for _ in range(200):                          # settle
        mujoco.mj_step(m, d)
    base = d.qpos[qadr[joint]]
    n = int(round(T / m.opt.timestep))
    y = np.empty(n)
    for k in range(n):
        if k * m.opt.timestep >= delay:
            d.ctrl[act[joint]] = q0[joint] + dq
        mujoco.mj_step(m, d)
        y[k] = d.qpos[qadr[joint]]
    return np.arange(1, n + 1) * m.opt.timestep, (y - base) / dq


def segments(df, min_step=0.01):
    """Yield (joint, q0, dq, t_rel, y_norm) for every commanded single-joint step in a log."""
    cmd = df[[f"q_cmd_{i}" for i in range(1, 7)]].to_numpy()
    q = df[[f"q_{i}" for i in range(1, 7)]].to_numpy()
    t = df["t"].to_numpy()
    tc = df["t_cmd"].to_numpy() if "t_cmd" in df else t
    jumps = np.abs(np.diff(cmd, axis=0))
    for r in np.nonzero(jumps.max(1) > min_step)[0]:
        j = int(jumps[r].argmax())
        dq = cmd[r + 1, j] - cmd[r, j]
        t0 = tc[r + 1]
        pre = (t > t0 - 0.05) & (t <= t0)
        post = (t > t0) & (t <= t0 + T_RESP)
        if pre.sum() < 2 or post.sum() < 10:
            continue
        base = q[pre, j].mean()
        yield j, q[pre].mean(0), dq, t[post] - t0, (q[post, j] - base) / dq


def fit(paths, out=None):
    df_all = [pd.read_csv(p) for p in paths]
    segs = [s for df in df_all for s in segments(df)]
    if not segs:
        sys.exit("no commanded steps found (need q_cmd_i columns that jump by > 0.01 rad)")
    res = {}
    for j in sorted({s[0] for s in segs}):
        S = [s for s in segs if s[0] == j]
        q0 = np.mean([s[1] for s in S], 0)
        dq = float(np.median([abs(s[2]) for s in S]))
        t_all = np.concatenate([s[3] for s in S])
        y_all = np.concatenate([s[4] for s in S])
        best = (np.inf, None, None)
        for sc in SCALES:
            ts, ys = sim_step(q0, j, dq, sc)
            for tau in np.arange(0, 0.1005, 0.001):
                yp = np.interp(t_all - tau, ts, ys, left=0.0)
                err = float(np.mean((y_all - yp) ** 2))
                if err < best[0]:
                    best = (err, sc, tau)
        err, sc, tau = best
        y_mean_late = float(np.mean(y_all[t_all > 0.3]))
        res[f"joint{j + 1}"] = dict(n_steps=len(S), step_rad=dq, delay_ms=round(tau * 1e3, 1),
                                    delay_control_steps=round(tau / CONTROL_DT, 2), gain_scale=sc,
                                    rms_fit=round(err ** 0.5, 4), final_fraction=round(y_mean_late, 3))
        print(f"joint{j + 1}: {len(S)} steps, delay {tau * 1e3:.0f} ms "
              f"({tau / CONTROL_DT:.2f} control steps), gain scale {sc}, rms {err ** 0.5:.3f}")
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def selftest():
    """Fake 'real' logs from the sim with a 23 ms delay and 0.7x gains, read at 250 Hz with noise."""
    rng = np.random.default_rng(0)
    q0 = np.array([0.0, 1.96, 1.29, -0.9, 0.0, 0.0])
    rows = []
    t = 0.0
    for j in (0, 3):
        for sgn in (1, -1):
            ts, ys = sim_step(q0, j, 0.04 * sgn, 0.7, T=0.6, delay=0.2 + 0.023)
            for k in range(0, len(ts), 4):        # 250 Hz reads
                cmd = q0.copy()
                if ts[k] >= 0.2:
                    cmd[j] += 0.04 * sgn
                q = q0.copy()
                q[j] += 0.04 * sgn * ys[k] + rng.normal(0, 1e-4)
                rows.append([t + ts[k], *cmd, *q])
            t += 1.0
    cols = ["t", *[f"q_cmd_{i}" for i in range(1, 7)], *[f"q_{i}" for i in range(1, 7)]]
    p = Path(tempfile.gettempdir()) / "real_yam_selftest.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    res = fit([p])
    for k, r in res.items():
        # reads every 4 ms, so the step shows at the first read after the send: up to +4 ms
        assert 20 <= r["delay_ms"] <= 28 and r["gain_scale"] == 0.7, (k, r)
    print("selftest OK (true delay 23 ms, scale 0.7)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("csv", nargs="+")
    f.add_argument("--out", default="")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "fit":
        paths = [p for g in a.csv for p in sorted(glob.glob(g))]
        fit(paths, a.out or None)
    else:
        selftest()


if __name__ == "__main__":
    main()
