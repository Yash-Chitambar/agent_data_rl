"""Plot a logged real-arm approach and native encoder feedback, without hardware access."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="motion run directory")
    p.add_argument("--side", choices=("right", "left"), default="right")
    p.add_argument("--joint", type=int, choices=range(1, 7), default=3)
    a = p.parse_args()
    with (a.run / "approach.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["side"] == a.side and r["q_cmd"]]
    if not rows:
        raise SystemExit("no completed approach commands")
    j = a.joint - 1
    t = np.array([float(r["t_request"]) for r in rows]); origin = t[0]
    target = np.array([json.loads(r["q_target"])[j] for r in rows])
    sent_target = np.array([json.loads(r["q_cmd"])[j] for r in rows])
    observed = np.array([json.loads(r["q"])[j] for r in rows])
    with (a.run / f"encoder_{a.side}.csv").open(newline="") as f:
        native = list(csv.DictReader(f))
    tn = np.array([float(r["host_arrival_s"]) for r in native])
    qn = np.array([float(r[f"q_{a.joint}_rad"]) for r in native])
    mask = (tn >= origin) & (tn <= t[-1] + 1.2)
    fig, (ax, err) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(t-origin, target, label="Requested target", linewidth=1.5)
    ax.plot(t-origin, sent_target, label="Safety-limited target", linewidth=1, linestyle="--")
    ax.plot(tn[mask]-origin, qn[mask], label="Native encoder arrivals", linewidth=1)
    ax.set_ylabel("Joint position (rad)"); ax.legend(loc="best"); ax.grid(alpha=.25)
    err.plot(t-origin, (observed-target)*1000, label="Observed minus requested")
    err.axhline(0, color="black", linewidth=.5); err.grid(alpha=.25)
    err.set_ylabel("Error (mrad)"); err.set_xlabel("Seconds from first approach command")
    fig.suptitle(f"{a.side} joint {a.joint}: {a.run.name}")
    fig.tight_layout(); out = a.run / f"approach_{a.side}_joint{a.joint}.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
